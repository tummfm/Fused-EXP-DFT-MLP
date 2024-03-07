import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as onp

import jax
import jax.numpy as np

import haiku as hk

import optax

import pickle

import matplotlib.pyplot as plt

from jax import lax
from jax import random
from jax.api import jit, grad
from jax import tree_util
# config.update("jax_enable_x64", True) # very slow!

from jax_md import space, simulate, partition

from chemtrain.jax_md_mod import custom_energy
from chemtrain.jax_md_mod import custom_nn
from chemtrain import jax_md_mod as customn_quantity

from functools import partial

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")

#--------------------------- PLOT_FNS -----------------------------------------#

def plot_convergence_history(test_error):
    plt.figure(facecolor="w")
    plt.semilogy(test_error)
    plt.xlim([0, epochs])
    plt.xlabel("Parameter Updates [-]")
    plt.ylabel("Loss $\mathcal{L}$ [-]")
    plt.tight_layout()
    plt.savefig(f"results/relative_entropy/{folder}/plots/convergence_plot.pdf", bbox_inches="tight")
    plt.show()

def plot_rdf(rdf):
    plt.figure(facecolor="w")
    plt.plot(np.linspace(0, 1, num=300), rdf, label="Learned")
    plt.plot(np.linspace(0, 1, num=300), reference_rdf, label="Reference")
    plt.ylabel("RDF [-]")
    plt.xlabel("Distance $r$ [nm]")
    plt.legend()
    plt.savefig(f"results/relative_entropy/{folder}/plots/rdf_plot.pdf", bbox_inches="tight")
    plt.show()

def plot_adf(adf):
    plt.figure(facecolor="w")
    plt.plot(np.linspace(0, 3.14, num=150), adf, label="Learned")
    plt.plot(np.linspace(0, 3.14, num=150), reference_adf, label="Reference")
    plt.ylabel("ADF [-]")
    plt.xlabel("Angle $\\alpha$ [rad]")
    plt.legend()
    plt.savefig(f"results/relative_entropy/{folder}/plots/adf_plot.pdf", bbox_inches="tight")
    plt.show()


# ------------------------------ SIM FUNCTIONS  ------------------------------ #

# computes a single printout based on apply_fn
def get_generate_single_printout(apply_fn, neighbor_fn, steps_per_printout):

    def single_simulation_step(state_and_neighbors, i):
        state, neighbors = state_and_neighbors
        new_state = apply_fn(state, neighbor=neighbors)
        new_neighbors = neighbor_fn(new_state.position, neighbors)
        return (new_state, new_neighbors), i


    @jit
    def generate_single_printout(state_and_neighbors, i):
        new_state_and_neighbors, _ = lax.scan(single_simulation_step,
                                              state_and_neighbors,
                                              xs=np.arange(steps_per_printout))

        new_state, new_neighbors = new_state_and_neighbors
        
        return new_state_and_neighbors, new_state_and_neighbors

    return generate_single_printout

# computes a number of printouts after some equilibration time
def get_generate_printouts(apply_fn, 
                           neighbor_fn, 
                           steps_per_printout, 
                           number_of_printouts,
                           dumped_printouts):

    generate_single_printout = get_generate_single_printout(apply_fn,
                                                            neighbor_fn,
                                                            steps_per_printout)

    @jit
    def generate_printouts(init_state_and_neighbors):

        # equilibrate
        new_state_and_neighbors, _ = lax.scan(generate_single_printout,
                                              init_state_and_neighbors,
                                              xs=np.arange(dumped_printouts))


        # compute printouts
        final_state_and_neighbors, states_and_neighbors = lax.scan(generate_single_printout,
                                                                   new_state_and_neighbors,
                                                                   xs=np.arange(number_of_printouts))

        return final_state_and_neighbors, states_and_neighbors

    return generate_printouts

# ------------------------------ LOAD DATA ----------------------------------- #

# more than 100 samples should not be needed
R = np.array(onp.load("samples/conf_SPC_1k.npy"))
F = np.array(onp.load("samples/forces_SPC_1k.npy"))

R_test = np.array(onp.load("samples/conf_SPC_test.npy"))
F_test = np.array(onp.load("samples/forces_SPC_test.npy"))

dataset_size = R.shape[0]

R_init = R[0]

folder = "D" # folder in which plots and data are saved

#--------------------------- NEIGHBOR_FN --------------------------------------#

box_size = np.array([3.0, 3.0, 3.0])
r_cutoff = 0.5

displacement_fn, shift_fn = space.periodic(box_size)

neighbor_fn = partition.neighbor_list(displacement_fn, 
                                    box_size, 
                                    r_cutoff, 
                                    dr_threshold=0.05,
                                    capacity_multiplier=2.0)

neighbors_init = neighbor_fn(R_init, extra_capacity=0)

#--------------------------- ENERGY_FN ----------------------------------------#

mlp_init = {
    "w_init": custom_nn.OrthogonalVarianceScalingInit(scale=1.0),
    "b_init": hk.initializers.Constant(0.0)}

prior_kwargs = {
    "sigma": 0.3,
    "epsilon": 1.0,
    "exp": 12}

init_fn, energy_fn_GNN = custom_energy.dimenetpp_neighborlist(
    displacement_fn,
    R_init,
    neighbors_init,
    r_cutoff,
    embed_size=32,
    n_interaction_blocks=4,
    num_residual_before_skip=1,
    num_residual_after_skip=2,
    out_embed_size=None,
    type_embed_size=None,
    angle_int_embed_size=None,
    basis_int_embed_size=8,
    num_dense_out=3,
    num_rbf=6,
    num_sbf=7,
    activation=jax.nn.swish,
    envelope_p=6,
    init_kwargs=mlp_init,
    prior_potential=custom_energy.generic_repulsion,
    prior_kwargs=prior_kwargs,
    onset_prior_ratio=0.9,
    max_angle_multiplier=1.25,
    max_edge_multiplier=1.25)

key = random.PRNGKey(0)

params = init_fn(key, R_init, neighbor=neighbors_init)

@jit
def energy_fn(params, R, **kwargs):
    neighbors = neighbor_fn(R, neighbors_init)
    return energy_fn_GNN(params, R, neighbors)

def energy_fn_temp(params):
    energy = partial(energy_fn_GNN, params)
    return energy

#--------------------------- LOSS_FN ------------------------------------------#

grad_fn_R = grad(energy_fn, argnums=1)
grad_fn_params = grad(energy_fn, argnums=0)

force_fn = lambda params, R: -grad_fn_R(params, R)

@jit
def force_loss(params, R, F):
    dforces = lax.map(partial(force_fn, params), R) - F
    return np.sqrt(np.mean(dforces.flatten()**2))

@jit 
def entropy_loss(params, R_AA, R_CG):

    # R_AA and R_CG have shape (n, 905, 3)

    grad_potential_AA = lax.map(partial(grad_fn_params, params), R_AA)
    grad_potential_CG = lax.map(partial(grad_fn_params, params), R_CG)

    mean_AA = tree_util.tree_map(partial(np.mean, axis=0), grad_potential_AA)
    mean_CG = tree_util.tree_map(partial(np.mean, axis=0), grad_potential_CG)

    beta = 1 / kT

    return tree_util.tree_multimap(
        lambda x, y: beta * (x - y),
        mean_AA,
        mean_CG
    )

#--------------------------- SETUP SIMULATION ---------------------------------#

mass = 18.0154 # g/mol

system_temperature = 296.15 # K
boltzmann_constant = 0.0083145107 # kJ / mol K
kT = system_temperature * boltzmann_constant

dt = 0.002 # ps

print_every = 0.1 # ps
steps_per_printout = int(print_every / dt)

equilibration_time = 5.0 # ps
dumped_printouts = int(equilibration_time / print_every)

nose_hoover_simulator = partial(simulate.nvt_nose_hoover,
                                shift_fn=shift_fn,
                                dt=dt,
                                kT=kT,
                                chain_length=3)

#--------------------------- RDF_FN -------------------------------------------#

reference_rdf = np.array(onp.loadtxt("data/SPC_FW_RDF.csv"))
rdf_discretization_params = customn_quantity.rdf_discretization(RDF_cut=1.0,
                                                                nbins=300,
                                                                RDF_start=0.0)
rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization_params
rdf_params = customn_quantity.RDFParams(reference_rdf=reference_rdf,
                                        rdf_bin_centers=rdf_bin_centers,
                                        rdf_bin_boundaries=rdf_bin_boundaries,
                                        sigma_RDF=sigma_RDF)

rdf_fn = customn_quantity.initialize_radial_distribution_fun(box_size,
                                                             displacement_fn,
                                                             rdf_params)

#--------------------------- ADF_FN -------------------------------------------#

reference_adf = np.array(onp.loadtxt("data/SPC_FW_ADF.csv"))
adf_discretization_params = customn_quantity.adf_discretization(nbins_theta=150)
adf_bin_centers, sigma_ADF = adf_discretization_params
adf_params = customn_quantity.ADFParams(reference_adf=reference_adf,
                                        adf_bin_centers=adf_bin_centers,
                                        sigma_ADF=sigma_ADF,
                                        r_outer=0.318,
                                        r_inner=0.0)
adf_fn = customn_quantity.initialize_angle_distribution_neighborlist(displacement_fn,
                                                                     adf_params,
                                                                     smoothing_dr=0.01,
                                                                     R_init=R_init,
                                                                     nbrs_init=neighbors_init)

#--------------------------- PRESSURE_FN --------------------------------------#

pressure_fn = customn_quantity.init_pressure(energy_fn_temp, box_size)

#--------------------------- OPTIMIZER ----------------------------------------#

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jit
def update_params(params, opt_state, R_AA, R_CG):
    updates, opt_state = optimizer.update(
        entropy_loss(params, R_AA, R_CG),
        opt_state
    )
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

#--------------------------- MAIN LOOP ----------------------------------------#

number_of_printouts = 200
print(f"Number of Printouts: {number_of_printouts}")

epochs = 200
print(f"Epochs: {epochs}")

test_error = []

print_RDF_ADF_at = []
# empty dicts that save interrim rdfs/adfs
saved_rdfs = {}
saved_adfs = {}

for iteration in range(epochs):

    print(f"{iteration + 1}. Epoch")

    test_error += [float(force_loss(params, R_test, F_test))]
    plot_convergence_history(test_error)
    onp.savetxt(f"results/relative_entropy/{folder}/data/error.csv", test_error)

    # setup MD function with current params
    sim_energy_fn = partial(energy_fn, params)
    sim_init_fn, sim_apply_fn = nose_hoover_simulator(sim_energy_fn)
    init_sim_state = sim_init_fn(key,
                                 R_init,
                                 mass=mass,
                                 neighbors=neighbors_init)
    generate_printouts = get_generate_printouts(sim_apply_fn,
                                                neighbor_fn,
                                                steps_per_printout,
                                                number_of_printouts,
                                                dumped_printouts)

    # compute MD simulation 
    printouts = generate_printouts((init_sim_state, neighbors_init))

    current_state_and_neighbors, states_and_neighbors = printouts
    states, neighbors = states_and_neighbors

    if iteration + 1 in print_RDF_ADF_at:

        # new printout function with different number_of_printouts
        generate_printouts = get_generate_printouts(sim_apply_fn,
                                                    neighbor_fn,
                                                    steps_per_printout,
                                                    500,
                                                    dumped_printouts)

        # compute MD simulation 
        printouts = generate_printouts((init_sim_state, neighbors_init))

        current_state_and_neighbors, states_and_neighbors = printouts
        states, neighbors = states_and_neighbors

        rdfs = lax.map(rdf_fn, states)
        adfs = lax.map(adf_fn, states_and_neighbors)

        rdf = np.mean(rdfs, axis=0)
        adf = np.mean(adfs, axis=0)

        saved_rdfs[iteration + 1] = rdf
        saved_adfs[iteration + 1] = adf

        with open(f"results/relative_entropy/{folder}/data/rdfs.pkl", "wb") as output:
            pickle.dump(saved_rdfs, output, pickle.HIGHEST_PROTOCOL)

        with open(f"results/relative_entropy/{folder}/data/adfs.pkl", "wb") as output:
            pickle.dump(saved_adfs, output, pickle.HIGHEST_PROTOCOL)

    R_CG = states.position
    R_AA = random.permutation(key, R)[:number_of_printouts]

    params, opt_state = update_params(params, opt_state, R_AA, R_CG)

    with open(f"results/relative_entropy/{folder}/data/params.pkl", "wb") as output:
        pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)

#--------------------------- COMPUTE RDF/ADF/PRESSURE -------------------------#

sys.exit()

print("Computing RDF/ADF/PRESSURE")

number_of_printouts = 500

sim_energy_fn = partial(energy_fn, params)
sim_init_fn, sim_apply_fn = nose_hoover_simulator(sim_energy_fn)
init_sim_state = sim_init_fn(key, R_init, mass=mass, neighbors=neighbors_init)
generate_printouts = get_generate_printouts(sim_apply_fn,
                                            neighbor_fn,
                                            steps_per_printout,
                                            number_of_printouts,
                                            dumped_printouts)

printouts = generate_printouts((init_sim_state, neighbors_init))
final_state_and_neighbors, states_and_neighbors = printouts
states, neighbors = states_and_neighbors

rdfs = lax.map(rdf_fn, states)
adfs = lax.map(adf_fn, states_and_neighbors)

rdf = np.mean(rdfs, axis=0)
adf = np.mean(adfs, axis=0)

pressure_fn_ = lambda x, y: pressure_fn(y, x) # switch order of arguments
pressure_list = lax.map(partial(pressure_fn_, params), states_and_neighbors)
p = np.mean(pressure_list)

onp.savetxt(f"results/relative_entropy/{folder}/data/rdf.csv", rdf)
onp.savetxt(f"results/relative_entropy/{folder}/data/adf.csv", adf)
onp.savetxt(f"results/relative_entropy/{folder}/data/pressure.csv", p)

plot_rdf(rdf)
plot_adf(adf)
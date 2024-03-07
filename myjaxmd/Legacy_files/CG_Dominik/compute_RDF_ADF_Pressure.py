import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as onp

import jax
import jax.numpy as np

import haiku as hk

import pickle

import matplotlib.pyplot as plt

from jax import lax
from jax import random
from jax.api import jit
# config.update("jax_enable_x64", True) # very slow!

from jax_md import space, simulate, partition

from chemtrain.jax_md_mod import custom_energy, custom_space
from chemtrain.jax_md_mod import custom_nn
from chemtrain import jax_md_mod as customn_quantity

from functools import partial

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")

#--------------------------- PLOT_FNS -----------------------------------------#

def plot_rdf(rdf):
    plt.figure(facecolor="w")
    plt.plot(np.linspace(0, 1, num=300), rdf, label="Learned")
    plt.plot(np.linspace(0, 1, num=300), reference_rdf, label="Reference")
    plt.ylabel("RDF [-]")
    plt.xlabel("Distance $r$ [nm]")
    plt.legend()
    plt.savefig(f"results/{method}/{folder}/plots/rdf_plot.pdf")
    plt.show()

def plot_adf(adf):
    plt.figure(facecolor="w")
    plt.plot(np.linspace(0, 3.14, num=150), adf, label="Learned")
    plt.plot(np.linspace(0, 3.14, num=150), reference_adf, label="Reference")
    plt.ylabel("ADF [-]")
    plt.xlabel("Angle $\phi$ [?]")
    plt.legend()
    plt.savefig(f"results/{method}/{folder}/plots/adf_plot.pdf")
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
                           dumped_printouts,
                           equilibrate=True):

    generate_single_printout = get_generate_single_printout(apply_fn,
                                                            neighbor_fn,
                                                            steps_per_printout)

    @jit
    def generate_printouts(init_state_and_neighbors):

        if equilibrate:
        # equilibrate
            new_state_and_neighbors, _ = lax.scan(generate_single_printout,
                                                    init_state_and_neighbors,
                                                    xs=np.arange(dumped_printouts))
        else:
            new_state_and_neighbors = init_state_and_neighbors


        # compute printouts
        final_state_and_neighbors, states_and_neighbors = lax.scan(generate_single_printout,
                                                                   new_state_and_neighbors,
                                                                   xs=np.arange(number_of_printouts))

        return final_state_and_neighbors, states_and_neighbors

    return generate_printouts

# ------------------------------ LOAD DATA ----------------------------------- #
method = "RDF_matching"
folder = "B"
print(f"Method: {method}    Version: {folder}")

# more than 100 samples should not be needed
R = np.array(onp.load("samples/conf_SPC_test.npy"))
F = np.array(onp.load("samples/forces_SPC_test.npy"))

dataset_size = R.shape[0]

with open(f"results/{method}/{folder}/data/params.pkl", "rb") as input:
    params = pickle.load(input)

R_init = R[0]

#--------------------------- NEIGHBOR_FN --------------------------------------#

box_size = np.array([3.0, 3.0, 3.0])
r_cutoff = 0.5

fractional = True
print(f"fractional: {fractional}")

if fractional:
    R_scaled, box_size = custom_space.scale_to_fractional_coordinates(R_init, box_size)
    displacement_fn, shift_fn = custom_space.periodic_general(box_size, fractional_coordinates=True)
    R_init = R_scaled
    neighbor_fn = partition.neighbor_list(displacement_fn, 
                                        np.ones(3), 
                                        r_cutoff, 
                                        dr_threshold=0.05,
                                        capacity_multiplier=2.0)

else:
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

if method == "RDF_matching":
    prior_kwargs = {
        "sigma": 0.3165, 
        "epsilon": 1.0,
        "exp": 12}
else:
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

@jit
def energy_fn(params, R, **kwargs):
    neighbors = neighbor_fn(R, neighbors_init)
    return energy_fn_GNN(params, R, neighbors)

def energy_fn_temp(params):
    energy = partial(energy_fn_GNN, params)
    return energy

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

#--------------------------- COMPUTE RDF/ADF/PRESSURE -------------------------#

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

final_state, final_neighbors = final_state_and_neighbors

p_ref = 1 / 16.6054
pressure_fn_ = lambda x, y: pressure_fn(y, x) # switch order of arguments for partial
pressure_list = lax.map(partial(pressure_fn_, params), states_and_neighbors)
p = np.mean(pressure_list)
print(f"Reference Pressure: {p_ref}")
print(f"Predicted Pressure: {p}")

# save results
pressure_output = open(f"results/{method}/{folder}/data/pressure.out", "w")
pressure_output.write(str(float(p)))
pressure_output.close()
onp.savetxt(f"results/{method}/{folder}/data/rdf.csv", rdf)
onp.savetxt(f"results/{method}/{folder}/data/adf.csv", adf)

plot_rdf(rdf)
plot_adf(adf)                                                                    

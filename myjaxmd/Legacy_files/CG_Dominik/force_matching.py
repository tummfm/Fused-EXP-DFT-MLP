import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as onp

import jax
import jax.numpy as np

import haiku as hk

import optax

import time

import pickle

import matplotlib.pyplot as plt

from jax import lax
from jax import random
from jax.api import jit, grad
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

def plot_convergence_history(train_error, val_error):
    plt.figure(facecolor="w")
    plt.semilogy(train_error, label="Training")
    plt.semilogy(val_error, label="Validation")
    plt.xlim([0, epochs * (R_train.shape[0] / update_gradient_every)])
    plt.xlabel("Parameter Updates [-]")
    plt.ylabel("Loss $\mathcal{L}$ [-]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/force_matching/{folder}/plots/convergence_plot.pdf", bbox_inches="tight")
    plt.show()

def plot_rdf(rdf):
    plt.figure(facecolor="w")
    plt.plot(np.linspace(0, 1, num=300), rdf, label="Learned")
    plt.plot(np.linspace(0, 1, num=300), reference_rdf, label="Reference")
    plt.ylabel("RDF [-]")
    plt.xlabel("Distance $r$ [nm]")
    plt.legend()
    plt.savefig(f"results/force_matching/{folder}/plots/rdf_plot.pdf")
    plt.show()

def plot_adf(adf):
    plt.figure(facecolor="w")
    plt.plot(np.linspace(0, 3.14, num=150), adf, label="Learned")
    plt.plot(np.linspace(0, 3.14, num=150), reference_adf, label="Reference")
    plt.ylabel("ADF [-]")
    plt.xlabel("Angle $\phi$ [?]")
    plt.legend()
    plt.savefig(f"results/force_matching/{folder}/plots/adf_plot.pdf")
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

global_start_time = time.time()
# ------------------------------ LOAD DATA ----------------------------------- #

position_data = np.array(onp.load("samples/conf_SPC_1k.npy"))
force_data = np.array(onp.load("samples/forces_SPC_1k.npy"))

percentage_training = 0.7
dataset_size = position_data.shape[0]

print(f"Dataset size: {dataset_size}")

R_train, R_val = np.split(position_data, [percentage_training * dataset_size])
F_train, F_val = np.split(force_data, [percentage_training * dataset_size])

R_init = R_train[0]

folder = "K" # folder in which plots and data are saved

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

grad_fn = grad(energy_fn, argnums=1)

force_fn = lambda params, R: -grad_fn(params, R)

# @jit
# def force_loss(params, R, F):
#     # use of lax.map instead of vmap to avoid out of memory
#     start_time = time.time()
#     dforces = lax.map(partial(force_fn, params), R) - F
#     return np.sqrt(np.mean(np.sum(dforces ** 2, axis=(1,2))))

@jit
def force_loss(params, R, F):
    dforces = lax.map(partial(force_fn, params), R) - F
    return np.sqrt(np.mean(dforces.flatten()**2))

# computes loss for training and validation datsets
def compute_losses(params):
    train_error = force_loss(params, R_train, F_train)
    val_error = force_loss(params, R_val, F_val)
    return train_error, val_error

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

# batch size larger than one usually leads to out of memory error
# instead we use gradient accumulation with the MultiSteps class
update_gradient_every = 100
print(f"Update Gradient every: {update_gradient_every}")

optimizer = optax.MultiSteps(optax.adam(1e-3), update_gradient_every, True)
opt_state = optimizer.init(params)

@jit
def update_params(params, opt_state, R, F):
    updates, opt_state = optimizer.update(grad(force_loss)(params, R, F),
                                          opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

@jit
def update_epoch(params_and_opt_state, batches):

    def update_batch(params_and_opt_state, batch):
        params, opt_state = params_and_opt_state
        R, F = batch

        params, opt_state = update_params(params, opt_state, R, F)

        # NOTE only compute errors when gradient is actually updated. However, 
        # we need to return some value (0.0) every iteration. These get later 
        # removed
        # NOTE if gradient was updated mini_steps resets to zero
        train_error, val_error = lax.cond(opt_state.mini_step == 0 ,
                                          compute_losses,
                                          lambda *args: (0.0, 0.0),
                                          params)

        return (params, opt_state), (train_error, val_error)

    return lax.scan(update_batch, params_and_opt_state, batches)

#--------------------------- BATCHING -----------------------------------------#

# batch size larger than one usually leads to out of memory error
# instead we use gradient accumulation (see OPTIMIZER)
batch_size = 1
print(f"Batch size: {batch_size}")

# shuffle array with index list so that R and F are both shuffled identically
index_list = onp.arange(int(dataset_size * percentage_training))
onp.random.shuffle(index_list)
 
# using jit with more than 2k samples get process killed by linux OOM killer
@jit
def make_batches(index_list):

    batch_Rs = []
    batch_Fs = []

    for i in range(0, dataset_size, batch_size):

        if i + batch_size > R_train.shape[0]:
            break # avoids creation of arrays with unequal shapes

        indices = np.array(index_list[i:i + batch_size])

        batch_Rs.append(R_train[indices])
        batch_Fs.append(F_train[indices])

    return np.stack(batch_Rs), np.stack(batch_Fs)

batch_Rs, batch_Fs = make_batches(index_list)

#--------------------------- MAIN LOOP ----------------------------------------#

epochs = 150
print(f"Epochs: {epochs}")

# initial error of prior
train_error = np.array([force_loss(params, R_train, F_train)])
val_error = np.array([force_loss(params, R_val, F_val)])

print("Start")

for iteration in range(epochs):

    print(f"{iteration + 1}. Epoch")

    start_time = time.time()
    params_and_opt_state, errors = update_epoch((params, opt_state), 
                                               (batch_Rs, batch_Fs))
    print(f"updated after {time.time() - start_time}s")

    # add new errors to current error array and remove empty values
    train_error = np.concatenate(
        (train_error, onp.delete(errors[0], np.where(errors[0]==0.0))))
    val_error = np.concatenate(
        (val_error, onp.delete(errors[1], np.where(errors[1]==0.0))))

    plot_convergence_history(train_error, val_error)
    onp.savetxt(f"results/force_matching/{folder}/data/train_error.csv", 
                train_error)
    onp.savetxt(f"results/force_matching/{folder}/data/val_error.csv", 
                val_error)

    # save current params
    params, opt_state = params_and_opt_state
    with open(f"results/force_matching/{folder}/data/params.pkl", "wb") as output:
        pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)

    # shuffle and create new batches
    onp.random.shuffle(index_list)
    batch_Rs, batch_Fs = make_batches(index_list)

#--------------------------- COMPUTE RDF/ADF/PRESSURE -------------------------#
print(f"Overall Duration: {time.time() - global_start_time}s")
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

p_ref = 1 / 16.6054
pressure_fn_ = lambda x, y: pressure_fn(y, x) # switch order of arguments for partial
pressure_list = lax.map(partial(pressure_fn_, params), states_and_neighbors)
p = np.mean(pressure_list)
print(f"Reference Pressure: {p_ref}")
print(f"Predicted Pressure: {p}")

pressure_output = open(f"results/force_matching/{folder}/data/pressure.out", "w")
pressure_output.write(str(float(p)))
pressure_output.close()
onp.savetxt(f"results/force_matching/{folder}/data/rdf.csv", rdf)
onp.savetxt(f"results/force_matching/{folder}/data/adf.csv", adf)

plot_rdf(rdf)
plot_adf(adf)

print(f"Overall Duration: {time.time() - global_start_time}s")
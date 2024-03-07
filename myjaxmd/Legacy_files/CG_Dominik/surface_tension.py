import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as onp

import jax
import jax.numpy as np

import haiku as hk

import pickle

import Initialization

from jax import lax
from jax import random
from jax.api import jit
# config.update("jax_enable_x64", True) # very slow!

from jax_md import space, simulate, partition

from chemtrain.jax_md_mod import custom_energy
from chemtrain.jax_md_mod import custom_nn

from functools import partial

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")

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

# load inital state and shift in the middle of a 3nm x 3nm x 9nm box
R_init = onp.load("samples/conf_SPC_test.npy")[0]
R_init[:, 2] += 3.0
R_init = np.array(R_init)

folder = "B" # folder in which plots and data are saved
method = "RDF_matching"
print(f"Method: {method}    Version: {folder}") 


#--------------------------- NEIGHBOR_FN --------------------------------------#

box_size = np.array([3.0, 3.0, 9.0])
r_cutoff = 0.5

displacement_fn, shift_fn = space.periodic(box_size)

neighbor_fn = partition.neighbor_list(displacement_fn, 
                                      box_size, 
                                      r_cutoff, 
                                      dr_threshold=0.05,
                                      capacity_multiplier=2.0)

neighbors_init = neighbor_fn(R_init, extra_capacity=0)

#--------------------------- ENERGY_FN ----------------------------------------#

key = random.PRNGKey(0)

if method == "tabulated":
    r_cutoff = 0.9
    tabulated_fns = Initialization.select_model("Tabulated", 
                                                R_init, 
                                                displacement_fn,
                                                box_size,
                                                key)
    energy_fn, neighbor_fn, params, neighbors_init = tabulated_fns
    sim_energy_fn  = energy_fn(params)

else:
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

    with open(f"results/{method}/{folder}/data/params.pkl", "rb") as input:
            params = pickle.load(input)

    @jit
    def energy_fn(params, R, **kwargs):
        neighbor = neighbor_fn(R, neighbors_init)
        return energy_fn_GNN(params, R, neighbor)

    sim_energy_fn = partial(energy_fn, params)

#--------------------------- SETUP SIMULATION ---------------------------------#

mass = 18.0154 # g/mol

system_temperature = 296.15 # K
boltzmann_constant = 0.0083145107 # kJ / mol K
kT = system_temperature * boltzmann_constant

dt = 0.002 # ps

print_every = 1.0 # ps
steps_per_printout = int(print_every / dt)

equilibration_time = 0.0 # ps
dumped_printouts = int(equilibration_time / print_every)

nose_hoover_simulator = partial(simulate.nvt_nose_hoover,
                                shift_fn=shift_fn,
                                dt=dt,
                                kT=kT,
                                chain_length=3)

#--------------------------- RUN SIMULATION -----------------------------------#

number_of_printouts = 5

sim_init_fn, sim_apply_fn = nose_hoover_simulator(sim_energy_fn)
init_sim_state = sim_init_fn(key, R_init, mass=mass, neighbor=neighbors_init)
generate_printouts = get_generate_printouts(sim_apply_fn,
                                            neighbor_fn,
                                            steps_per_printout,
                                            number_of_printouts,
                                            dumped_printouts)

printouts = generate_printouts((init_sim_state, neighbors_init))
final_state_and_neighbors, states_and_neighbors = printouts
states, neighbors = states_and_neighbors

for i, state in enumerate(states.position):
    
    np.save(f"results/{method}/{folder}/data/R_{i+1}ps.npy", state)

print(f"Finished surface tension calculation for {method}!")


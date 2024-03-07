from jax import random, lax, vmap, jit, ops, checkpoint
import jax.numpy as jnp
from jax.experimental import optimizers
from jax_md import energy, simulate, partition

import chemtrain.traj_util
from chemtrain.jax_md_mod import io, custom_space, custom_energy
from chemtrain.jax_md_mod import custom_quantity, custom_simulator
from chemtrain import difftre
from functools import partial
import time
import matplotlib.pyplot as plt


# plot functions
def plot_rdf_and_potential(rdf_bin_centers, predicted_rdf, target_rdf, pot_r_vals, predicted_pot, target_pot,
                           loss_history=None):
    if loss_history is None:
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(2. * 6.4, 4.8))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3. * 6.4, 4.8))
        step = len(loss_history) - 1
        ax1.plot(loss_history)
        fig.suptitle('Optimization state at step ' + str(step))

    ax2.plot(rdf_bin_centers, predicted_rdf, label='Initial guess')
    ax2.plot(rdf_bin_centers, target_rdf, label='LJ Target')
    ax2.set_xlabel('$r$ in nm')
    ax2.set_ylabel('$g(r)$')
    ax2.legend()
    ax3.plot(pot_r_vals, predicted_pot, label='Initial guess')
    ax3.plot(pot_r_vals, target_pot, label='LJ data-generating potential')
    ax3.legend()
    ax3.set_xlim([0, 1])
    ax3.set_ylim([-1.5, 3])
    ax3.set_xlabel('$r$ in nm')
    ax3.set_ylabel('$U$ in kJ/mol')
    plt.show()


seed = 0
kbT = 2.49435321
mass = 1.
time_step = 0.002
input_file = '../../examples/data/confs/Water_experimental_3nm.gro'

total_time = 50.
t_equilib = 10.  # equilibration time before sampling RDF
print_every = 0.1
timing_struct_LJ = chemtrain.traj_util.process_printouts(time_step, total_time, t_equilib, print_every)

R_init, box = io.load_box(input_file)
N = R_init.shape[0]
print(N, 'Lennard-Jones particles in cubic box of size', box[0], 'nm')

displacement, shift = custom_space.differentiable_periodic(box, wrapped=True)

key = random.PRNGKey(seed)
model_init_key, simuation_init_key = random.split(key, 2)

# define LJ reference model
sigma = 0.25  # in nm
epsilon = 1.  # in kJ / mol
# energy is cut-off at 3. * sigma = 0.9 nm
LJ_neighbor_fn, LJ_energy = energy.lennard_jones_neighbor_list(displacement, box, sigma=sigma, epsilon=epsilon,
                                                        r_onset=2.5, r_cutoff=3.)
LJ_nbrs_init = LJ_neighbor_fn(R_init)

nose_hoover_simulator = partial(simulate.nvt_nose_hoover, shift_fn=shift, dt=time_step, kT=kbT, chain_steps=1)

LJ_init, LJ_apply_fn = nose_hoover_simulator(LJ_energy)
LJ_state = LJ_init(simuation_init_key, R_init, mass=mass, neighbor=LJ_nbrs_init)
LJ_sim_state = (LJ_state, LJ_nbrs_init)

run_small_simulation = custom_simulator.run_to_next_printout_fn_neighbors(LJ_apply_fn, LJ_neighbor_fn,
                                                                          timing_struct_LJ.timesteps_per_printout)

t_start = time.time()
LJ_sim_state, _ = lax.scan(run_small_simulation, LJ_sim_state, xs=jnp.arange(timing_struct_LJ.num_dumped))  # equilibrate
LJ_sim_state, traj = lax.scan(run_small_simulation, LJ_sim_state, xs=jnp.arange(timing_struct_LJ.num_printouts_production))
print('Time for generating reference dataset:', time.time() - t_start, 's')

# compute RDF:

LJ_traj_state = (LJ_sim_state, traj, 0)
rdf_struct = custom_quantity.init_rdf_params(1.)
rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box)
LJ_quantity_traj = chemtrain.traj_util.quantity_traj(LJ_traj_state, {'rdf': rdf_fn})
target_rdf = jnp.mean(LJ_quantity_traj['rdf'], axis=0)


# once target quantity is available we can train potential

r_cut = 1.
spline_xvals = jnp.linspace(0., r_cut, 100)

# random initialisation
init_params = 0.1 * random.normal(model_init_key, spline_xvals.shape)

tabulated_energy = partial(custom_energy.tabulated_neighbor_list, displacement, spline_xvals,
                           box_size=box, r_onset=(r_cut - 0.1), r_cutoff=r_cut,
                           dr_threshold=0.2, capacity_multiplier=1.25)
repulsion_prior = partial(custom_energy.generic_repulsion_neighborlist, displacement,
                          box_size=box, r_onset=(r_cut - 0.1), r_cutoff=r_cut)
neighbor_fn, _ = tabulated_energy(init_params)
nbrs_init = neighbor_fn(R_init, extra_capacity=0)

# for gradient visual: 0.6, 0.4
# alternative for initial guess: 0.2, 0.5
def energy_fn_template(energy_params, sigma=0.3, epsilon=1., exp=6):
    spline_energy = tabulated_energy(energy_params, initialize_neighbor_list=False)
    prior_energy = repulsion_prior(sigma=sigma, epsilon=epsilon, exp=exp, initialize_neighbor_list=False)

    @jit
    def combined_energy(R, neighbor):
        return spline_energy(R, neighbor) + prior_energy(R, neighbor)

    return combined_energy


initial_energy_fn = energy_fn_template(init_params)

init, apply = nose_hoover_simulator(initial_energy_fn)
state = init(simuation_init_key, R_init, mass=mass, neighbor=nbrs_init)
init_sim_state = (state, nbrs_init)

# test potential
# total_time = 200.
# t_equilib = 20.
total_time = 30.
t_equilib = 5.
print_every = 0.1
timing_struct = chemtrain.traj_util.process_printouts(time_step, total_time, t_equilib, print_every)

run_small_simulation = custom_simulator.run_to_next_printout_fn_neighbors(apply, neighbor_fn,
                                                                          timing_struct.timesteps_per_printout)

equilibrated_sim_state, _ = lax.scan(run_small_simulation, init_sim_state, xs=jnp.arange(timing_struct.num_dumped))
sim_state, traj = lax.scan(run_small_simulation, equilibrated_sim_state, xs=jnp.arange(timing_struct.num_printouts_production))

dummy_traj_state = (sim_state, traj, 0)
quantity_traj = chemtrain.traj_util.quantity_traj(dummy_traj_state, {'rdf': rdf_fn})
initial_rdf = jnp.mean(quantity_traj['rdf'], axis=0)



def compute_potential_over_r(energy_fn, r_vals):
    # r0 = jnp.zeros((nbins, 1, 3))
    # r1 = jnp.linspace(r_start, r_end, nbins)
    # r1_zeros = jnp.zeros((nbins, 2))

    @vmap
    def build_r_pair(r1):
        r_pair = jnp.zeros([2, 3])
        return ops.index_update(r_pair, ops.index[1, 0], r1)

    r_pairs = build_r_pair(r_vals)
    neighborlist = partition.NeighborList(jnp.array([[1], [0]]), jnp.zeros([2, 3]), False, 1, None)
    vectorized_energy_fn = vmap(partial(energy_fn, neighbor=neighborlist))
    potential_vals = vectorized_energy_fn(r_pairs)
    return potential_vals

# Potential dominated by prior, will correct towards data-generating potential via spline

pot_r_vals = jnp.linspace(0.1, 1., 100)
target_pot = compute_potential_over_r(LJ_energy, pot_r_vals)
initil_guess_pot = compute_potential_over_r(initial_energy_fn, pot_r_vals)

# ruggs in rdf not due to undersampling, but random initialisation of spline

plot_rdf_and_potential(rdf_struct.rdf_bin_centers, initial_rdf, target_rdf, pot_r_vals, initil_guess_pot, target_pot)

# Learning tabulated potential: Visualize RDF, Potential and Gradient

# Let's define the inputs for DiffTRe: A dictionary for all functions
# We need a function to compute the quantity of interest (QoI)
quantities = {'rdf': {'compute_fn': checkpoint(rdf_fn), 'target': target_rdf, 'weight': 1.}}

# store functions needed to define new simulation for different energy_params
simulation_funs = (nose_hoover_simulator, energy_fn_template, neighbor_fn)

# TODO add regularisation

gradient_and_propagation_fn, trajectory_state = difftre.difftre_init(
        simulation_funs, timing_struct, quantities, kbT, init_params, init_sim_state)

stepsize_schedule = optimizers.exponential_decay(0.1, 100, 0.1)
opt_init, opt_update, get_params = optimizers.adam(stepsize_schedule, b1=0.1, b2=0.4)

# stepsize_schedule = optimizers.exponential_decay(50., 100, 0.1)
# opt_init, opt_update, get_params = optimizers.sgd(20.)

opt_state = opt_init(init_params)  # initialize optimizer state


@jit
def update(step, params, opt_state, traj_state):
    traj_state, curr_grad, loss_val, error, predictions = gradient_and_propagation_fn(params, traj_state)
    opt_state = opt_update(step, curr_grad, opt_state)
    return get_params(opt_state), opt_state, traj_state, loss_val, error, curr_grad, predictions

#
# def potential_gradient(params, gradient, grid, eps=1.e-2):
#     """
#     A function used to visualize the effect of moving down the gradient
#     on the potential over x and y.
#     """
#
#     # TODO try
#     # U = energy_fn_template(params)
#     # pot_grad = grad(U)(gradient)
#
#     def update_params(params, gradient):
#         return params - eps * gradient
#     perturbed_params = tree_multimap(update_params, params, gradient)
#     U0 = energy_fn_template(params)
#     U1 = energy_fn_template(perturbed_params)
#
#     def pot_gradient_fn(R):
#         R = jnp.expand_dims(R, 0)
#         return (U1(R) - U0(R)) / eps
#
#     def pot_fn(R):
#         R = jnp.expand_dims(R, 0)
#         return U0(R)
#
#     vectorized_potential_gradient = vmap(vmap(pot_gradient_fn))
#     vectorized_potential = vmap(vmap(pot_fn))
#
#     potential = vectorized_potential(grid)
#     gradient = vectorized_potential_gradient(grid)
#
#     return potential, gradient

loss_history = []
params = init_params

num_updates = 100
for step in range(num_updates):

    start_time = time.time()
    params, opt_state, trajectory_state, loss_val, curr_grad, predictions = update(
        step, params, opt_state, trajectory_state)

    loss_history.append(loss_val)
    print("Step {} in {:0.2f} s. Loss: {:0.3f}".format(step, time.time() - start_time, loss_val))

    if jnp.isnan(loss_val):  # stop learning when diverged
        break

    current_pot = compute_potential_over_r(energy_fn_template(params), pot_r_vals)
    predicted_rdf = predictions['rdf']
    plot_rdf_and_potential(rdf_struct.rdf_bin_centers, predicted_rdf, target_rdf, pot_r_vals, current_pot,
                           target_pot, loss_history)

# # Prior ablation study

# purpose of prior: speed-up convergence of trained model for finite time training:
# 2 mechanisms: 1. spend less time with trajectories that can be rules out from first principles:
#                  No need to spend time on learning first principles from data, before reaching a point where
#               2. Potential values often span multiple orders of magnitude (e.g. LJ Potential rdf onset vs minimum)
#                  Large steps for some parameters necessary for tabulated potential
#                  --> not a good analogy for neural networks: Output can be scaled -->
#                  network weights remain between 0 and 1
#

# Visualisation of Training dynamics: Gradients without prior

# Loss curves for different priors


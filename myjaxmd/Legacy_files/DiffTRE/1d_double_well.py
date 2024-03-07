import os
import sys
if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs


import jax.numpy as jnp
from jax import random, vmap, jit, lax, checkpoint, value_and_grad, tree_multimap
from jax_md import simulate, space
import matplotlib.pyplot as plt
import time
from functools import partial
import optax
import pickle
from jax.scipy.stats.norm import pdf as normal_pdf
from chemtrain.jax_md_mod import custom_interpolate


# TODO rename functions --> wait till final derivations in paper --> use same notation

# Purpose of notebook: Explain DiffTRe and provide insight into dynamics of learning

# N = 2000
N = 2000  # the more particles in the simulation, the less snapshots are needed for the same smoothness of outputs
dt = 1.e-3  # 10 fs
kbT = 1.
mass = 1.
resolution = 100
box = jnp.array(1.)

def plot_1d_over_x(x, z, x_label=None, y_label=None, ylim=None):
    fig, ax = plt.subplots()
    ax.plot(x, z)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(ylim)

def plot_training_process(loss_history, x_vals, predicted_density, reference_density, cur_potential, generating_potential):
    step = len(loss_history) - 1
    middle_point = int(cur_potential.size / 2)
    shift = cur_potential[middle_point] - generating_potential[middle_point]
    shifted_potential = cur_potential - shift
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3. * 6.4, 4.8))
    ax1.plot(loss_history)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax2.plot(x_vals, predicted_density, label='Prediction at step ' + str(step))
    ax2.plot(x_vals, reference_density, label='Target', linestyle='--')
    ax2.legend()
    ax3.plot(x_vals, shifted_potential, label='Predicted Potential at step ' + str(step))
    ax3.plot(x_vals, generating_potential, label='Data-generating Potential', linestyle='--')
    ax3.set_ylim([-1, 2])
    ax3.legend()

    plt.tight_layout()
    plt.show()


def analytic_double_well(x):
    double_well = 2500 * (x - 0.5)**6 - 10 * (x - 0.55)**2
    return kbT * double_well


x_vals = jnp.linspace(0., box, resolution + 1)
vectorized_analytic_double_well = vmap(analytic_double_well)

analytic_potential_vals = vectorized_analytic_double_well(x_vals)
plot_1d_over_x(x_vals, analytic_potential_vals, x_label='$x / L$', y_label='$U / k_BT$', ylim=[-1, 2])
plt.show()


_, shift = space.periodic(box)
position_key, model_key, simuation_key = random.split(random.PRNGKey(0), 3)

# initialize particles
R_init = 0.1 * box * random.uniform(position_key, (N, 1))  # TODO back to proper


def analytic_energy_fn(R, t=0):
    """Ideal gas particles don't interact. The only energy contribution comes from the position in the box."""
    energies = vectorized_analytic_double_well(R)
    return jnp.sum(energies)


init, apply = simulate.nvt_langevin(analytic_energy_fn, shift, dt, kbT)
init_state = init(simuation_key, R_init)


def run_to_next_printout_fn(apply_fn, n_steps_per_printout):
    """Helper function to sample a specified number of snapshots."""
    do_step = lambda state, t: (apply_fn(state), ())

    @jit
    def run_small_simulation(start_state, dummy_carry):
        printout_state, _ = lax.scan(do_step, start_state, xs=jnp.arange(n_steps_per_printout))
        return printout_state, printout_state

    return run_small_simulation


run_to_printout = run_to_next_printout_fn(apply, n_steps_per_printout=100)

# Note: this takes a minute as it is mostly sequential and cannot saturate the GPU
t_start = time.time()
# for proper sampling: (5000 - 10000 at 1000); full equilibration: 10000 (from very bad init)
state, _ = lax.scan(run_to_printout, init_state, xs=jnp.arange(1000))  # dump first 1000 printouts as equilibration
_, reference_traj = lax.scan(run_to_printout, state, xs=jnp.arange(10000))  # sample 10000 snapshots
reference_traj.position.block_until_ready()
print('Time for generating reference dataset:', time.time() - t_start, 's')


# map over Gaussian mean values to compute contribution of multiple particles simultaneously
vectorized_normal_pdf = vmap(normal_pdf, (None, 0, None))

# the (discrete) integral over a pdf gives 1 making sure that the contribution of each single particle
# is 1 as in the case of discrete binning
dx_bin = (x_vals[1] - x_vals[0])  # assuming a homogeneous grid


def init_normalised_density_fn(x_vals, dx_bin):
    """
    Initializes a function that computes the normalized density given simulation states.
    We approximate the binning operation by a 2D Gaussian to have a differentiable function.
    """

    @jit
    def normalized_density(R):
        pdf_vals = vectorized_normal_pdf(x_vals, R, dx_bin)
        return jnp.mean(pdf_vals, axis=0)

    return normalized_density


normalized_density_fn = init_normalised_density_fn(x_vals, dx_bin)


def compute_quantity_traj(traj, quantity_fn):
    """
    Helper function applying a function onto each state in the trajectory.

    An alternative implementation via vmap is possible in principle, but this would lead to
    OOM errors for longer trajectories.
    """

    def quantity_trajectory(dummy_carry, state):
        R = state.position
        quantity = quantity_fn(R)
        return dummy_carry, quantity

    _, quantity_trajs = lax.scan(quantity_trajectory, 0., traj)
    return quantity_trajs


target_density = compute_quantity_traj(reference_traj, normalized_density_fn)
target_density = jnp.mean(target_density, axis=0)  # average over all trajectory snapshots

plot_1d_over_x(x_vals, target_density)
plt.show()


def spline_potential(x_vals, energy_params):
    spline = custom_interpolate.MonotonicInterpolate(x_vals, energy_params)
    v_spline = vmap(spline)

    def spline_energy_fn(R):
        energies = v_spline(R)
        return jnp.sum(energies)
    return spline_energy_fn


def prior_energy(confinement_x):
    def single_well(x):
        harmonic_confinement_x = confinement_x * (x - 0.5) ** 2
        return harmonic_confinement_x

    vectorized_single_well = vmap(single_well)

    def prior_energy_fn(R):
        energies = vectorized_single_well(R)
        return jnp.sum(energies)

    return prior_energy_fn


# TODO: Run multiple simulations with different priors: Plot loss curves and Potentials / Densities in Supplement
# prior_fn = prior_energy(500.)
prior_fn = prior_energy(70.)

spline_grid = jnp.linspace(0., 1., 50)


def energy_fn_template(params):
    spline = spline_potential(spline_grid, params)
    def energy_initialized(R, t=0):
        # nn_energy_initialized = partial(nn_energy, params)
        # return nn_energy_initialized(R) + prior_fn(R)
        return spline(R) + prior_fn(R)
    return jit(energy_initialized)


# def energy_fn_template(params):
#     nn = spline_potential(x_vals, params)
#     def energy_initialized(R):
#         return spline(R) + prior_fn(R)
#
#     return jit(energy_initialized)


simulator_template = partial(simulate.nvt_langevin, shift=shift, dt=dt, kT=kbT)

# initialize network and simulation
init_params = 0.01 * random.normal(model_key, spline_grid.shape)
energy_fn_init = energy_fn_template(init_params)
init_sim, apply_fn = simulator_template(energy_fn_init)
init_state = init_sim(simuation_key, R_init, mass=mass)

timesteps_per_printout = 100
num_dumped = 200
num_printouts_production = 2000  # for quick, but under-sampled results  # TODO increase or increase N
# num_printouts_production = 10000  # uncomment and re-run cells for production run


@jit
def generate_reference_trajectory(params, sim_state):
    # initialize simulation with current params / energy_fn
    energy_fn = energy_fn_template(params)
    _, apply_fn = simulator_template(energy_fn)
    run_small_simulation = run_to_next_printout_fn(apply_fn, timesteps_per_printout)

    # run simulation and compute energies
    sim_state, _ = lax.scan(run_small_simulation, sim_state, xs=jnp.arange(num_dumped))  # equilibrate
    sim_state, traj = lax.scan(run_small_simulation, sim_state, xs=jnp.arange(num_printouts_production))
    U_traj = compute_quantity_traj(traj, energy_fn)

    return sim_state, traj, U_traj


def compute_weights(params, traj_state):
    def estimate_effective_samples(weights):
        weights = jnp.where(weights > 1.e-10, weights,
                            1.e-10)  # mask to avoid n_eff==NaN from log(0) if any single w==0.
        exponent = - jnp.sum(weights * jnp.log(weights))
        return jnp.exp(exponent)

    def proper_normed_weights(prob_ratio):
        weights = prob_ratio / jnp.sum(prob_ratio)
        n_eff = estimate_effective_samples(weights)
        return weights, n_eff

    _, trajectory, U_traj = traj_state

    # cannot save whole energy computation for backward pass to memory: need checkpointing
    energy_fn = checkpoint(energy_fn_template(params))
    # Compute energies of reference trajectory using the new potential
    U_traj_new = compute_quantity_traj(trajectory, energy_fn)

    prob_ratio = jnp.exp(-(1. / kbT) * (U_traj_new - U_traj))

    # handle case when weights are inf or sum of weights are 0: in this case, n_eff cannot be computed and recompute
    # of the trajectory is necessary.
    weights_not_defined = jnp.bitwise_or(jnp.any(~jnp.isfinite(prob_ratio)), jnp.sum(prob_ratio) < 1.e-4)
    # if weights are properly defined, we can compute n_eff to
    weights, n_eff = lax.cond(weights_not_defined, lambda x: (x, jnp.array(0., dtype=prob_ratio.dtype)),
                              proper_normed_weights, prob_ratio)
    return weights, n_eff


# Note: could include regularisation here, if necessary

def reweighting_loss(params, traj_state):
    def rms_loss(predictions, targets):
        squared_difference = jnp.square(targets - predictions)
        mean_of_squares = jnp.mean(squared_difference)
        return jnp.sqrt(mean_of_squares)

    weights, _ = compute_weights(params, traj_state)
    _, traj, _ = traj_state
    pdf_traj = compute_quantity_traj(traj, normalized_density_fn)

    weighted_pdfs = (pdf_traj.T * weights).T  # weight each snapshot
    predicted_pdf_ensemble_average = jnp.sum(weighted_pdfs,
                                             axis=0)  # sum over weighted quantity, weights account for "averaging"
    loss = rms_loss(predicted_pdf_ensemble_average, target_density)  # times 1000. to have loss in order of 1

    return loss, predicted_pdf_ensemble_average


def propagation_fn(params, traj_state, reweight_ratio=0.8):
    def trajectory_identity_mapping(input):
        """Helper function to re-use trajectory if no recomputation is necessary"""
        _, traj_state = input
        return traj_state

    def recompute_trajectory(input):
        """Recomputes the reference trajectory, starting the last state of the previous trajectory,
        saving equilibration time."""
        params, traj_state = input
        sim_state = traj_state[0]
        updated_traj_state = generate_reference_trajectory(params, sim_state)
        return updated_traj_state

    # check if trajectory re-use is possible; recompute trajectory if not
    weights, n_eff = compute_weights(params, traj_state)
    n_snapshots = traj_state[2].size
    recompute = n_eff < reweight_ratio * n_snapshots
    traj_state = lax.cond(recompute, recompute_trajectory, trajectory_identity_mapping, (params, traj_state))

    outputs, curr_grad = value_and_grad(reweighting_loss, has_aux=True)(params, traj_state)
    loss_val, predicted_pdf = outputs

    return traj_state, curr_grad, loss_val, predicted_pdf


initial_lr = -0.2  # step towards negative gradient direction
# lr_schedule = optimizers.exponential_decay(initial_lr, 100, 0.1)
lr_schedule = optax.polynomial_schedule(initial_lr, initial_lr * 0.01, power=2, transition_steps=150)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)



@jit
def update(params, opt_state, traj_state):
    traj_state, curr_grad, loss_val, predicted_pdf = propagation_fn(params, traj_state)
    scaled_grad, opt_state = optimizer.update(curr_grad, opt_state, params)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, opt_state, traj_state, curr_grad, loss_val, predicted_pdf


# Initialize DiffTRe by computing the first reference trajectory
traj_state = generate_reference_trajectory(init_params, init_state)  # equilibrate initially
t_start = time.time()
traj_state = generate_reference_trajectory(init_params, traj_state[0])
traj_state[0].position.block_until_ready()
print('Time for initial reference trajectory generation', time.time() - t_start, 's')

initial_nn_traj = traj_state[1]

initial_pdf = compute_quantity_traj(initial_nn_traj, normalized_density_fn)
initial_pdf = jnp.mean(initial_pdf, axis=0)

plot_1d_over_x(x_vals, initial_pdf)
plt.show()


def potential_over_x(params):
    energy_fn = energy_fn_template(params)

    def pot_fn(R):
        R = jnp.expand_dims(R, 0)
        return energy_fn(R)

    return vmap(pot_fn)(jnp.expand_dims(x_vals, -1))


# save intermediate results for postprocesing
loss_history = []
predicted_densities = []
gradients = []
intermediate_params = [init_params]

opt_state = optimizer.init(init_params)
params = init_params
num_updates = 150
for step in range(num_updates):
    start_time = time.time()
    params, opt_state, traj_state, curr_grad, loss_val, predicted_pdf = update(params, opt_state, traj_state)
    print('Loss at step', step, ':', str(jnp.round(loss_val, decimals=3)), 'Time for completion:',
          str(round(time.time() - start_time, 1)), 's')
    loss_history.append(loss_val)
    gradients.append(curr_grad)
    intermediate_params.append(params)
    predicted_densities.append(predicted_pdf)

    cur_potential = potential_over_x(params)
    plot_training_process(loss_history, x_vals, predicted_pdf, target_density, cur_potential,
                          analytic_potential_vals)

# save and or load optimization
save_str = 'data/saved/1d_double_well_optimization_results_' + str(visible_device) + '.pkl'
with open(save_str, 'wb') as f:
    pickle.dump([loss_history, gradients, intermediate_params, predicted_densities], f)

# with open(save_str, 'rb') as f:
#     loss_history, gradients, intermediate_params, predicted_pdfs = pickle.load(f)

# visualize gradient:
# Let's visualize how the gradient alters the potential

def potential_gradient(params, gradient, eps=1.e-3):
    """A function for visualizing the effect on the potential from moving down the negative gradient."""

    def update_params(params, gradient):
        return params - eps * gradient
    perturbed_params = tree_multimap(update_params, params, gradient)
    U0 = energy_fn_template(params)
    U1 = energy_fn_template(perturbed_params)

    def pot_gradient_fn(R):
        R = jnp.expand_dims(R, 0)
        return (U1(R) - U0(R)) / eps

    return vmap(pot_gradient_fn)(jnp.expand_dims(x_vals, -1))

visualize_step = 0
potential = potential_over_x(intermediate_params[visualize_step])
pot_gradient = potential_gradient(intermediate_params[visualize_step], gradients[visualize_step])
loss = predicted_densities[visualize_step] - target_density

plt.figure()
plt.plot(x_vals, potential, label='Initial Potential')
plt.plot(x_vals, pot_gradient, label='$\Delta U$')
plt.plot(x_vals, loss, label='$\rho - \tilde\rho$')
plt.legend()
plt.show()

# # Prior ablation study

# feel free to change the prior and see how it affects the initial loss and convergence speed to the target potential

# purpose of prior: speed-up convergence of trained model for finite time training:
# 2 mechanisms: 1. spend less time with trajectories that can be rules out from first principles:
#                  No need to spend time on learning first principles from data
#               2. Gradient depends on sampled states: Less informative if far away from target distribution
#                  (gradient is 0 where observable is 0) e.g. observable = density(x_bar)


# finally it is good practice to check that the learned potential also produces target observables during long
# simulation runs. This ensures that the potential was not overfit to initial states since optimization
# is run on rather short equilibration and production runs.

# let's deliberately start from a state far from equilibrium

predicted_energy_fn = energy_fn_template(intermediate_params[-1])
R_init = 0.1 * box * random.uniform(random.PRNGKey(123), (N, 1))  # all particles in leftmost decile of box
init_sim, apply_fn = simulator_template(predicted_energy_fn)
init_state = init_sim(random.PRNGKey(456), R_init, mass=mass)
run_small_simulation = run_to_next_printout_fn(apply_fn, timesteps_per_printout)

sim_state, _ = lax.scan(run_small_simulation, init_state, xs=jnp.arange(1000))  # equilibrate
_, traj = lax.scan(run_small_simulation, sim_state, xs=jnp.arange(10000))

predicted_density = jnp.mean(compute_quantity_traj(traj, normalized_density_fn), axis=0)

plt.figure()
plt.plot(x_vals, predicted_density, label='Prediction')
plt.plot(x_vals, target_density, label='Target', linestyle='--')
plt.legend()
plt.show()


# TODO maybe additionally for RDF in appendix - tabulated potential or GNN (problem with global support)?
#  --> might be helpful for some, only show gradient as function of r



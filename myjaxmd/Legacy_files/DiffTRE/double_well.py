import os
import sys
if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs



import jax.numpy as jnp
from jax import random, nn, vmap, jit, lax, checkpoint, value_and_grad, tree_multimap
from jax_md import simulate, space
import haiku as hk
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import time
from jax.scipy.stats.multivariate_normal import pdf as mvn_pdf
from functools import partial
from jax.experimental import optimizers
import pickle

N = 30000  # the more particles in the simulation, the less snapshots are needed for the same smoothness of outputs
dimension = 2
dt = 2.e-1  # 10 fs
kbT = 1.
mass = 1.
resolution = 100  # TODO maybe reduce to reduce N?
x_range = [-5, 5]
y_range = [-5, 5]

def plot_1d_over_x(x, z, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    ax.plot(x, z)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def _colorplot(ax, x_mesh, y_mesh, z, levels, x_label=None, y_label=None, title=None):
    contours = ax.contourf(x_mesh, y_mesh, z, cmap=plt.get_cmap("coolwarm"), levels=levels)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return contours


def plot_2d_colorplot(x_mesh, y_mesh, z, title=None, x_label=None, y_label=None, z_label=None, n_bins=15):
    fig, ax = plt.subplots()
    levels = MaxNLocator(nbins=n_bins).tick_values(z.min(), z.max())
    contours = _colorplot(ax, x_mesh, y_mesh, z, levels, x_label=x_label, y_label=y_label, title=title)
    cbar = fig.colorbar(contours)
    cbar.ax.set_ylabel(z_label)


def plot_training_process(loss_history, x_mesh, y_mesh, predicted_pdf, reference_pdf, x_label=None, y_label=None,
                          z_label=None, n_bins=15):
    step = len(loss_history) - 1
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3. * 6.4, 4.8))
    ax1.plot(loss_history)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0., 0.3)
    levels = MaxNLocator(nbins=n_bins).tick_values(reference_pdf.min(), reference_pdf.max())
    _ = _colorplot(ax2, x_mesh, y_mesh, predicted_pdf, levels, x_label=x_label,
                   y_label=y_label, title='Predicted pdf step' + str(step))
    contours = _colorplot(ax3, x_mesh, y_mesh, reference_pdf, levels, x_label=x_label,
                          y_label=y_label, title='Target PDF')
    cbar = fig.colorbar(contours)
    cbar.ax.set_ylabel(z_label)
    plt.tight_layout()
    plt.show()



def analytic_double_well(position):
    x = position[0]
    y = position[1]
    harmonic_confinement = 0.2 * y ** 2
    double_well = 0.02 * (x - 4.) * (x - 2.) * (x + 1.) * (x + 4.)
    return kbT * (harmonic_confinement + double_well)

y_center = int(resolution / 2)
x_vals = jnp.linspace(*x_range, resolution)
y_vals = jnp.linspace(*y_range, resolution)
x_mesh, y_mesh = jnp.meshgrid(x_vals, y_vals)
vectorized_analytic_double_well = vmap(analytic_double_well)
matrix_vectorized_analytic_double_well = vmap(vectorized_analytic_double_well)
grid = jnp.dstack([x_mesh, y_mesh])
analytic_potential_vals = matrix_vectorized_analytic_double_well(grid)

plot_1d_over_x(x_vals, analytic_potential_vals[y_center, :], x_label='$x / L$', y_label='$U / k_BT$')
plot_2d_colorplot(x_mesh, y_mesh, analytic_potential_vals, title='Analytic double well potential',
                  x_label='$x / L$', y_label='$y / L$', z_label='$U / k_BT$')
plt.show()


_, shift = space.free()
key = random.PRNGKey(0)
position_key, model_key, simuation_key = random.split(key, 3)

# initialize particles
dx = x_range[1] - x_range[0]
dy = y_range[1] - y_range[0]
center = jnp.mean(jnp.array([x_range, y_range]), axis=1)
R_init = 0.01 * jnp.array([dx, dy]) * (random.uniform(position_key, (N, dimension)) - 0.5) + center


def analytic_energy_fn(R, t=0):
    """Ideal gas particles don't interact. The only energy contribution comes from the position in the box."""
    energies = vectorized_analytic_double_well(R)
    return jnp.sum(energies)


# init, apply = simulate.nvt_nose_hoover(analytic_energy_fn, shift, dt, kbT, chain_steps=1)
init, apply = simulate.nvt_langevin(analytic_energy_fn, shift, dt, kbT, gamma=0.1)
init_state = init(simuation_key, R_init)


def run_to_next_printout_fn(apply_fn, n_steps_per_printout):
    """Helper function to sample a specified number of snapshots."""
    do_step = lambda state, t: (apply_fn(state), ())

    @jit
    def run_small_simulation(start_state, dummy_carry):
        printout_state, _ = lax.scan(do_step, start_state, xs=jnp.arange(n_steps_per_printout))
        return printout_state, printout_state

    return run_small_simulation


run_small_simulation = run_to_next_printout_fn(apply, n_steps_per_printout=100)

# Note: this takes a minute as it is mostly sequential and cannot saturate the GPU
t_start = time.time()
state, _ = lax.scan(run_small_simulation, init_state, xs=jnp.arange(10000))  # dump first 1000 printouts as equilibration
_, reference_traj = lax.scan(run_small_simulation, state, xs=jnp.arange(3000))  # sample 10000 snapshots
print('Time for generating reference dataset:', time.time() - t_start, 's')

# use vmap to compute PDF values of a 2D Gaussian centered at a given particle position over the x-y grid
matrix_mvn_pdf = vmap(vmap(mvn_pdf, (0, None, None)), (0, None, None))

# map over Gaussian mean values to compute contribution of multiple particles simultaneously
vectorized_matrix_mvn_pdf = vmap(matrix_mvn_pdf, (None, 0, None))

# the (discrete) integral over a pdf gives 1 making sure that the contribution of each single particle
# is 1 as in the case of discrete binning
dx_bin = x_vals[1] - x_vals[0]  # assuming a homogeneous grid
dy_bin = y_vals[1] - y_vals[0]
cov_matrix = jnp.array([[dx_bin ** 2, 0.], [0., dy_bin ** 2]])  # defines Gaussian width, goes to 0 for bin -> 0
bin_area = dx_bin * dy_bin

test_particle_positions = jnp.array([[1., -1.], [0., 1.], [2., 2.]])
multiple_pdf_vals = vectorized_matrix_mvn_pdf(grid, test_particle_positions, cov_matrix)
# aggregate multiple particles, multiplication by bin area ensures sum of 1
aggregated_pdf_vals = bin_area * jnp.mean(multiple_pdf_vals, axis=0)
print('Sum of particle distribution =', jnp.sum(aggregated_pdf_vals))

plot_2d_colorplot(x_mesh, y_mesh, aggregated_pdf_vals, title='Test particle correlation function')

def init_particle_distribution_function(x_y_grid, cov):
    """
    Initializes a function that computes the particle distriution function given simulation states.
    We approximate the binning operation by a 2D Gaussian to have a differentiable function.
    """
    bin_area = jnp.sqrt(cov[0, 0]) * jnp.sqrt(cov[1, 1])

    @jit
    def particle_distribution_function(R):
        pdf_vals = vectorized_matrix_mvn_pdf(x_y_grid, R, cov)
        return bin_area * jnp.mean(pdf_vals, axis=0)

    return particle_distribution_function


particle_distribution_fn = init_particle_distribution_function(grid, cov_matrix)


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


target_pdf = compute_quantity_traj(reference_traj, particle_distribution_fn)
target_pdf = jnp.mean(target_pdf, axis=0)  # average over all trajectory snapshots

plot_2d_colorplot(x_mesh, y_mesh, target_pdf, title='Target Particle Distribution Function')
plt.show()

def NN_potential(layers, activation_fn=nn.swish):
    @hk.without_apply_rng
    @hk.transform
    def energy_model(R):
        nn_potential = vmap(hk.nets.MLP(output_sizes=layers, activation=activation_fn))  # map over all particles
        energies = nn_potential(R)
        return jnp.sum(energies)

    return energy_model.init, energy_model.apply


def prior_energy(confinement_x, confinement_y):
    def single_well(position):
        x = position[0]
        y = position[1]
        harmonic_confinement_x = confinement_x * x ** 2
        harmonic_confinement_y = confinement_y * y ** 2
        return harmonic_confinement_x + harmonic_confinement_y

    vectorized_single_well = vmap(single_well)

    def prior_energy_fn(R):
        energies = vectorized_single_well(R)
        return jnp.sum(energies)

    return prior_energy_fn


# a NN with 2 hidden layers outputting a scalar energy
energy_init_fn, nn_energy = NN_potential([100, 100, 1])

# a tight single-well as prior energy guess
prior_fn = prior_energy(0.5, 0.5)


def energy_fn_template(params):
    def energy_initialized(R):
        nn_energy_initialized = partial(nn_energy, params)
        return nn_energy_initialized(R) + prior_fn(R)

    return jit(energy_initialized)


nose_hoover_simulator = partial(simulate.nvt_langevin, shift_fn=shift, dt=dt, kT=kbT)

# initialize network and simulation
init_params = energy_init_fn(model_key, R_init)
energy_fn_init = energy_fn_template(init_params)
init_sim, apply_fn = nose_hoover_simulator(energy_fn_init)
init_state = init_sim(simuation_key, R_init, mass=mass)

timesteps_per_printout = 100
num_dumped = 1000
num_printouts_production = 1000  # for quick, but under-sampled results


# num_printouts_production = 10000  # uncomment and re-run cells for production run

@jit
def generate_reference_trajectory(params, sim_state):
    # initialize simulation with current params / energy_fn
    energy_fn = energy_fn_template(params)
    _, apply_fn = nose_hoover_simulator(energy_fn)
    run_small_simulation = run_to_next_printout_fn(apply_fn, timesteps_per_printout)

    # run simulation and compute energies
    # sim_state, _ = lax.scan(run_small_simulation, sim_state, xs=jnp.arange(num_dumped))  # equilibrate
    sim_state, _ = lax.scan(run_small_simulation, init_state, xs=jnp.arange(num_dumped))  # always start from same broad particle distribution
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

def reweighting_loss(params, traj_state):
    def rms_loss(predictions, targets):
        squared_difference = jnp.square(targets - predictions)
        mean_of_squares = jnp.mean(squared_difference)
        return jnp.sqrt(mean_of_squares)

    weights, _ = compute_weights(params, traj_state)
    _, traj, _ = traj_state
    pdf_traj = compute_quantity_traj(traj, particle_distribution_fn)

    weighted_pdfs = (pdf_traj.T * weights).T  # weight each snapshot
    predicted_pdf_ensemble_average = jnp.sum(weighted_pdfs,
                                             axis=0)  # sum over weighted quantity, weights account for "averaging"
    loss = 1000. * rms_loss(predicted_pdf_ensemble_average, target_pdf)  # times 1000. to have loss in order of 1

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

initial_lr = 0.0025
lr_schedule = optimizers.exponential_decay(initial_lr, 100, 0.1)
opt_init, opt_update, get_params = optimizers.adam(lr_schedule, b1=0.2, b2=0.5)


@jit
def update(step, params, opt_state, traj_state):
    traj_state, curr_grad, loss_val, predicted_pdf = propagation_fn(params, traj_state)
    opt_state = opt_update(step, curr_grad, opt_state)
    return get_params(opt_state), opt_state, traj_state, curr_grad, loss_val, predicted_pdf


# Initialize DiffTRe by computing the first reference trajectory
t_start = time.time()
traj_state = generate_reference_trajectory(init_params, init_state)
print('Time for initial reference trajectory generation', time.time() - t_start, 's')

initial_nn_traj = traj_state[1]

initial_pdf = compute_quantity_traj(initial_nn_traj, particle_distribution_fn)
initial_pdf = jnp.mean(initial_pdf, axis=0)

plot_2d_colorplot(x_mesh, y_mesh, initial_pdf, title='Target Particle Distribution Function')

plot_training_process([], x_mesh, y_mesh, initial_pdf, target_pdf, x_label='$x / L$', y_label='$y / L$',
                      z_label='PDF')

# save intermediate results for postprocesing
loss_history = []
predicted_pdfs = []
gradients = []
intermediate_params = []
intermediate_params.append(init_params)

opt_state = opt_init(init_params)
params = init_params

num_updates = 100
for step in range(num_updates):
    start_time = time.time()
    params, opt_state, traj_state, curr_grad, loss_val, predicted_pdf = update(step, params, opt_state, traj_state)
    print('Loss at step', step, ':', str(jnp.round(loss_val, decimals=3)), 'Time for completion:',
          str(round(time.time() - start_time, 1)), 's')
    grad_norm = optimizers.l2_norm(curr_grad)
    print('Gradient norm:', grad_norm)
    loss_history.append(loss_val)
    gradients.append(curr_grad)
    intermediate_params.append(params)
    predicted_pdfs.append(predicted_pdf)
    # TODO fix horizontal scaling from bar
    plot_training_process(loss_history, x_mesh, y_mesh, predicted_pdf, target_pdf, x_label='$x / L$', y_label='$y / L$',
                          z_label='PDF')

# save and or load optimization
save_str = 'data/saved/double_well_optimization_results_' + str(visible_device) + '.pkl'
# with open(save_str, 'wb') as f:
#     pickle.dump([loss_history, gradients, intermediate_params, predicted_pdfs], f)

with open(save_str, 'rb') as f:
    loss_history, gradients, intermediate_params, predicted_pdfs = pickle.load(f)



def potential_gradient(params, gradient, grid, eps=1.e-2):
    """
    A function used to visualize the effect of moving down the gradient
    on the potential over x and y.
    """

    def update_params(params, gradient):
        return params - eps * gradient
    perturbed_params = tree_multimap(update_params, params, gradient)
    U0 = energy_fn_template(params)
    U1 = energy_fn_template(perturbed_params)

    def pot_gradient_fn(R):
        R = jnp.expand_dims(R, 0)
        return (U1(R) - U0(R)) / eps

    def pot_fn(R):
        R = jnp.expand_dims(R, 0)
        return U0(R)

    vectorized_potential_gradient = vmap(vmap(pot_gradient_fn))
    vectorized_potential = vmap(vmap(pot_fn))

    potential = vectorized_potential(grid)
    gradient = vectorized_potential_gradient(grid)

    return potential, gradient

target_idx = 95

potential, gradient = potential_gradient(intermediate_params[target_idx], gradients[target_idx], grid)

plot_2d_colorplot(x_mesh, y_mesh, potential, n_bins=100)
plot_1d_over_x(x_vals, potential[50, :], x_label=None, y_label=None)
plot_2d_colorplot(x_mesh, y_mesh, gradient)
plot_2d_colorplot(x_mesh, y_mesh, predicted_pdfs[target_idx] - target_pdf)
plot_2d_colorplot(x_mesh, y_mesh, predicted_pdfs[target_idx])





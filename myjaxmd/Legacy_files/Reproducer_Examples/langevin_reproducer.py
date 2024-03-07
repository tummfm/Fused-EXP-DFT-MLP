import jax.numpy as jnp
from jax import random, vmap, jit, lax
from jax_md import simulate, space
import matplotlib.pyplot as plt


def plot_1d_over_x(x, z, x_label=None, y_label=None, title=None):
    fig, ax = plt.subplots()
    ax.plot(x, z)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


N = 1000
dt = 1.e-3
kbT = 1.
mass = 1.
resolution = 100
box = jnp.array([1.])

n_steps_per_printout = 100
snapshots_dumped = 10000
snapshots_kept = 10000


@vmap
def analytic_well(x):
    well = 2000 * (x - 0.5)**6
    return kbT * well


def analytic_energy_fn(R, t=0):
    # the dummy value of t is necessary to avoid error in langevin initialisation
    energies = analytic_well(R)
    return jnp.sum(energies)


x_vals = jnp.linspace(0., box, resolution)
analytic_potential_vals = analytic_well(x_vals)
plot_1d_over_x(x_vals, analytic_potential_vals, x_label='x', y_label='U / kbT')

_, shift = space.periodic(box)
key = random.PRNGKey(0)
position_key, simuation_key = random.split(key, 2)
R_init = box * random.uniform(position_key, (N, 1))


init, apply = simulate.nvt_nose_hoover(analytic_energy_fn, shift, dt, kbT)  # works
init, apply = simulate.nvt_langevin(analytic_energy_fn, shift, dt, kbT, gamma=0.1)  # incorrect results, irrespective of gamma

init_state = init(simuation_key, R_init)
do_step = lambda state, t: (apply(state), ())


@jit
def run_to_printout(start_state, dummy_carry):
    printout_state, _ = lax.scan(do_step, start_state, xs=jnp.arange(n_steps_per_printout))
    return printout_state, printout_state


state, _ = lax.scan(run_to_printout, init_state, xs=jnp.arange(snapshots_dumped))  # equilibration
_, traj = lax.scan(run_to_printout, state, xs=jnp.arange(snapshots_kept))  # sample

hist, _ = jnp.histogram(traj.position, bins=resolution)
plot_1d_over_x(x_vals, hist, x_label='x', y_label='Frequency', title='Langevin')

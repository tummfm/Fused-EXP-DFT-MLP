"""Runs a forward simulation in Jax M.D. Can also be used to debug the forward-pass through the simulation.
   Also good to check accuracy of generated RDF with given number of samples and Gaussian smoothing length"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)  # controls on which gpu the program runs

import numpy as onp
import sys


import jax.numpy as np
from jax import random, lax
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)  # to execute 64 bit high precision sums, BUT: slows down simulation a lot
from jax import jit
from chemtrain.jax_md_mod import io
from chemtrain.jax_md_mod import custom_quantity

import time

from jax_md import space, energy, simulate, util
import matplotlib.pyplot as plt

from scipy import interpolate as sci_interpolate  # for interpolation of tabulated potential raw data

# TODO only proof of concept files, needs lots of rework to new library files

# simulation parameters:
model = 'LJ'
# model = 'Tabulated'

file = '../../examples/data/confs/SPC_FW_3nm.gro'
dt = util.f32(0.002)  # in fs
kbT = util.f32(2.49435321)
mass = util.f32(18.0154)

total_time = 30.
t_equilib = 1.  # equilibration time before sampling RDF.
print_every = 0.1

num_printouts_production = int((total_time - t_equilib) / print_every)
sample_times = np.array([t_equilib + (i + 1) * print_every for i in range(num_printouts_production)])  # times to sample state

# RDF parameters:
RDF_start = 0.
RDF_cut = 1.0
nbins = 250


dx_bin = (RDF_cut - RDF_start) / float(nbins)
rdf_bin_centers = np.linspace(RDF_start + dx_bin / 2., RDF_cut -  dx_bin / 2., nbins)
rdf_bin_boundaries = np.linspace(RDF_start, RDF_cut, nbins + 1)
sigma_RDF = dx_bin


R, box = io.load_box(file)
dimension = box.size
box_size = box[0]  # assuming cubic box
total_vol = box[0] * box[1] * box[2]  # volume of partition
N = R.shape[0]
particle_density = N / total_vol

displacement, shift = space.periodic(1.)  # here needs to be 1 instead of box_size due to reparametrization
def reparameterized_displacement(Ra, Rb, **kwargs):
    Ra = Ra / box[0]
    Rb = Rb / box[0]
    return displacement(Ra, Rb, **kwargs) * box[0]
def reparameterized_shift(R, dR, **kwargs):
    return shift(R / box[0], dR / box[0]) * box[0]

key = random.PRNGKey(0)

if model == 'LJ':
    energy_params = np.array([0.3, 1.])
    def energy_fn(energy_params):
        return jit(energy.lennard_jones_pair(reparameterized_displacement, sigma=energy_params[0],
                                             epsilon=energy_params[1], r_onset=0.8, r_cutoff=0.9))
elif model == 'Tabulated':
    r_cut = 0.9
    delta_cut = 0.1
    x_vals = np.linspace(0.05, r_cut + delta_cut, 100)
    table_loc = 'data/CG_potential_SPC_955.csv'
    tabulated_array = onp.loadtxt(table_loc)
    # compute tabulated values at spline support points
    U_init_int = sci_interpolate.interp1d(tabulated_array[:, 0], tabulated_array[:, 1], kind='cubic')
    energy_params =  np.array(U_init_int(x_vals))
    def energy_fn(energy_params):
        return jit(energy.tabulated_pair(reparameterized_displacement, x_vals, energy_params, r_onset=r_cut - 0.03, r_cutoff=r_cut))
else:
    print('This model is not implemented!')
    sys.exit()

# print(energy_fn(R))
# force_fun = grad(lambda R: -energy_fn(R))
# force_fun = jit(force_fun)  # throughs an error, but when jitted does not as NaNs are then not visible
# # unclear why throwing an error at all as force result does not contain any NaNs
# initial_forces = force_fun(R)

# draws initial velocities from Maxwell Boltzmann --> if we want to set velocities, we need to rewrite init function
# init, apply_fn, _ = simulate.nvt_nose_hoover(energy_fn, reparameterized_shift, dt, kbT, chain_length=3)

# alternative: use Langevin thermostat
# Caution: Langevin Thermostat is rather unstable, while Nose-Hoover is much more stable with the same settings!
# init, apply_fn = simulate.nvt_langevin(energy_fn, reparameterized_shift, dt, kbT, gamma=100.)

# alternative: nve simulation --> seems to be very stable
init, integrate = simulate.nve_adjoint(energy_fn)
state, dynamics = init(key, R, mass=mass, T_initial=kbT)

pair_corr_fun = custom_quantity.pair_correlation(reparameterized_displacement, rdf_bin_centers, sigma_RDF)  # initialize RDF compute function
rdf_fun = custom_quantity.radial_distribution_function(pair_corr_fun, particle_density, rdf_bin_boundaries)


def compute_RDF(state, sample_times):
    trajectory, state = integrate(state, dynamics, sample_times, energy_params, rtol=1.e-4)

    # cant use vmap for RDF computation, as this quickly consumes tons memory
    rdf_snapshots = lax.map(rdf_fun, trajectory.position)
    g_average = np.mean(rdf_snapshots, axis=0)
    return state, g_average

compute_RDF = jit(compute_RDF)

t_start = time.time()
state, g_average = compute_RDF(state, sample_times)  # rdf_list is device array of shape (n_runs, bins)
print('ps/min: ', (total_time) / ((time.time() - t_start) / 60.))


# compare with literature
if model == 'Tabulated':
    tabulated_array = onp.loadtxt("data/SPC_955_RDF.csv")
    plt.figure()
    plt.plot(tabulated_array[:, 0], tabulated_array[:, 1], label='Reference')  # sum over all particles to get RDF of current snapshot
    plt.plot(rdf_bin_centers, g_average, label='CG Simulation')  # sum over all particles to get RDF of current snapshot
    plt.legend()
    plt.savefig('Figures/CG_simulation_Tabulated_RDF.png')
elif model == 'LJ':
    # onp.savetxt('data/computed_RDF_LJ.csv', onp.asarray(g_average))
    plt.figure()
    plt.plot(rdf_bin_centers, g_average, label='LJ RDF')  # sum over all particles to get RDF of current snapshot
    plt.savefig('Figures/CG_simulation_LJ_RDF.png')


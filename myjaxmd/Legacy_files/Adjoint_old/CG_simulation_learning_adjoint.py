
# set which GPU to use:
# https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)  # controls on which gpu the program runs

import numpy as onp

import jax.numpy as np
from jax import random, value_and_grad, checkpoint
# config.update("jax_debug_nans", True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)
from jax import jit, lax
from jax.experimental import optimizers
from jax.ops import index_add
from functools import partial

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)

import time

from jax_md import space, energy, simulate
import matplotlib.pyplot as plt
from chemtrain.jax_md_mod import io
from chemtrain.jax_md_mod import custom_quantity

from scipy import interpolate as sci_interpolate  # for interpolation of tabulated potential raw data

# import jax.profiler
# jax.profiler.save_device_memory_profile("memory.prof")  # put this line where to output current GPU memory

# TODO only proof of concept files, needs lots of rework to new library files

def initialize_simulation(conf):
    model = conf['model']
    box = conf['box']
    total_vol = box[0] * box[1] * box[2]  # volume of partition
    N = conf['R'].shape[0]
    particle_density = N / total_vol

    # wrapped=False: Particles are not mapped back to original box after each step --> saves compute effort and
    #                allows easier time integration as space can be handled without periodic BCs:
    #                We can integrate positions unconstrained; In force computation: displacement function ensures that
    #                particle distances are still computed correctly
    #                For sampled configurations to be used: We can easily remap it back to the original box
    #                For wrapped=False: shift function is only adding displacement, not handling any periodicity
    displacement, shift = space.periodic(1., wrapped=False)  # here needs to be 1 instead of box_size due to reparametrization
    def reparameterized_displacement(Ra, Rb, **kwargs):
        Ra = Ra / box[0]
        Rb = Rb / box[0]
        return displacement(Ra, Rb, **kwargs) * box[0]

    r_cut = conf['r_cut']
    if model == 'LJ':
        def energy_fn(energy_params):
            return jit(energy.lennard_jones_pair(reparameterized_displacement, sigma=energy_params[0],
                                                 epsilon=energy_params[1], r_onset=0.8, r_cutoff=r_cut))
    elif model == 'Tabulated':
        def energy_fn(energy_params):
            return jit(energy.tabulated_pair(reparameterized_displacement, conf['x_vals'], energy_params, r_onset=r_cut - 0.03,
                                             r_cutoff=r_cut))
    else:
        print('This model is not implemented!')
        sys.exit()

    init, integrate = simulate.nve_adjoint(energy_fn)
    state, dynamics = init(conf['key'], conf['R'], mass=conf['mass'], T_initial=conf['kbT'])

    # pair_corr_fun takes displacement to compute distances: Also works with unwrapped coordinates
    pair_corr_fun = custom_quantity.pair_correlation(reparameterized_displacement, conf['rdf_bin_centers'],
                                                     conf['sigma_RDF'])  # initialize RDF compute function
    rdf_fun = custom_quantity.radial_distribution_function(pair_corr_fun, particle_density, conf['rdf_bin_boundaries'])
    rdf_fun = checkpoint(rdf_fun)  # need to add checkpointing here to avoid memory overflow from reverse pass in rdf computation

    simulation_dict = {'dynamics': dynamics, 'rdf_fun': rdf_fun, 'integrate': integrate, 'energy_fn': energy_fn}
    return state, simulation_dict

def rms_loss(g_predict, g_target):
    return np.sqrt(np.mean(np.square(g_target - g_predict)))


def RDF_simulation(energy_params, state, sample_times, simulation_dict):
    integrate = simulation_dict['integrate']
    trajectory, state = integrate(state, simulation_dict['dynamics'], sample_times, energy_params, rtol=1.e-4)
    # cant use vmap for RDF computation, as this quickly consumes tons memory
    rdf_snapshots = lax.map(simulation_dict['rdf_fun'], trajectory.position)
    g_average = np.mean(rdf_snapshots, axis=0)
    return state, g_average

def RDF_loss(energy_params, state, sim_dict, reference_rdf):
    state, g_average = RDF_simulation(energy_params, state, sample_times, sim_dict)
    return rms_loss(g_average, reference_rdf), state


def update(step, params, opt_state, state):
    """ Compute the gradient, update the parameters and the opt_state and return loss"""
    outputs, curr_grad = value_and_grad(RDF_loss, has_aux=True)(params, state)
    loss_val = outputs[0]
    state = outputs[1]
    opt_state = opt_update(step, curr_grad, opt_state)  # update optimizer state after stepping
    # get_param is globally defined function --> OK since we do not change this function, only use it
    return get_params(opt_state), opt_state, loss_val, curr_grad, state

def check_gradient(energy_params, RDF_loss, dx=1.e-6):
    grad_findiff = onp.zeros(energy_params.size)
    for i in range(energy_params.size):
        param_plus = index_add(energy_params, i, dx)
        param_minus = index_add(energy_params, i, -dx)
        grad_findiff[i] = (RDF_loss(param_plus) - RDF_loss(param_minus)) / (2. * dx)
    return grad_findiff


################### user input ###########:
file = '../../examples/data/confs/SPC_FW_3nm.gro'  # maybe easier to vectorize over initial condition than file strings --> dict
# need to set util.f32(0.002)?? Has impact on gradient! Maybe use x64?

model = 'LJ'
# model = 'Tabulated'

kbT = 2.49435321
mass = 18.0154
# total time verlängern, je kleiner loss ist, um informativere Loss function später zu erhalten, aber exploding gradients problem am Anfang klein zu halten --> in nähe von equilibrium sollte nicht mehr so tragisch sein
# state immer updaten, um equilibration Zeit zu verringern
total_time = 1.5
t_equilib = 0.1  # equilibration time before sampling RDF
print_every = 0.1
# RDF parameters:
if model == 'LJ':
    r_cut = 0.9
    x_vals = None
    reference_rdf = onp.loadtxt('data/LJ_reference_RDF.csv')
    energy_params = np.array([0.2, 1.2])  # initial guess
else:
    reference_rdf = onp.loadtxt('data/CG_potential_SPC_955.csv')
    r_cut = 0.9
    delta_cut = 0.1
    x_vals = np.linspace(0.05, r_cut + delta_cut, 100)
    table_loc = 'data/CG_potential_SPC_955.csv'
    tabulated_array = onp.loadtxt(table_loc)
    # compute tabulated values at spline support points
    U_init_int = sci_interpolate.interp1d(tabulated_array[:, 0], tabulated_array[:, 1], kind='cubic')
    energy_params = np.array(U_init_int(x_vals))


RDF_start = 0.
RDF_cut = 1.5

# optimization parameters:
num_updates = 100
step_size = 0.01
#############################################

# preprocess user input
num_printouts_production = int((total_time - t_equilib) / print_every)
sample_times = np.array([t_equilib + (i + 1) * print_every for i in range(num_printouts_production)])  # times to sample state

nbins = reference_rdf.size  # bin size has big impact on gradients: resolving rdf properly makes loss more differentiable --> more informative gradients
dx_bin = (RDF_cut - RDF_start) / float(nbins)
rdf_bin_centers = np.linspace(RDF_start + dx_bin / 2., RDF_cut - dx_bin / 2., nbins)
rdf_bin_boundaries = np.linspace(RDF_start, RDF_cut, nbins + 1)
sigma_RDF = 2. * dx_bin

R, box = io.load_box(file)
key = random.PRNGKey(0)

# fill dictionary of configuration
conf_dict = {'R': R, 'box':box, 'key':key, 'kbT':kbT, 'mass':mass, 'sample_times':sample_times, 'model':model,
             'rdf_bin_centers':rdf_bin_centers, 'rdf_bin_boundaries':rdf_bin_boundaries, 'sigma_RDF':sigma_RDF,
             'r_cut': r_cut, 'x_vals':x_vals}


# true params: sig=0.3, eps=1.
# loss_val, grad_energy = value_and_grad(RDF_simulation)(energy_params, init_dict, key)  # naive reverse mode

# compute initial state
state, simulation_dict = initialize_simulation(conf_dict)

# RDF_simulation = jit(RDF_simulation, static_argnums=(1,))
RDF_loss = partial(RDF_loss, sim_dict=simulation_dict, reference_rdf=reference_rdf)
RDF_loss = jit(RDF_loss)

# initialize optimizer
# TODO change momentum of adam to fit our needs of convergence in few time steps
params = energy_params
opt_init, opt_update, get_params = optimizers.adam(step_size)  # define adam
#opt_init, opt_update, get_params = optimizers.sgd(step_size)
cur_opt_state = opt_init(params)  # initialize adam
# optimize parameters
loss_history = onp.zeros(num_updates)



update = jit(update)
for step in range(num_updates):
    start_time = time.time()
    #print('Gradient Findiff e-4:', check_gradient(params, RDF_loss, dx=1.e-4))
    #print('Gradient Findiff e-5:', check_gradient(params, RDF_loss, dx=1.e-5))
    #print('Gradient Findiff e-6:', check_gradient(params, RDF_loss, dx=1.e-6))
    params, cur_opt_state, loss_val, curr_grad, state = update(step, params, cur_opt_state, state)
    step_time = time.time() - start_time
    loss_history[step] = loss_val
    print('Gradient step', step, ':', curr_grad)
    print('sigma:', params[0],'epsilon:', params[1])
    print("Step {} in {:0.2f} sec".format(step, step_time))
    print('Loss = ', loss_val)


plt.figure()
plt.plot(loss_history)
plt.ylabel('loss')
plt.xlabel('update step')
plt.savefig('Train_history.png')

state, g_average = RDF_simulation(params, conf_dict)

plt.figure()
plt.plot(rdf_bin_centers, g_average, label='predicted')  # sum over all particles to get RDF of current snapshot
plt.plot(rdf_bin_centers, reference_rdf, label='reference')  # sum over all particles to get RDF of current snapshot
plt.plot(rdf_bin_centers, g_average_initguess, label='initial guess')  # sum over all particles to get RDF of current snapshot
plt.legend()
plt.savefig('RDF_test' + str(num_updates) + '.png')

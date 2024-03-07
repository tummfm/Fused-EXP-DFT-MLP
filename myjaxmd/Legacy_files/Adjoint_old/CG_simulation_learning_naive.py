
# set which GPU to use:
# https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)  # controls on which gpu the program runs

import numpy as onp

import jax.numpy as np
from jax import random, lax, jvp
# config.update("jax_debug_nans", True)  # throws a NaN error when computing the force from the energy, even though the simulation is running properly
# config.update('jax_disable_jit', True)  # to check for NaNs, jit needs to be disabled, otherwise it won't catch the NaN
# config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# config.update('jax_enable_x64', True)
from jax import jit
from jax.experimental import optimizers
from jax.ops import index_add
from functools import partial

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)

import time

from jax_md import space, energy
from chemtrain.jax_md_mod import io
from chemtrain.jax_md_mod import custom_quantity, custom_simulator

import matplotlib.pyplot as plt


# jax.profiler.save_device_memory_profile("memory.prof")  # put this line where to output current GPU memory

# TODO only proof of concept files, needs lots of rework to new library files

def loss(g_predict, g_target):
    return np.sqrt(np.mean(np.square(g_target - g_predict)))


def RDF_simulation(energy_params, conf):

    box = conf['box']
    total_vol = box[0] * box[1] * box[2]  # volume of partition
    N = conf['R'].shape[0]
    particle_density = N / total_vol

    displacement, shift = space.periodic(1.)  # here needs to be 1 instead of box_size due to reparametrization
    # TODO reparametrization did not change too much in practice! Why?
    displacement, shift = space.periodic(1.)
    def reparameterized_displacement(Ra, Rb, **kwargs):
        Ra = Ra / box[0]
        Rb = Rb / box[0]
        return displacement(Ra, Rb, **kwargs) * box[0]
    def reparameterized_shift(R, dR, **kwargs):
        return shift(R / box[0], dR / box[0]) * box[0]



    #energy_fn = jit(energy.soft_sphere_pair(displacement, sigma=0.3, epsilon=1.0))  # TODO just a test potential! Substitute this
    # TODO: they change lambda dynamically (or statically at beginning) to counteract exploding gradients --> find approach to do that
    #  --> this is essential for convergence!!
    #  Depends on initial condition of simulation (i.e. params) --> should be reselected for each params sample
    #  --> Lyaponov regularization!! --> adaptively change lambda --> this is different from what they do, we would just change lambda based on this, they regularize the time step function
    energy_fn = jit(energy.lennard_jones_pair(reparameterized_displacement, sigma=energy_params[0], epsilon=energy_params[1], r_onset=0.8, r_cutoff=0.9))
    # energy_fn = energy.lennard_jones_pair(reparameterized_displacement, sigma=energy_params[0], epsilon=energy_params[1], r_onset=0.8, r_cutoff=0.9)

    # draws initial velocities from Maxwell Boltzmann --> if we want to set velocities, we need to rewrite init function
    # Nose-Hoover-Thermostat:

    # init, apply_fn, grad_stop = simulate.nvt_nose_hoover(energy_fn, reparameterized_shift, conf['dt'], conf['kbT'], stop_ratio=0.05, chain_length=3)

    # alternative: use Langevin thermostat  --> currently seems to be very unstable (the Simulation itself, not the potential optimization); reason unclear
    #                                           uses second order Langevin integrator (with 2 random variables)
    # TODO implement Langevin Thermostat as in Espresso (just adding noise term to forces)
    #  --> structure should be similar to nve; does the noise from the thermostat help with exploding gradients? --> relation to gradient noise
    # grad_stop = None
    # init, apply_fn = simulate.nvt_langevin(energy_fn, reparameterized_shift, conf['dt'], conf['kbT'], gamma=1.)

    # alternative 2: nve simulation --> seems to be very stable (on the MD side)
    # maybe less noise from thermostat needs to be counteracted by stronger stop ratio? (Compared to Nose-Hoover)
    init, apply_fn, grad_stop = custom_simulator.nve_gradient_stop(energy_fn, reparameterized_shift, dt, stop_ratio=0.07)

    apply_fn = jit(apply_fn)
    state = init(conf['key'], conf['R'], mass=conf['mass'], T_initial=conf['kbT'])

    pair_corr_fun = custom_quantity.pair_correlation(reparameterized_displacement, conf['rdf_bin_centers'], conf['sigma_RDF'])  # initialize RDF compute function
    rdf_fun = custom_quantity.radial_distribution_function(pair_corr_fun, particle_density, conf['rdf_bin_boundaries'])
    rdf_fun = jit(rdf_fun, static_argnums=(0,))

    def do_step(state, t):
        state = apply_fn(state)
        # only position and velocity change over time --> set those to 0 --> other state variables are constants
        if not grad_stop is None:  # no gradient stop in Langevin Thermostat
            state = grad_stop(state)
        return state, t
    # do_step = lambda state, t: (apply_fn(state), ())
    def run_small_simulation(state, t=0.):
        # scan always needs this second argument to feed xs to. Here it is only used as dummy, as apply_fn is not time dependent
        state, _ = lax.scan(do_step, state, xs=np.arange(conf['steps_per_printout']))
        curr_g = rdf_fun(state.position)
        return state, curr_g


    def compute_RDF(state, printouts):
        state, rdf_list = lax.scan(run_small_simulation, state, xs=np.arange(printouts))
        g_average = np.mean(rdf_list, axis=0)
        return state, g_average

    state, _ = compute_RDF(state, conf['equilib_steps'])  # equilibrate
    state, g_average = compute_RDF(state, conf['rdf_samples'])  # rdf_list is device array of shape (n_runs, bins)
    return state, g_average


def RDF_loss(energy_params, conf_dict):
    _, g_average = RDF_simulation(energy_params, conf_dict)
    return loss(g_average, conf_dict['reference_rdf'])

def grad_normalization(curr_grad, p=10.):
    # if the magnitude of the gradient is > 2: take the log, otherwise just take the value itself --> not optimal, very small values stay very small! --> better to correct this at every step of gradient accumulation, if possible!
    # return np.where(np.abs(curr_grad) > 2., np.log(np.abs(curr_grad)) * np.sign(curr_grad), np.sign(curr_grad) * np.abs(np.log(curr_grad)) / 10.)
    return np.where(np.abs(curr_grad) > np.exp(-p), np.abs(np.log(np.abs(curr_grad))) * np.sign(curr_grad) / p, np.exp(p) * curr_grad)


def update(step, params, opt_state):
    """ Compute the gradient, update the parameters and the opt_state and return loss"""
    loss_val, grad_sigma = jvp(RDF_loss, (params,), (np.array([1., 0.]),))
    _, grad_epsilon = jvp(RDF_loss, (params,), (np.array([0., 1.]),))
    curr_grad = np.array([grad_sigma, grad_epsilon]) # / initgradient_norm  # normalize gradient with initial norm
    # curr_grad = grad_normalization(curr_grad)  # parameters are often on different orders of magnitude! Due to exploding gradients nature, the the most exploding one will extremly dominate other parameters --> correct by log!
    #curr_grad = optimizers.clip_grads(curr_grad, initgradient_norm)
    opt_state = opt_update(step, curr_grad, opt_state)  # update optimizer state after stepping
    # get_param is globally defined function --> OK since we do not change this function, only use it
    return get_params(opt_state), opt_state, loss_val, curr_grad

def check_gradient(energy_params, RDF_loss, dx=1.e-6):
    grad_findiff = onp.zeros(energy_params.size)
    for i in range(energy_params.size):
        param_plus = index_add(energy_params, i, dx)
        param_minus = index_add(energy_params, i, -dx)
        grad_findiff[i] = (RDF_loss(param_plus) - RDF_loss(param_minus)) / (2. * dx)
    return grad_findiff


################### user input ###########:
# TODO Gradient seems not to flow correctly: CHanges sign often! However, in close neighborhood of solutions gradients become much better! --> Gradient contains information
# can I debug the gradient somehow? I.e. see when it suddently changes sign?

# loss function is globally very smooth, but on a very small scale, it is rugged, as visible from comparing gradients
# computed at different finite difference lengths --> they dont converge and even give different signs often
# TODO possibilities: find non-differentiable part in simulation; otherwise: gradient smoothing of some sort
# TODO: gradient smoothing: maybe langevin dynamics und berechnung von Gradient mit mehreren seeds? Reparametrization trick?

file = '../../examples/data/confs/SPC_FW_3nm.gro'  # maybe easier to vectorize over initial condition than file strings --> dict
# need to set util.f32(0.002)?? Has impact on gradient! Maybe use x64?
dt = 0.002  # in fs
kbT = 2.49435321
mass = 18.0154
# total time verlängern, je kleiner loss ist, um informativere Loss function später zu erhalten, aber exploding gradients problem am Anfang klein zu halten --> in nähe von equilibrium sollte nicht mehr so tragisch sein
# state immer updaten, um equilibration Zeit zu verringern
total_time = 2.
t_equilib = 0.5 # equilibration time before sampling RDF
# this quickly makes gradients explode! try gradient clipping or more advanced methods to get a proper gradient for longer simulation times
# then also include equilibration into gradient computation
print_every = 0.1
# RDF parameters:
reference_rdf = onp.loadtxt('data/LJ_reference_RDF.csv')
RDF_start = 0.
RDF_cut = 1.5

# optimization parameters:
energy_params = np.array([0.2, 1.2])  # initial guess
num_updates = 50
step_size = 0.01
#############################################

# preprocess user input
steps_per_printout = int(0.1 / dt)  # print every 0.1 ps
num_equilibration = int(t_equilib / print_every)
num_printouts_production = int((total_time - t_equilib) / print_every)

nbins = reference_rdf.size  # bin size has big impact on gradients: resolving rdf properly makes loss more differentiable --> more informative gradients
dx_bin = (RDF_cut - RDF_start) / float(nbins)
rdf_bin_centers = np.linspace(RDF_start + dx_bin / 2., RDF_cut - dx_bin / 2., nbins)
rdf_bin_boundaries = np.linspace(RDF_start, RDF_cut, nbins + 1)
sigma_RDF = 2. * dx_bin  # has big impact on gradients! --> also should match RDF generating run, together with nbins

R, box = io.load_box(file)
key = random.PRNGKey(0)

# fill dictionary of configuration
conf_dict = {'R': R, 'box':box, 'key':key, 'dt':dt, 'kbT':kbT, 'mass':mass, 'steps_per_printout':steps_per_printout, 'rdf_samples':num_printouts_production, 'equilib_steps':num_equilibration,
             'rdf_bin_centers':rdf_bin_centers, 'rdf_bin_boundaries':rdf_bin_boundaries, 'sigma_RDF':sigma_RDF, 'reference_rdf':reference_rdf}


# true params: sig=0.3, eps=1.
# loss_val, grad_energy = value_and_grad(RDF_simulation)(energy_params, init_dict, key)  # naive reverse mode



# RDF_simulation = jit(RDF_simulation, static_argnums=(1,))
RDF_loss = partial(RDF_loss, conf_dict=conf_dict)
RDF_loss = jit(RDF_loss)
# forward mode can differentiate through the whole simulation easily --> optimization of few parameters easy (but might be more computational effort than tangent linear)
# backward mode can deal with many input parameters that require the gradient, but cannot AD through whole simulation
# as it gets memory bound

# debug:
# n_sigmas = 200
# n_epsis = 200
# sigmas = np.linspace(0.2, 0.4, n_sigmas)
# epsis = np.linspace(0.9, 1.1, n_epsis)
# losses = onp.zeros([n_sigmas, n_epsis])
# for i, sig in enumerate(sigmas):
#     for j, epsi in enumerate(epsis):
#         losses[i,j] = RDF_loss([sig, epsi])
#
# plt.figure()
# plt.imshow(losses)
# plt.colorbar()
# plt.savefig('loss_vis.png')

#loss_val, grad_sigma = jvp(RDF_loss, (energy_params,), (np.array([1.,0.]),))
#loss_val, grad_epsilon = jvp(RDF_loss, (energy_params,), (np.array([0.,1.]),))
# grad_energy = grad(RDF_simulation)(energy_params, init_conf_dict, key)
# grad_energy = grad(RDF_simulation)(energy_params)
# state, g_average = RDF_simulation(energy_params, conf_dict)


# print('Loss: ', loss_val, 'Grad_sigma: ', grad_sigma,'Grad_epsilon: ', grad_epsilon)
# initgradient_norm = np.linalg.norm(np.array([grad_sigma, grad_epsilon]))  # gradient increases with length of simulation --> do some initial norming to fix order of magnitude

# initialize optimizer
# TODO change momentum of adam to fit our needs of convergence in few time steps
params = energy_params
opt_init, opt_update, get_params = optimizers.adam(step_size)  # define adam
#opt_init, opt_update, get_params = optimizers.sgd(step_size)
cur_opt_state = opt_init(params)  # initialize adam
# optimize parameters
loss_history = onp.zeros(num_updates)

state, g_average_initguess = RDF_simulation(params, conf_dict)

# update = jit(update)
for step in range(num_updates):
    start_time = time.time()
    #print('Gradient Findiff e-4:', check_gradient(params, RDF_loss, dx=1.e-4))
    #print('Gradient Findiff e-5:', check_gradient(params, RDF_loss, dx=1.e-5))
    #print('Gradient Findiff e-6:', check_gradient(params, RDF_loss, dx=1.e-6))
    params, cur_opt_state, loss_val, curr_grad = update(step, params, cur_opt_state)
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

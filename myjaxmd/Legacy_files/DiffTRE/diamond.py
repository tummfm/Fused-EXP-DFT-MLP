import os
import sys

import chemtrain.neural_networks
from chemtrain.traj_util import process_printouts

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs


from jax import jit, random, device_get, checkpoint
from chemtrain.jax_md_mod import io, custom_space, custom_energy, \
    custom_quantity, custom_simulator
from chemtrain import reweighting
import jax.numpy as jnp
from jax_md import partition, simulate, space, quantity
from functools import partial
import numpy as np
import optax
import matplotlib.pyplot as plt
import time
from util import Postprocessing
import pickle


file_loc = '../../examples/data/confs/Diamond.pdb'

dt = 1.e-3
system_temperature = 298.15  # Kelvin = 25 celsius
Boltzmann_constant = 0.0083145107  # in kJ / mol K
kbT = system_temperature * Boltzmann_constant
mass = 12.011

total_time = 20.
t_equilib = 2.
print_every = 0.1

experimental_bondlength = 0.15445
experimental_density = 3512  # kg / m^3

c_11_target = 1079.  # in GPa
c_12_target = 124.
c_44_target = 578.

convert_from_GPa_to_kJ_mol_nm_3 = 10**3 / 1.66054
stiffness_targets = jnp.array([c_11_target, c_12_target, c_44_target]) * convert_from_GPa_to_kJ_mol_nm_3

RDF_cut = 0.5
rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = custom_quantity.rdf_discretization(RDF_cut, nbins=500)
rdf_dummy_target = jnp.zeros_like(rdf_bin_centers)
rdf_struct = custom_quantity.RDFParams(rdf_dummy_target, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF)

target_dict = {'bond': experimental_bondlength, 'stress': jnp.zeros((3, 3)), 'stiffness': stiffness_targets,
               'rdf': rdf_struct}

R_init, box = io.load_box(file_loc)  # initial configuration
R_init *= 0.1  # convert to nm
box *= 0.1
N, spacial_dim = R_init.shape
timing_struct = process_printouts(dt, total_time, t_equilib, print_every)
density = mass * N * 1.66054 / jnp.prod(box)
box *= (density / experimental_density) ** (1 / 3)  # adjust box size to experiment
density = mass * N * 1.66054 / jnp.prod(box)
print('Model Density:', density, 'kg/m^3. Experimental density:', experimental_density, 'kg/m^3')


key = random.PRNGKey(0)
model_init_key, simuation_init_key = random.split(key, 2)

box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
R_init = scale_fn(R_init)
# box_tensor = box_tensor + jnp.array([[0, 0.02 * box_tensor[0, 0], 0], [0, 0, 0], [0, 0, 0]])  # for manual strain and FD
displacement, shift = space.periodic_general(box_tensor)

# define prior parameters: For SW are coupled to cutoff
eps = 200
sigma = 0.14
r_cut_SW = sigma * 1.8  # a = 1.8 in original SW potential
r_cut_NN = 0.2
r_cut_nbrs = max([r_cut_SW, r_cut_NN])

box_nbrs = jnp.ones(3)  # give hypercube box with fractional coordinates
neighbor_fn = partition.neighbor_list(displacement, box_nbrs, r_cut_nbrs, dr_threshold=0.05)
nbrs_init = neighbor_fn(R_init)

# compute bond connections as check
initial_dispacement = space.map_neighbor(displacement)(R_init, R_init[nbrs_init.idx])
initial_distance = space.distance(initial_dispacement)
initial_distance = jnp.where(nbrs_init.idx != N, initial_distance, 1.)  # apply neighbor_mask
bond_i, j = jnp.nonzero(initial_distance < 0.2)
bond_j = nbrs_init.idx[bond_i, j]
bonds = jnp.hstack([bond_i.reshape([-1, 1]), bond_j.reshape([-1, 1])])
print('System with', N, 'atoms and', bonds.shape[0], 'bonds')


init_fn, GNN_energy = chemtrain.neural_networks.dimenetpp_neighborlist(displacement, R_init, nbrs_init, r_cut_NN)
init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init)
prior_fn = custom_energy.stillinger_weber_neighborlist(displacement, cutoff=r_cut_SW, sigma=sigma, epsilon=eps,
                                                       initialize_neighbor_list=False)


def energy_fn_template(energy_params):
    gnn_energy = partial(GNN_energy, energy_params)
    def energy(R, neighbor, **dynamic_kwargs):
        return gnn_energy(R, neighbor, **dynamic_kwargs) + prior_fn(R, neighbor=neighbor, **dynamic_kwargs)
    return jit(energy)


energy_fn_init = energy_fn_template(init_params)
simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift, dt=dt, kT=kbT,
                             chain_length=5, chain_steps=1)
init, apply_fn = simulator_template(energy_fn_init)
state = init(simuation_init_key, R_init, mass=mass, neighbor=nbrs_init)
init_sim_state = (state, nbrs_init)  # store neighbor list together with current simulation state


simulation_funs = (simulator_template, energy_fn_template, neighbor_fn)

quantity_dict = {}
if 'bond' in target_dict:
    bond_length_fn = custom_quantity.init_bond_length(displacement, bonds, average=True)
    quantity_dict['bond'] = {'compute_fn': bond_length_fn, 'target': target_dict['bond'], 'weight': 0.}
if 'stress' in target_dict:
    stress_fn = checkpoint(
        custom_quantity.init_virial_stress_tensor(energy_fn_template, box_tensor))
    quantity_dict['stress'] = {'compute_fn': stress_fn, 'target': target_dict['stress'], 'weight': 1.e-8}
if 'stiffness' in target_dict:
    stiffness_fn = checkpoint(custom_quantity.init_stiffness_tensor_components_cubic_crystal(
        energy_fn_template, box_tensor))
    quantity_dict['stiffness'] = {'compute_fn': stiffness_fn, 'target': target_dict['stiffness'], 'weight': 1.e-10}
if 'rdf' in target_dict:  # dummy
    rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)
    quantity_dict['rdf'] = {'compute_fn': rdf_fn, 'target': target_dict['rdf'].reference, 'weight': 0.}
if 'pressure' in target_dict:
    pressure_fn = custom_quantity.init_pressure(energy_fn_template, box_tensor)
    pressure_target_dict = {'compute_fn': pressure_fn, 'target': target_dict['pressure'], 'weight': 0.0}
    quantity_dict['pressure'] = pressure_target_dict

# initialize optimizer
num_updates = 350
initial_step_size = -0.002
lr_schedule = optax.exponential_decay(initial_step_size, 200, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

update_fn, trajectory_state = difftre.difftre_init(
          simulation_funs, timing_struct, quantity_dict, kbT, init_params, init_sim_state, optimizer)

loss_history = np.zeros(num_updates)
gradient_history = np.zeros(num_updates)
Traj_list = []
predicted_quantities = []

opt_state = optimizer.init(init_params)
params = init_params
for step in range(num_updates):
    start_time = time.time()
    params, opt_state, trajectory_state, loss_val, curr_grad, predictions = update_fn(
        step, params, opt_state, trajectory_state)
    step_time = time.time() - start_time
    _, traj, _ = trajectory_state
    loss_history[step] = loss_val
    predicted_quantities.append(predictions)
    Traj_list.append(traj.position)
    grad_norm = optax.global_norm(curr_grad)
    print('Gradient norm:', grad_norm)
    cur_state = trajectory_state[0][0]
    print('Temperature:', quantity.temperature(cur_state.velocity, mass=mass))
    print('Mean:', jnp.mean(cur_state.position, axis=0), 'Min:', jnp.min(cur_state.position), 'Max:', jnp.max(cur_state.position))
    print('Position particle 100:', cur_state.position[100], 'Position particle 200:', cur_state.position[200])
    gradient_history[step] = grad_norm

    print("Step {} in {:0.2f} sec".format(step, step_time))
    print('Loss = ', loss_val, '\n')

    if np.isnan(loss_val):  # stop learning when diverged
        print('Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup '
              'causing a NaN trajectory.')
        break


Postprocessing.plot_loss_and_gradient_history(loss_history, gradient_history, visible_device)

# save optimized energy params for re-use
energy_pickle_file_path = 'output/Energy_params_Diamond_DimeNet' + str(visible_device) + '.pkl'
final_energy_params = device_get(params)
with open(energy_pickle_file_path, 'wb') as f:
    pickle.dump(final_energy_params, f)

if 'bond' in predicted_quantities[0]:
    bond_series = [prediction_dict['bond'] for prediction_dict in predicted_quantities]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Average bond length in nm')
    ax1.plot(bond_series, label='Average bond length')
    ax1.axhline(y=experimental_bondlength, linestyle='--', label='Target', color='k')
    ax1.legend()
    plt.savefig('Figures/Bondlength_history_DimeNet' + str(visible_device) + '.png')

if 'stress' in predicted_quantities[0]:
    pressure_series = [jnp.trace(prediction_dict['stress']) / convert_from_GPa_to_kJ_mol_nm_3 / 3.e-3 for
                       prediction_dict in predicted_quantities]  # in MPa
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Hydrostatic stress in MPa')
    ax1.plot(pressure_series, label='Predicted')
    ax1.axhline(y=0.1, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('Figures/Pressure_history_DimeNet' + str(visible_device) + '.png')
    print('Predicted stress tensor:', 1000. * predicted_quantities[-1]['stress'] / convert_from_GPa_to_kJ_mol_nm_3)

if 'stiffness' in predicted_quantities[0]:
    stiffness_list = [prediction_dict['stiffness'] for prediction_dict in predicted_quantities]
    stiffness_array = jnp.stack(stiffness_list) / convert_from_GPa_to_kJ_mol_nm_3  # back to GPa
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('$C_{ij}$ in GPa')
    ax1.plot(stiffness_array[:, 0], label='$C_{11}$')
    ax1.plot(stiffness_array[:, 1], label='$C_{12}$')
    ax1.axhline(y=c_12_target, linestyle='--', color='k')
    ax1.plot(stiffness_array[:, 2], label='$C_{44}$')
    ax1.axhline(y=c_44_target, linestyle='--', color='k')
    ax1.axhline(y=c_11_target, linestyle='--', label='Targets', color='k')
    ax1.legend()
    plt.savefig('Figures/Stiffness_history_DimeNet' + str(visible_device) + '.png')

if 'rdf' in predicted_quantities[0]:
    RDF_save_dict = {'reference': rdf_struct.reference, 'x_vals': rdf_struct.rdf_bin_centers}
    rdf_series = [prediction_dict['rdf'] for prediction_dict in predicted_quantities]
    RDF_save_dict['series'] = rdf_series
    rdf_pickle_file_path = 'output/Gif/RDFs_Diamond' + str(visible_device) + '.pkl'
    with open(rdf_pickle_file_path, 'wb') as f:
        pickle.dump(RDF_save_dict, f)
    # Postprocessing.visualize_time_series('RDFs_Diamond' + str(visible_device) + '.pkl')
    Postprocessing.plot_initial_and_predicted_rdf(rdf_struct.rdf_bin_centers, rdf_series[-1], 'Diamond', visible_device,
                                                  rdf_struct.reference, rdf_series[0])

if 'pressure' in predicted_quantities[0]:
    pressure_series = [prediction_dict['pressure'] for prediction_dict in predicted_quantities]
    Postprocessing.plot_pressure_history(pressure_series, 'DimeNetP', visible_device, reference_pressure=0.)

# bond_lengths = difftre.compute_quantity_traj(trajectory_state, compute_bondlengths_dict, neighbor_fn, init_params)
# bond_hist_final, _ = jnp.histogram(bond_lengths['bond'], bins=n_bond_bins)
#
#
# plt.figure()
# plt.plot(bond_hist_bin_centers, bond_hist_initial, label='Initial bond lengths')
# plt.plot(bond_hist_bin_centers, bond_hist_final, label='Predicted bond lengths')
# plt.legend()
# plt.ylabel('Frequency')
# plt.xlabel('$r$ in nm')
# plt.savefig('Figures/Bond_length_histogram' + str(visible_device) + '.png')



import os

visible_device = 0
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
from chemtrain import trainers, traj_util, data_processing
from util import Initialization
from util import Ti_reader as ti
from chemtrain.sparse_graph import pad_forces_positions_species

import optax
from jax import vmap, random
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from pathlib import Path


# ************ General Settings ***************
date = "040324"
system_temperature_1 = 23.0
system_temperature_8 = 323.0
system_temperature_14 = 623.0
system_temperature_20 = 923.0

boltzmann_constant = 0.0083145107  # in kJ / mol K
kbT_1 = system_temperature_1 * boltzmann_constant
kbT_8 = system_temperature_8 * boltzmann_constant
kbT_14 = system_temperature_14 * boltzmann_constant
kbT_20 = system_temperature_20 * boltzmann_constant

mass = 47.867  # mass of Ti atoms in u
num_updates = 10

# ************ DiffTRe Settings**************
model_difftre = 'TiDimeNetDiffTRe'
kbt_dependent = False
checkpoint = None
integrator = 'Langevin'

unique_key_difftre = date + "_AllInitAndBulk_bs30_Init_gu1emin6_gf1emin2_gp4emin6_Bulk_gu1emin6_gf1emin2"
trained_model_path_difftre = 'output/force_matching/trained_model_Ti_' + unique_key_difftre + '.pkl'

dt_difftre = 0.5e-3
total_time_difftre = 2  # 70.
t_equilib_difftre = 1  # 10.  # discard all states within the first 5 ps as equilibration
print_every_difftre = 0.1  # save states every 0.1 ps for use in averaging

# Optimizer - TODO: Changed initial LR -0.0001 to -0.0002
check_freq_difftre = 10.
initial_lr_difftre = -0.0002
lr_schedule_difftre = optax.exponential_decay(initial_lr_difftre, num_updates, 0.1)
optimizer_difftre = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule_difftre)
)


# ***** Quantity Matching Settings ******
shuffle_bool_QM = False
shuffle_subset_bool_QM = False
only_len_16_bool_QM = False
only_len_32_bool_QM = False
only_len_128_bool_QM = False
batch_per_device_QM = 30
batch_cache_QM = 30
num_transition_steps_QM = 26626
### ****
batch_per_device_QM = 5
batch_cache_QM = 5
epochs = 10
num_transition_steps = int(epochs * 10 / batch_per_device_QM)


### **** Experimental observables
ec_exp = pd.read_csv('data/Exp_data/Exp_EC/EC_From_ExpLatticeConstants.txt', header=None, delim_whitespace=True, skiprows=1).to_numpy()[:, 2:]*100

# 23 K
c_11_target_1 = round(ec_exp[1][0], 1)  # in GPa
c_12_target_1 = round(ec_exp[1][1], 1)
c_13_target_1 = round(ec_exp[1][2], 1)
c_33_target_1 = round(ec_exp[1][3], 1)
c_44_target_1 = round(ec_exp[1][4], 1)

# 323 K
c_11_target_8 = round(ec_exp[8][0], 1)  # in GPa
c_12_target_8 = round(ec_exp[8][1], 1)
c_13_target_8 = round(ec_exp[8][2], 1)
c_33_target_8 = round(ec_exp[8][3], 1)
c_44_target_8 = round(ec_exp[8][4], 1)

# 623 K
c_11_target_14 = round(ec_exp[14][0], 1)  # in GPa
c_12_target_14 = round(ec_exp[14][1], 1)
c_13_target_14 = round(ec_exp[14][2], 1)
c_33_target_14 = round(ec_exp[14][3], 1)
c_44_target_14 = round(ec_exp[14][4], 1)

# 923 K
c_11_target_20 = round(ec_exp[20][0], 1)  # in GPa
c_12_target_20 = round(ec_exp[20][1], 1)
c_13_target_20 = round(ec_exp[20][2], 1)
c_33_target_20 = round(ec_exp[20][3], 1)
c_44_target_20 = round(ec_exp[20][4], 1)


convert_from_GPa_to_kJ_mol_nm_3 = 10 ** 3 / 1.66054
stiffness_targets_1 = jnp.array([c_11_target_1, c_33_target_1, c_44_target_1, c_12_target_1, c_13_target_1]) * \
                    convert_from_GPa_to_kJ_mol_nm_3

stiffness_targets_8 = jnp.array([c_11_target_8, c_33_target_8, c_44_target_8, c_12_target_8, c_13_target_8]) * \
                    convert_from_GPa_to_kJ_mol_nm_3

stiffness_targets_14 = jnp.array([c_11_target_14, c_33_target_14, c_44_target_14, c_12_target_14, c_13_target_14]) * \
                    convert_from_GPa_to_kJ_mol_nm_3

stiffness_targets_20 = jnp.array([c_11_target_20, c_33_target_20, c_44_target_20, c_12_target_20, c_13_target_20]) * \
                    convert_from_GPa_to_kJ_mol_nm_3

target_dict_difftre_1 = {'pressure_scalar': jnp.zeros(1,), 'stiffness': stiffness_targets_1}
target_dict_difftre_8 = {'pressure_scalar': jnp.zeros(1,), 'stiffness': stiffness_targets_8}
target_dict_difftre_14 = {'pressure_scalar': jnp.zeros(1,), 'stiffness': stiffness_targets_14}
target_dict_difftre_20 = {'pressure_scalar': jnp.zeros(1,), 'stiffness': stiffness_targets_20}

shifted_box_difftre_1 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/23K_expt_box.npy'))
shifted_pos_difftre_1 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/23K_expt_coordinates.npy'))

shifted_box_difftre_8 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/323K_expt_box.npy'))
shifted_pos_difftre_8 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/323K_expt_coordinates.npy'))

shifted_box_difftre_14 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/623K_expt_box.npy'))
shifted_pos_difftre_14 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/623K_expt_coordinates.npy'))

shifted_box_difftre_20 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/923K_expt_box.npy'))
shifted_pos_difftre_20 = jnp.array(onp.load('data/Exp_data/Exp_Boxes_AtomPositions/ExperimentalLattice_Boxes_AtomPositions/923K_expt_coordinates.npy'))

# Convert Å to nm
box_tensor_difftre_1 = shifted_box_difftre_1 * 0.1
R_init_difftre_1 = shifted_pos_difftre_1 * 0.1
species_difftre_1 = jnp.ones(R_init_difftre_1.shape[0], dtype=int)
precom_edge_mask_difftre_1 = jnp.zeros((R_init_difftre_1.shape[0], R_init_difftre_1.shape[0]-1), dtype=int)
N_atoms_difftre_1 = R_init_difftre_1.shape[0]

box_tensor_difftre_8 = shifted_box_difftre_8 * 0.1
R_init_difftre_8 = shifted_pos_difftre_8 * 0.1
species_difftre_8 = jnp.ones(R_init_difftre_8.shape[0], dtype=int)
precom_edge_mask_difftre_8 = jnp.zeros((R_init_difftre_8.shape[0], R_init_difftre_8.shape[0]-1), dtype=int)
N_atoms_difftre_8 = R_init_difftre_8.shape[0]

box_tensor_difftre_14 = shifted_box_difftre_14 * 0.1
R_init_difftre_14 = shifted_pos_difftre_14 * 0.1
species_difftre_14 = jnp.ones(R_init_difftre_14.shape[0], dtype=int)
precom_edge_mask_difftre_14 = jnp.zeros((R_init_difftre_14.shape[0], R_init_difftre_14.shape[0]-1), dtype=int)
N_atoms_difftre_14 = R_init_difftre_14.shape[0]

box_tensor_difftre_20 = shifted_box_difftre_20 * 0.1
R_init_difftre_20 = shifted_pos_difftre_20 * 0.1
species_difftre_20 = jnp.ones(R_init_difftre_20.shape[0], dtype=int)
precom_edge_mask_difftre_20 = jnp.zeros((R_init_difftre_20.shape[0], R_init_difftre_20.shape[0]-1), dtype=int)
N_atoms_difftre_20 = R_init_difftre_20.shape[0]


# Initial configuration
simulation_data_difftre_1 = Initialization.InitializationClass(R_init=R_init_difftre_1, box=box_tensor_difftre_1,
                                                               kbT=kbT_1, masses=mass, dt=dt_difftre,
                                                               species=species_difftre_1)

simulation_data_difftre_8 = Initialization.InitializationClass(R_init=R_init_difftre_8, box=box_tensor_difftre_8,
                                                               kbT=kbT_8, masses=mass, dt=dt_difftre,
                                                               species=species_difftre_8)

simulation_data_difftre_14 = Initialization.InitializationClass(R_init=R_init_difftre_14, box=box_tensor_difftre_14,
                                                               kbT=kbT_14, masses=mass, dt=dt_difftre,
                                                               species=species_difftre_14)

simulation_data_difftre_20 = Initialization.InitializationClass(R_init=R_init_difftre_20, box=box_tensor_difftre_20,
                                                                kbT=kbT_20, masses=mass, dt=dt_difftre,
                                                                species=species_difftre_20)


timings_difftre = traj_util.process_printouts(dt_difftre, total_time_difftre, t_equilib_difftre, print_every_difftre)

reference_state_difftre_1, init_params_difftre_1, simulation_fns_difftre_1, compute_fns_difftre_1, targets_difftre_1 = \
    Initialization.initialize_simulation_ti_hcp(simulation_data_difftre_1, model_difftre, target_dict_difftre_1,
                                                integrator=integrator,
                                                kbt_dependent=kbt_dependent,
                                                precom_edge_mask=precom_edge_mask_difftre_1,
                                                load_pretrained_model_path=trained_model_path_difftre)


reference_state_difftre_8, init_params_difftre_8, simulation_fns_difftre_8, compute_fns_difftre_8, targets_difftre_8 = \
    Initialization.initialize_simulation_ti_hcp(simulation_data_difftre_8, model_difftre, target_dict_difftre_8,
                                                integrator=integrator,
                                                kbt_dependent=kbt_dependent,
                                                precom_edge_mask=precom_edge_mask_difftre_8,
                                                load_pretrained_model_path=trained_model_path_difftre)

reference_state_difftre_14, init_params_difftre_14, simulation_fns_difftre_14, compute_fns_difftre_14, targets_difftre_14 = \
    Initialization.initialize_simulation_ti_hcp(simulation_data_difftre_14, model_difftre, target_dict_difftre_14,
                                                integrator=integrator,
                                                kbt_dependent=kbt_dependent,
                                                precom_edge_mask=precom_edge_mask_difftre_14,
                                                load_pretrained_model_path=trained_model_path_difftre)

reference_state_difftre_20, init_params_difftre_20, simulation_fns_difftre_20, compute_fns_difftre_20, targets_difftre_20 = \
    Initialization.initialize_simulation_ti_hcp(simulation_data_difftre_20, model_difftre, target_dict_difftre_20,
                                                integrator=integrator,
                                                kbt_dependent=kbt_dependent,
                                                precom_edge_mask=precom_edge_mask_difftre_20,
                                                load_pretrained_model_path=trained_model_path_difftre)


simulator_template_difftre_1, energy_fn_template_difftre_1, neighbor_fn_difftre_1 = simulation_fns_difftre_1
simulator_template_difftre_8, energy_fn_template_difftre_8, neighbor_fn_difftre_8 = simulation_fns_difftre_8
simulator_template_difftre_14, energy_fn_template_difftre_14, neighbor_fn_difftre_14 = simulation_fns_difftre_14
simulator_template_difftre_20, energy_fn_template_difftre_20, neighbor_fn_difftre_20 = simulation_fns_difftre_20

trainerDiffTRe = trainers.DifftreTiHCP(init_params_difftre_1,
                                  optimizer_difftre,
                                  reweight_ratio=1.0,
                                  energy_fn_template=energy_fn_template_difftre_1)

trainerDiffTRe.add_statepoint(energy_fn_template_difftre_1, simulator_template_difftre_1,
                         neighbor_fn_difftre_1, timings_difftre, kbT_1, compute_fns_difftre_1, reference_state_difftre_1,
                         targets_difftre_1, loss_fn='stiffness_and_pressure_loss', box_tensor=box_tensor_difftre_1, N=N_atoms_difftre_1,
                         target_dict=target_dict_difftre_1)

trainerDiffTRe.add_statepoint(energy_fn_template_difftre_8, simulator_template_difftre_8,
                         neighbor_fn_difftre_8, timings_difftre, kbT_8, compute_fns_difftre_8, reference_state_difftre_8,
                         targets_difftre_8, loss_fn='stiffness_and_pressure_loss', box_tensor=box_tensor_difftre_8, N=N_atoms_difftre_8,
                         target_dict=target_dict_difftre_8)

trainerDiffTRe.add_statepoint(energy_fn_template_difftre_14, simulator_template_difftre_14,
                         neighbor_fn_difftre_14, timings_difftre, kbT_14, compute_fns_difftre_14, reference_state_difftre_14,
                         targets_difftre_14, loss_fn='stiffness_and_pressure_loss', box_tensor=box_tensor_difftre_14, N=N_atoms_difftre_14,
                         target_dict=target_dict_difftre_14)

trainerDiffTRe.add_statepoint(energy_fn_template_difftre_20, simulator_template_difftre_20,
                         neighbor_fn_difftre_20, timings_difftre, kbT_20, compute_fns_difftre_20, reference_state_difftre_20,
                         targets_difftre_20, loss_fn='stiffness_and_pressure_loss', box_tensor=box_tensor_difftre_20, N=N_atoms_difftre_20,
                         target_dict=target_dict_difftre_20)

# ************* Setting up DiffTRe until here *************
print('Batch size: ', batch_per_device_QM)
print('Batch catch: ', batch_cache_QM)

check_freq_QM = None
model_QM = 'TiDimeNet'

unique_key_QM = date + "_DiffTreAndQM_batchsize30_InitAndBulk_23K_323K_623K_923K"
save_path_QM = 'output/force_matching/trained_model_Ti_' + unique_key_QM + '.pkl'


load_data_name_QM = 'InitAndBulk'
data_list_QM = ['data/DFT_data/261022_AllInitAndBulk_256atoms_with_types_curatedData']

predef_weights = onp.load(data_list_QM[0] + '/types.npy')
predef_weights = jnp.array(predef_weights)
#******************** Use input until here ********************


# Get train, val, and test set. Convert coordinates to fractional coordinates.
train_energy_QM, train_force_QM, train_virial_QM, \
    train_boxes_QM, train_coords_QM, test_energy_QM,\
    test_force_QM, test_virial_QM, test_boxes_QM, \
    test_coords_QM, val_energy_QM, val_force_QM,\
    val_virial_QM, val_boxes_QM, val_coords_QM,\
    total_test_len_QM = \
    ti.get_train_val_test_set(data_list_QM, shuffle=shuffle_bool_QM,
                              shuffle_subset=shuffle_subset_bool_QM,
                              only_consider_len_16=only_len_16_bool_QM,
                              only_consider_len_32=only_len_32_bool_QM,
                              only_consider_len_128=only_len_128_bool_QM, lammpsdata_bool=False)


# Convert to gromacs units
# scale_y = 96.49  #ev_to_kJ_per_mol
scale_energy_QM = 96.4853722                                   # [eV] ->   [kJ/mol]
scale_force_QM = scale_energy_QM / 0.1           # [eV/Å] -> [kJ/mol*nm]
scale_virial_QM = scale_energy_QM                # [eV] -> [kJ/mol]
scale_pos_QM = 0.1                            # [Å] -> [nm]

train_energy_QM *= scale_energy_QM
test_energy_QM *= scale_energy_QM
val_energy_QM *= scale_energy_QM

train_force_QM = list(onp.array(train_force_QM)*scale_force_QM)
test_force_QM = list(onp.array(test_force_QM)*scale_force_QM)
val_force_QM = list(onp.array(val_force_QM)*scale_force_QM)

train_virial_QM *= scale_virial_QM
test_virial_QM *= scale_virial_QM
val_virial_QM *= scale_virial_QM

train_boxes_QM *= scale_pos_QM
test_boxes_QM *= scale_pos_QM
val_boxes_QM *= scale_pos_QM

train_coords_QM = list(onp.array(train_coords_QM)*scale_pos_QM)
test_coords_QM = list(onp.array(test_coords_QM)*scale_pos_QM)
val_coords_QM = list(onp.array(val_coords_QM)*scale_pos_QM)


# Transpose train, validation, and test boxes
train_boxes_QM = vmap(lambda box: onp.transpose(box), in_axes=(0))(train_boxes_QM)
test_boxes_QM = vmap(lambda box: onp.transpose(box), in_axes=(0))(test_boxes_QM)
val_boxes_QM = vmap(lambda box: onp.transpose(box), in_axes=(0))(val_boxes_QM)

# Get max edges and max angles
max_edges_QM, max_triplets_QM = (10752, 440832)

train_coords_QM = data_processing.scale_dataset_fractional_varying_boxes(train_coords_QM, train_boxes_QM)
test_coords_QM = data_processing.scale_dataset_fractional_varying_boxes(test_coords_QM, test_boxes_QM)
val_coords_QM = data_processing.scale_dataset_fractional_varying_boxes(val_coords_QM, val_boxes_QM)

# Concatenate train, val, and test (order important) sets to new set
train_len_QM = len(train_energy_QM)
val_len_QM = len(val_energy_QM)
energy_QM = onp.concatenate((train_energy_QM, val_energy_QM, test_energy_QM), axis=0)
virial_QM = onp.concatenate((train_virial_QM, val_virial_QM, test_virial_QM), axis=0)
boxes_QM = onp.concatenate((train_boxes_QM, val_boxes_QM, test_boxes_QM), axis=0)
position_data_QM = train_coords_QM + val_coords_QM + test_coords_QM
forces_QM = train_force_QM + val_force_QM + test_force_QM
species_QM = [onp.ones(i.shape[0], dtype=int) for i in position_data_QM]


print('Train_len = ', train_len_QM)

# Get padded positions, forces, and species - np arrays are returned. position_data needs to be fractional!
pos_pad_QM, forces_pad_QM, species_pad_QM,\
    force_mask_QM, edge_mask_QM = pad_forces_positions_species(species=species_QM,
                                                               position_data=position_data_QM,
                                                               forces_data=forces_QM,
                                                               box=boxes_QM)


# Save the data if shuffle is False to get sorted training data for postprocessing
# if not shuffle_bool_QM:
#     Path("data/save_Ti_graphs").mkdir(parents=True, exist_ok=True)
#     onp.save('data/save_Ti_graphs/Training_energy_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', train_energy_QM)
#     onp.save('data/save_Ti_graphs/Training_virial_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', train_virial_QM)
#     onp.save('data/save_Ti_graphs/Training_boxes_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', train_boxes_QM)
#     onp.save('data/save_Ti_graphs/Training_pos_pad_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', pos_pad_QM[:train_len_QM])
#     onp.save('data/save_Ti_graphs/Training_forces_pad_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', forces_pad_QM[:train_len_QM])
#     onp.save('data/save_Ti_graphs/Training_species_pad_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', species_pad_QM[:train_len_QM])
#     onp.save('data/save_Ti_graphs/Training_edge_mask_sorted_'+load_data_name_QM+'_'+unique_key_QM+'.npy', edge_mask_QM[:train_len_QM])
#
#     # Validation data
#     onp.save('data/save_Ti_graphs/Validation_energy_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy', val_energy_QM)
#     onp.save('data/save_Ti_graphs/Validation_virial_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy', val_virial_QM)
#     onp.save('data/save_Ti_graphs/Validation_boxes_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy', val_boxes_QM)
#     onp.save('data/save_Ti_graphs/Validation_pos_pad_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              pos_pad_QM[train_len_QM:train_len_QM + val_len_QM])
#     onp.save('data/save_Ti_graphs/Validation_forces_pad_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              forces_pad_QM[train_len_QM:train_len_QM + val_len_QM])
#     onp.save('data/save_Ti_graphs/Validation_species_pad_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              species_pad_QM[train_len_QM:train_len_QM + val_len_QM])
#     onp.save('data/save_Ti_graphs/Validation_edge_mask_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              edge_mask_QM[train_len_QM:train_len_QM + val_len_QM])
#
#     # Test data
#     onp.save('data/save_Ti_graphs/Test_energy_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              test_energy_QM)
#     onp.save('data/save_Ti_graphs/Test_virial_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              test_virial_QM)
#     onp.save('data/save_Ti_graphs/Test_boxes_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              test_boxes_QM)
#     onp.save('data/save_Ti_graphs/Test_pos_pad_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              pos_pad_QM[train_len_QM + val_len_QM:])
#     onp.save('data/save_Ti_graphs/Test_forces_pad_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              forces_pad_QM[train_len_QM + val_len_QM:])
#     onp.save('data/save_Ti_graphs/Test_species_pad_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              species_pad_QM[train_len_QM + val_len_QM:])
#     onp.save('data/save_Ti_graphs/Test_edge_mask_sorted_' + load_data_name_QM + '_' + unique_key_QM + '.npy',
#              edge_mask_QM[train_len_QM + val_len_QM:])

# Init values
R_init_QM = pos_pad_QM[0]
f_init_QM = forces_pad_QM[0]
species_init_QM = species_pad_QM[0]
box_init_QM = boxes_QM[0]
edge_mask_init_QM = edge_mask_QM[0]

print('Dataset size:', pos_pad_QM.shape[0])


# Optimizer setup - TODO: Changed inital_lr from 0.001 to 0.0001
if model_QM == 'TiDimeNet':
    initial_lr_QM = 5e-7
else:
    raise NotImplementedError


lr_schedule_QM = optax.exponential_decay(-initial_lr_QM, num_transition_steps_QM, 0.01)
optimizer_QM = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule_QM)
)

# Random initialization
model_init_key_QM = random.PRNGKey(0)

energy_fn_template_QM, _, init_params_QM, _ = Initialization.select_model(
    model=model_QM, R_init=R_init_QM, displacement=None, box=box_init_QM,
    model_init_key=model_init_key_QM, species=species_init_QM,
    precom_edge_mask_init=edge_mask_init_QM, precom_max_edges=max_edges_QM,
    precom_max_triplets=max_triplets_QM)


trainer_DFT = trainers.ForceMatching_on_the_fly(init_params=init_params_QM, energy_fn_template=energy_fn_template_QM,
                                              boxes=boxes_QM, species=species_pad_QM, optimizer=optimizer_QM,
                                              position_data=pos_pad_QM, precom_edge_mask=edge_mask_QM,
                                              energy_data=energy_QM, force_data=forces_pad_QM,
                                              virial_data=virial_QM, gamma_u=None, gamma_f=None, gamma_p=None,
                                              force_mask=force_mask_QM, batch_per_device=batch_per_device_QM,
                                              batch_cache=batch_cache_QM, checkpoint_folder=unique_key_QM+'_checkpoints',
                                              train_len=train_len_QM, val_len=val_len_QM, predef_weights=predef_weights)

hybrid_trainer = trainers.hybridTrainer4temps_and_DFT(trainerDiffTRe=trainerDiffTRe, trainerDFT=trainer_DFT)

hybrid_trainer.train(num_epochs=num_updates, stiffness_path='../examples/output/figures/stiffness_name.npy',
                     pressure_path='../examples/output/figures/pressure_name.npy')



# Save energy params
path_trainerDiffTRe_energy_params = 'output/difftre/DiffTRe_energy_params_name.pkl'
path_trainerDFT_energy_params ='output/difftre/DFT_energy_params_name.pkl'


print("Save energy params DFT")
trainer_DFT.save_energy_params(file_path=path_trainerDFT_energy_params, save_format='.pkl')
print("Save energy params DiffTRe")
trainerDiffTRe.save_energy_params(file_path=path_trainerDiffTRe_energy_params, save_format='.pkl')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(trainer_DFT.train_losses[:], label='Train')
plt.plot(trainer_DFT.val_losses[:], label='Val')
plt.ylabel('MSE Loss')
plt.xlabel('Update step')
plt.savefig('output/figures/force_matching_losses'+str(unique_key_QM)+'_280423_DFT_4Exp_Pressure_DFT_200epochs_80_10ps_gf1emin2.png')
"""Do force matching on Ti data."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

from jax import random
import numpy as onp
import optax
from jax import vmap
import jax.numpy as jnp

from chemtrain import trainers, data_processing
from chemtrain.sparse_graph import pad_forces_positions_species
from util import Initialization
from util import Ti_reader as ti



shuffle_bool = False    # Don't shuffle extended dataset again, this could cause val samples to not be representative with training samples.
shuffle_subset_bool = False
only_len_16_bool = False
only_len_32_bool = False
only_len_128_bool = False
only_len_192_bool = False
batch_per_device = 30
batch_cache = 30
epochs = 400

num_transition_steps = 53253

batch_per_device = 5
batch_cache = 5
epochs = 40
num_transition_steps = int(epochs * 30 / batch_per_device)

print('Batch size: ', batch_per_device)
print('Batch catch: ', batch_cache)

check_freq = 50
model = 'TiDimeNet'

date = "040324"
unique_key = date + "_AllInitAndBulk_bs30_Init_gu1emin6_gf1emin2_gp4emin6_Bulk_gu1emin6_gf1emin2"
save_path = 'output/force_matching/trained_model_Ti_' + unique_key + '.pkl'

date = "0403224"
load_data_name = date + '_AllInitAndBulk_256atoms_with_types_curatedData'

data_list = ['data/DFT_data/261022_AllInitAndBulk_256atoms_with_types_curatedData']

predef_weights = onp.load(data_list[0] + '/types.npy')
predef_weights = jnp.array(predef_weights)
# predef_weights = None
# predef_weights = None

# Get train, val, and test set. Convert coordinates to fractional coordinates.
train_energy, train_force, train_virial, train_boxes, train_coords, test_energy, test_force, test_virial, test_boxes, \
    test_coords, val_energy, val_force, val_virial, val_boxes, val_coords, total_test_len = \
    ti.get_train_val_test_set(data_list, shuffle=shuffle_bool, shuffle_subset=shuffle_subset_bool,
                              only_consider_len_16=only_len_16_bool, only_consider_len_32=only_len_32_bool,
                              only_consider_len_128=only_len_128_bool, only_consider_len_192=only_len_192_bool,
                              lammpsdata_bool=False)


# Convert to gromacs units
scale_energy = 96.4853722                  # [eV] ->   [kJ/mol]
scale_force = scale_energy / 0.1           # [eV/Å] -> [kJ/mol*nm]
scale_virial = scale_energy                # [eV] -> [kJ/mol]
scale_pos = 0.1                            # [Å] -> [nm]

train_energy *= scale_energy
test_energy *= scale_energy
val_energy *= scale_energy

train_force = list(onp.array(train_force)*scale_force)
test_force = list(onp.array(test_force)*scale_force)
val_force = list(onp.array(val_force)*scale_force)

train_virial *= scale_virial
test_virial *= scale_virial
val_virial *= scale_virial

train_boxes *= scale_pos
test_boxes *= scale_pos
val_boxes *= scale_pos

train_coords = list(onp.array(train_coords)*scale_pos)
test_coords = list(onp.array(test_coords)*scale_pos)
val_coords = list(onp.array(val_coords)*scale_pos)


# Transpose train, validation, and test boxes
train_boxes = vmap(lambda box: onp.transpose(box), in_axes=(0))(train_boxes)
test_boxes = vmap(lambda box: onp.transpose(box), in_axes=(0))(test_boxes)
val_boxes = vmap(lambda box: onp.transpose(box), in_axes=(0))(val_boxes)

# Get max edges and max angles
# check_boxes = onp.concatenate((train_boxes, val_boxes, test_boxes), axis=0)
# check_positions = onp.concatenate((train_coords, val_coords, test_coords), axis=0)
# max_edges, max_triplets = get_max_edges_max_triplets(r_cutoff=0.5, position_data=check_positions, box=check_boxes)
# print("Max edges: ", max_edges)
# print("Max triplets: ", max_triplets)
max_edges, max_triplets = (10752, 440832)


train_coords = data_processing.scale_dataset_fractional_varying_boxes(train_coords, train_boxes)
test_coords = data_processing.scale_dataset_fractional_varying_boxes(test_coords, test_boxes)
val_coords = data_processing.scale_dataset_fractional_varying_boxes(val_coords, val_boxes)

# Concatenate train, val, and test (order important) sets to new set
train_len = len(train_energy)
val_len = len(val_energy)
train_len = None
val_len = None

energy = onp.concatenate((train_energy, val_energy, test_energy), axis=0)
virial = onp.concatenate((train_virial, val_virial, test_virial), axis=0)
boxes = onp.concatenate((train_boxes, val_boxes, test_boxes), axis=0)
position_data = train_coords + val_coords + test_coords
forces = train_force + val_force + test_force
species = [onp.ones(i.shape[0], dtype=int) for i in position_data]

print('Before padding')
# Get padded positions, forces, and species - np arrays are returned. position_data needs to be fractional!
pos_pad, forces_pad, species_pad, force_mask, edge_mask = pad_forces_positions_species(species=species,
                                                                                       position_data=position_data,
                                                                                       forces_data=forces, box=boxes)


print('After padding')


# Save the data if shuffle is False to get sorted training data for postprocessing
# if not shuffle_bool:
#     onp.save('data/save_Ti_graphs/Training_energy_sorted_'+load_data_name+'_'+unique_key+'.npy', train_energy)
#     onp.save('data/save_Ti_graphs/Training_virial_sorted_'+load_data_name+'_'+unique_key+'.npy', train_virial)
#     onp.save('data/save_Ti_graphs/Training_boxes_sorted_'+load_data_name+'_'+unique_key+'.npy', train_boxes)
#     onp.save('data/save_Ti_graphs/Training_pos_pad_sorted_'+load_data_name+'_'+unique_key+'.npy', pos_pad[:train_len])
#     onp.save('data/save_Ti_graphs/Training_forces_pad_sorted_'+load_data_name+'_'+unique_key+'.npy', forces_pad[:train_len])
#     onp.save('data/save_Ti_graphs/Training_species_pad_sorted_'+load_data_name+'_'+unique_key+'.npy', species_pad[:train_len])
#     onp.save('data/save_Ti_graphs/Training_edge_mask_sorted_'+load_data_name+'_'+unique_key+'.npy', edge_mask[:train_len])
#
#     # Validation data
#     onp.save('data/save_Ti_graphs/Validation_energy_sorted_' + load_data_name + '_' + unique_key + '.npy', val_energy)
#     onp.save('data/save_Ti_graphs/Validation_virial_sorted_' + load_data_name + '_' + unique_key + '.npy', val_virial)
#     onp.save('data/save_Ti_graphs/Validation_boxes_sorted_' + load_data_name + '_' + unique_key + '.npy', val_boxes)
#     onp.save('data/save_Ti_graphs/Validation_pos_pad_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              pos_pad[train_len:train_len + val_len])
#     onp.save('data/save_Ti_graphs/Validation_forces_pad_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              forces_pad[train_len:train_len + val_len])
#     onp.save('data/save_Ti_graphs/Validation_species_pad_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              species_pad[train_len:train_len + val_len])
#     onp.save('data/save_Ti_graphs/Validation_edge_mask_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              edge_mask[train_len:train_len + val_len])
#
#     # Test data
#     onp.save('data/save_Ti_graphs/Test_energy_sorted_' + load_data_name + '_' + unique_key + '.npy', test_energy)
#     onp.save('data/save_Ti_graphs/Test_virial_sorted_' + load_data_name + '_' + unique_key + '.npy', test_virial)
#     onp.save('data/save_Ti_graphs/Test_boxes_sorted_' + load_data_name + '_' + unique_key + '.npy', test_boxes)
#     onp.save('data/save_Ti_graphs/Test_pos_pad_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              pos_pad[train_len + val_len:])
#     onp.save('data/save_Ti_graphs/Test_forces_pad_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              forces_pad[train_len + val_len:])
#     onp.save('data/save_Ti_graphs/Test_species_pad_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              species_pad[train_len + val_len:])
#     onp.save('data/save_Ti_graphs/Test_edge_mask_sorted_' + load_data_name + '_' + unique_key + '.npy',
#              edge_mask[train_len + val_len:])


# Init values
R_init = pos_pad[0]
f_init = forces_pad[0]
species_init = species_pad[0]
box_init = boxes[0]
edge_mask_init = edge_mask[0]

print('Dataset size:', pos_pad.shape[0])

# Optimizer setup
if model == 'TiDimeNet':
    initial_lr = 0.001
else:
    raise NotImplementedError


lr_schedule = optax.exponential_decay(-initial_lr, num_transition_steps, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

# Random initialization
model_init_key = random.PRNGKey(0)

energy_fn_template, _, init_params, _ = Initialization.select_model(
    model=model, R_init=R_init, displacement=None, box=box_init, model_init_key=model_init_key,
    species=species_init, precom_edge_mask_init=edge_mask_init,
    precom_max_edges=max_edges, precom_max_triplets=max_triplets)

# Train on virial, forces, and energy
trainer = trainers.ForceMatching_on_the_fly(init_params=init_params, energy_fn_template=energy_fn_template,
                                            boxes=boxes, species=species_pad, optimizer=optimizer,
                                            position_data=pos_pad, precom_edge_mask=edge_mask,
                                            energy_data=energy, force_data=forces_pad, gamma_u=None, gamma_f=None,
                                            gamma_p=None, virial_data=virial, force_mask=force_mask,
                                            batch_per_device=batch_per_device, batch_cache=batch_cache,
                                            checkpoint_folder=unique_key+'_checkpoints', train_len=train_len,
                                            val_len=val_len, predef_weights=predef_weights)


trainer.train(epochs, checkpoint_freq=check_freq)
trainer.save_trainer(save_path)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(trainer.train_losses[1:], label='Train')
plt.plot(trainer.val_losses[1:], label='Val')
plt.ylabel('MSE Loss')
plt.xlabel('Update step')
plt.savefig('output/figures/force_matching_losses'+str(unique_key)+'.png')


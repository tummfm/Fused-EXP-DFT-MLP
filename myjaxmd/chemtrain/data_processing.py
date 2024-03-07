"""Functions that facilitate common data processing operations for machine
learning.
"""
import numpy as onp
import pandas as pd
from jax import tree_flatten, lax
from jax_sgmc import data

from chemtrain.jax_md_mod import custom_space
from chemtrain import util


def get_dataset(data_location_str, retain=None, subsampling=1):
    """Loads .pyy numpy dataset.

    Args:
        data_location_str: String of .npy data location
        retain: Number of samples to keep in the dataset
        subsampling: Only keep every subsampled sample of the data, e.g. 2.

    Returns:
        Subsampled data array
    """
    loaded_data = onp.load(data_location_str)
    loaded_data = loaded_data[:retain:subsampling]
    return loaded_data


def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.1):
    """Train-validation-test split for datasets. Works on arbitrary pytrees,
    including chex.dataclasses, dictionaries and single arrays.

    Args:
        dataset: Dataset pytree. Samples are assumed to be stacked along
                 axis 0.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.

    Returns:
        Tuple (train_data, val_data, test_data) with the same shape as the input
        pytree, but split along axis 0.
    """
    leaves, _ = tree_flatten(dataset)
    dataset_size = leaves[0].shape[0]
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    train_data = util.tree_get_slice(dataset, 0, train_size)
    val_data = util.tree_get_slice(dataset, train_size, train_size + val_size)
    test_data = util.tree_get_slice(dataset, train_size + val_size, None)
    return train_data, val_data, test_data


def train_val_test_split_precomputed_dataset(dataset, train_len, val_len):
    """Train-validation-test split for datasets. Works on arbitrary pytrees,
    including chex.dataclasses, dictionaries and single arrays.

    Args:
        dataset: Dataset pytree. Samples are assumed to be stacked along
                 axis 0.
        train_len: length of dataset to use for training.
        val_len: length of dataset to use for validation.

    Returns:
        Tuple (train_data, val_data, test_data) with the same shape as the input
        pytree, but split along axis 0.
    """
    leaves, _ = tree_flatten(dataset)
    train_size = train_len
    val_size = val_len
    train_data = util.tree_get_slice(dataset, 0, train_size)
    val_data = util.tree_get_slice(dataset, train_size, train_size + val_size)
    test_data = util.tree_get_slice(dataset, train_size + val_size, None)
    return train_data, val_data, test_data


def init_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1):
    """Splits dataset and initializes dataloaders.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
                 returns batches with the same kwargs as provided in dataset.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.

    Returns:
        A tuple (train_loader, val_loader, test_loader) of NumpyDataLoaders.
    """
    train_set, val_set, test_set = train_val_test_split(
        dataset, train_ratio, val_ratio)
    train_loader = data.NumpyDataLoader(**train_set)
    val_loader = data.NumpyDataLoader(**val_set)
    test_loader = data.NumpyDataLoader(**test_set)
    return train_loader, val_loader, test_loader


def init_dataloaders_precomputed_dataset(dataset, train_len, val_len):
    """Splits dataset and initializes dataloaders.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
                 returns batches with the same kwargs as provided in dataset.
        train_len: Length of dataset to use for training.
        val_len: Length of dataset to use for validation.

    Returns:
        A tuple (train_loader, val_loader, test_loader) of NumpyDataLoaders.
    """
    train_set, val_set, test_set = train_val_test_split_precomputed_dataset(
        dataset, train_len, val_len)
    train_loader = data.NumpyDataLoader(**train_set)
    val_loader = data.NumpyDataLoader(**val_set)
    test_loader = data.NumpyDataLoader(**test_set)
    return train_loader, val_loader, test_loader


def scale_dataset_fractional(traj, box):
    """Scales a dataset of positions from real space to fractional coordinates.

    Args:
        traj: A (N_snapshots, N_particles, 3) array of particle positions
        box: A 1 or 2-dimensional jax_md box

    Returns:
        A (N_snapshots, N_particles, 3) array of particle positions in
        fractional coordinates.
    """
    _, scale_fn = custom_space.init_fractional_coordinates(box)
    scaled_traj = lax.map(scale_fn, traj)
    return scaled_traj


def scale_dataset_fractional_varying_boxes(traj, boxes):
    """Scales a dataset of positions from real space to fractional coordinates.

    Args:
        traj: A (N_snapshots, N_particles, 3) array of particle positions
        boxes: A (N_snapshots, 1 or 2-dimensional jax_md box)

    Returns:
        A (N_snapshots, N_particles, 3) array of particle positions in
        fractional coordinates.
    """
    scaled_traj = []
    for i in range(len(boxes)):
        # Old version commmented out
        #_, scale_fn = custom_space.init_fractional_coordinates(boxes[i])
        # temp_scaled_traj = lax.map(scale_fn, traj[i])
        scale_fn = custom_space.fractional_coordinates_triclinic_box(boxes[i])
        temp_scaled_traj = scale_fn(traj[i])
        scaled_traj.append(temp_scaled_traj)
    return scaled_traj


def read_xyz_file(file_path):
    xyz_file = pd.read_csv(file_path, header=0, names=['name', 'x', 'y', 'z'], skip_blank_lines=True,
                           delim_whitespace=True)
    x = xyz_file['x'].to_numpy(dtype='float32')
    y = xyz_file['y'].to_numpy(dtype='float32')
    z = xyz_file['z'].to_numpy(dtype='float32')

    pos_arr = onp.array([[x[i], y[i], z[i]] for i in range(0, len(x))])

    return pos_arr

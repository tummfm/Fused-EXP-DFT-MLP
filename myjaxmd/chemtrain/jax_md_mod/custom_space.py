"""Custom functions simplifying the handling of fractional coordinates."""
from typing import Union, Tuple, Callable

from jax_md import space, util
import jax.numpy as jnp
from jax import vmap

Box = Union[float, util.Array]


def _rectangular_boxtensor(box: Box) -> Box:
    """Transforms a 1-dimensional box to a 2D box tensor."""
    spatial_dim = box.shape[0]
    return jnp.eye(spatial_dim).at[jnp.diag_indices(spatial_dim)].set(box)


def init_fractional_coordinates(box: Box) -> Tuple[Box, Callable]:
    """Returns a 2D box tensor and a scale function that projects positions
    within a box in real space to the unit-hypercube as required by fractional
    coordinates.

    Args:
        box: A 1 or 2-dimensional box

    Returns:
        A tuple (box, scale_fn) of a 2D box tensor and a scale_fn that scales
        positions in real-space to the unit hypercube.
    """
    if box.ndim != 2:  # we need to transform to box tensor
        box_tensor = _rectangular_boxtensor(box)
    else:
        box_tensor = box
    inv_box_tensor = space.inverse(box_tensor)
    scale_fn = lambda positions: jnp.dot(positions, inv_box_tensor)
    return box_tensor, scale_fn


def fractional_coordinates_triclinic_box(box: Box) -> Callable:
    """The function init_fractional_coordinates(box: Box) scales positions by
    multiplying jnp.dot(positions, inv_box_tensor), which only works for the
    case of a diagonal box. For a triclinic box this is False XInv(T) != Inv(T)X. It has to be
    X = Tu -> Inv(T)X = u, doing

     Given X=Tu, where X is the real space of a triclinic box, u is the unit box and T is the
     transformation matrix, function returns a Callable to take in atom positions in X and applying
     Inv(T)X to it to scale to unit cube.

     Args:
         box: A 2-dimensional box, can as well be triclinic.

    Returns:
         A scale_fn that scales positions in real-space X to the unit hypercube.
     """
    assert box.ndim == 2, 'The function fractional_coordindates_triclinic_box only works with 2 dimensions'
    inv_box = jnp.linalg.inv(box)
    scale_fn = lambda positions: vmap(lambda in_transf, in_pos: jnp.dot(in_transf, in_pos), in_axes=(None, 0))(inv_box, positions)
    return scale_fn

"""Molecular dynamics observable functions acting on trajectories rather than
single snapshots.

Builds on the TrajectoryState object defined in traj_util.py.
"""
from jax import vmap, numpy as jnp
from jax_md import simulate, quantity

from chemtrain.jax_md_mod import custom_quantity


def init_traj_mean_fn(quantity_key):
    """Initializes the 'traj_fn' for the DiffTRe 'target' dict for simple
    trajectory-averaged observables.

    This function builds the 'traj_fn' of the DiffTRe 'target' dict for the
    common case of target observables that simply consist of a
    trajectory-average of instantaneous quantities, such as RDF, ADF, pressure
    or density.

    This function also serves as a template on how to build the 'traj_fn' for
    observables that are a general function of one or many instantaneous
    quantities, such as stiffness via the stress-fluctuation method or
    fluctuation formulas in this module. The 'traj_fn' receives a dict of all
    quantity trajectories as input under the same keys as instantaneous
    quantities are defined in 'quantities'. The 'traj_fn' then returns the
    ensemble-averaged quantity, possibly taking advantage of fluctuation
    formulas defined in the traj_quantity module.

    Args:
        quantity_key: Quantity key used in 'quantities' to generate the
                      quantity trajectory at hand, to be averaged over.

    Returns:
        The 'traj_fn' to be used in building the 'targets' dict for DiffTRe.
    """
    def traj_mean(quantity_trajs):
        quantity_traj = quantity_trajs[quantity_key]
        return jnp.mean(quantity_traj, axis=0)
    return traj_mean


def volumes(traj_state):
    """Returns array of volumes for all boxes in a NPT trajectory.

    Args:
        traj_state: TrajectoryState containing the NPT trajectory
    """
    dim = traj_state.sim_state[0].position.shape[-1]
    boxes = vmap(simulate.npt_box)(traj_state.trajectory)
    return vmap(quantity.volume, (None, 0))(dim, boxes)


def comp_stiffness_tensor(born_stress_traj, born_stiffness_traj, energy_fn_template, box_tensor, kbT, N, weights):
    """ Returns the stiffness tensor via the energy-strain method."""
    born_stiffness_fn, sigma_born, sigma_tensor_prod, stiffness_tensor_fn = \
        custom_quantity.init_stiffness_tensor_stress_fluctuation(energy_fn_template,
                                                                 box_tensor, kbT, N)

    def reweighting_average(quantity_snapshots):
        weighted_snapshots = (quantity_snapshots.T * weights).T
        return jnp.sum(weighted_snapshots, axis=0)

    # compute all contributions to stiffness tensor
    born_stress_prod_snapshots = sigma_tensor_prod(born_stress_traj)
    stress_product_born_mean = reweighting_average(born_stress_prod_snapshots)  # <sigma^B_ij sigma^B_kl>
    born_stress_tensor_mean = reweighting_average(born_stress_traj)
    born_stiffness_mean = reweighting_average(born_stiffness_traj)
    stiffness_tensor = stiffness_tensor_fn(born_stiffness_mean,
                                           born_stress_tensor_mean,
                                           stress_product_born_mean)
    stiffness_components = custom_quantity.stiffness_tensor_components_hexagonal_crystal(stiffness_tensor)


    return stiffness_components


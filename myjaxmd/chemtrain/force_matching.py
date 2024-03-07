"""Functions for learning via direct matching of per-snapshot quantities,
such as energy, forces and virial pressure.
"""
from collections import namedtuple
import sys

from jax import vmap, value_and_grad, numpy as jnp, device_count

from chemtrain import max_likelihood, util
from chemtrain.jax_md_mod import custom_quantity

# Note:
#  Computing the neighborlist in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterwards. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepted for now.
#  A more efficient implementation is based on pre-computation of neighborlists
#  for each snapshot in the dataset.

State = namedtuple(
    'State',
    ['position']
)
State.__doc__ = """Emulates structure of simulation state for
compatibility with quantity functions.

position: atomic positions
"""


def build_dataset(position_data, energy_data=None, force_data=None,
                  virial_data=None):
    """Builds the force-matching dataset depending on available data.

    Interface of force-loss function depends on dict keys set here.
    """
    dataset = {'R': position_data}
    if energy_data is not None:
        dataset['U'] = energy_data
    if force_data is not None:
        dataset['F'] = force_data
    if virial_data is not None:
        dataset['p'] = virial_data
    return dataset


def build_dataset_on_the_fly(position_data, box, species, energy_data=None, force_data=None,
                  virial_data=None, force_mask=None, precom_edge_mask=None,
                             predef_weights=None):
    """Builds the force-matching dataset depending on available data.

    Interface of force-loss function depends on dict keys set here.
    """
    dataset = {'R': position_data, 'box': box, 'species': species}
    if energy_data is not None:
        dataset['U'] = energy_data
    if force_data is not None:
        dataset['F'] = force_data
    if virial_data is not None:
        dataset['p'] = virial_data
    if force_mask is not None:
        dataset['force_mask'] = force_mask
    if precom_edge_mask is not None:
        dataset['precom_edge_mask'] = precom_edge_mask
    if predef_weights is not None:
        dataset['predef_weights'] = predef_weights
    return dataset


def init_virial_fn(virial_data, energy_fn_template, box_tensor):
    """Initializes the correct virial function depending on the target data
    type.
    """
    if virial_data is not None:
        assert box_tensor is not None, ('If the virial is to be matched, '
                                        'box_tensor is a mandatory input.')
        if virial_data.ndim == 3:
            virial_fn = custom_quantity.init_virial_stress_tensor(
                energy_fn_template, box_tensor, include_kinetic=False,
                pressure_tensor=True
            )
        elif virial_data.ndim in [1, 2]:
            virial_fn = custom_quantity.init_pressure(
                energy_fn_template, box_tensor, include_kinetic=False)
        else:
            raise ValueError('Format of virial dataset incompatible.')
    else:
        virial_fn = None

    return virial_fn


def init_virial_fn_on_the_fly(virial_data, energy_fn_template):
    """Initializes the correct virial function depending on the target data
    type.
    """
    if virial_data is not None:
        # assert box_tensor is not None, ('If the virial is to be matched, '
        #                                 'box_tensor is a mandatory input.')
        if virial_data.ndim == 3:
            virial_fn = custom_quantity.init_virial_stress_tensor_on_the_fly(
                energy_fn_template, include_kinetic=False,
                pressure_tensor=False, only_virial=True
            )
        # elif virial_data.ndim in [1, 2]:
        #     virial_fn = custom_quantity.init_pressure(
        #         energy_fn_template, box_tensor, include_kinetic=False)
        else:
            raise ValueError('Format of virial dataset incompatible.')
    else:
        virial_fn = None

    return virial_fn


def init_single_prediction(nbrs_init, energy_fn_template, virial_fn=None):
    """Initialize predictions for a single snapshot. Can be used to
    parametrize potentials from per-snapshot energy, force and/or virial.
    """
    def single_prediction(params, positions):
        energy_fn = energy_fn_template(params)
        # TODO check for neighborlist overflow and hand through
        nbrs = nbrs_init.update(positions)
        energy, negative_forces = value_and_grad(energy_fn)(positions,
                                                            neighbor=nbrs)
        predictions = {'U': energy, 'F': -negative_forces}
        if virial_fn is not None:
            predictions['p'] = - virial_fn(State(positions), nbrs, params)
        return predictions
    return single_prediction


def init_loss_fn(energy_fn_template, nbrs_init, gamma_u=1.,
                 gamma_f=1., gamma_p=1.e-6, virial_fn=None,
                 error_fn=max_likelihood.mse_loss):
    """Initializes update functions for energy and/or force matching.

    The returned functions are jit and can therefore not be pickled.

    Args:
        energy_fn_template: Energy function template
        nbrs_init: Initial neighbor list
        gamma_u: Weight for potential energy loss component
        gamma_f: Weight for force loss component
        gamma_p: Weight for virial loss component
        virial_fn: Function to compute virial pressure
        error_fn: Function quantifying the deviation of the model and the
                  targets. By default, a mean-squared error.

    Returns:
        A tuple (batch_update, batched_loss_fn) of pmapped functions. The former
        computes the gradient and updates the parameters via the optimizer.
        The latter returns the loss value, e.g. for the validation set.
    """
    single_prediction = init_single_prediction(nbrs_init, energy_fn_template,
                                               virial_fn)

    def loss_fn(params, batch, mask=None):
        if mask is None:  # only used for full_data_map for validation
            mask = jnp.ones(util.tree_multiplicity(batch))

        predictions = vmap(single_prediction, in_axes=(None, 0))(params,
                                                                 batch['R'])
        loss = 0.
        if 'U' in batch.keys():  # energy loss component
            u_mask = jnp.ones_like(predictions['U']) * mask
            loss += gamma_u * error_fn(predictions['U'], batch['U'], u_mask)
        if 'F' in batch.keys():  # forces loss component
            f_mask = jnp.ones_like(predictions['F']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            loss += gamma_f * error_fn(predictions['F'], batch['F'], f_mask)
        if 'p' in batch.keys():  # virial loss component
            p_mask = jnp.ones_like(predictions['p']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            loss += gamma_p * error_fn(predictions['p'], batch['p'], p_mask)
        return loss
    return loss_fn


def init_single_prediction_on_the_fly(energy_fn_template, virial_fn=None):
    """Initialize predictions for a single snapshot. Can be used to
    parametrize potentials from per-snapshot energy, force (and/or virial - not yet).
    """
    def single_prediction_on_the_fly(params, positions, box, species, precom_edge_mask):
        energy_fn = energy_fn_template(params)
        # TODO check for neighborlist overflow and hand through
        # nbrs = nbrs_init.update(positions)
        energy, negative_forces = value_and_grad(energy_fn)(positions,
                                                            box=box,
                                                            species=species,
                                                            precom_edge_mask=precom_edge_mask)
        predictions = {'U': energy, 'F': -negative_forces}
        if virial_fn is not None:
            # #TODO-Attention: This is currently not implemented for on the fly
            # sys.exit('virial_fn is not implemented for on the fly')
            # predictions['p'] = - virial_fn(State(positions), nbrs, params)
            predictions['p'] = - virial_fn(state=State(positions), box=box, species=species,
                                           precom_edge_mask=precom_edge_mask, energy_params=params)
        return predictions
    return single_prediction_on_the_fly


def init_loss_fn_on_the_fly(energy_fn_template, gamma_u=1.,
                            gamma_f=1., gamma_p=1.e-6, virial_fn=None,
                            error_fn=max_likelihood.mse_loss_predef_weights):
    """Initializes update functions for energy and/or force matching.

    The returned functions are jit and can therefore not be pickled.

    Args:
        energy_fn_template: Energy function template
        gamma_u: Weight for potential energy loss component
        gamma_f: Weight for force loss component
        gamma_p: Weight for virial loss component
        virial_fn: Function to compute virial pressure
        error_fn: Function quantifying the deviation of the model and the
                  targets. By default, a mean-squared error.

    Returns:
        A tuple (batch_update, batched_loss_fn) of pmapped functions. The former
        computes the gradient and updates the parameters via the optimizer.
        The latter returns the loss value, e.g. for the validation set.
    """
    single_prediction_on_the_fly = init_single_prediction_on_the_fly(energy_fn_template,
                                                                     virial_fn)

    def loss_fn_on_the_fly(params, batch, mask=None):
        # if mask is None:  # only used for full_data_map for validation
        #     mask = jnp.ones(util.tree_multiplicity(batch))
        if mask is None:  # only used for full_data_map for validation
            mask = jnp.ones(util.tree_multiplicity(batch))

        predictions = vmap(single_prediction_on_the_fly, in_axes=(None, 0, 0, 0, 0))(params,
                                                                                     batch['R'],
                                                                                     batch['box'],
                                                                                     batch['species'],
                                                                                     batch['precom_edge_mask'])


        loss = 0.

        # Set default predef_weights which are just ones
        u_weight = jnp.ones(util.tree_multiplicity(batch))
        f_weight = jnp.ones(util.tree_multiplicity(batch))
        p_weight = jnp.ones(util.tree_multiplicity(batch))
        g_u = gamma_u
        g_f = gamma_f
        g_p = gamma_p
        if 'predef_weights' in batch.keys():
            # Below values are hard coded. predef_weights contains 0 for Init and 1 for Bulk.
            predef_weights = batch['predef_weights']
            u_weight = jnp.where(predef_weights, 1.e-6, 1.e-6)
            # f_weight = jnp.where(predef_weights, 1.e-4, 1.e-4)
            f_weight = jnp.where(predef_weights, 1.e-2, 1.e-2)
            p_weight = jnp.where(predef_weights, 0., 4.e-6)
            g_u = 1.0
            g_f = 1.0
            g_p = 1.0
        if 'U' in batch.keys():  # energy loss component
            # u_mask = jnp.ones_like(predictions['U']) * mask
            loss += g_u * error_fn(predictions['U'], batch['U'], mask=None, predef_weights=u_weight)
        if 'F' in batch.keys():  # forces loss component
            # f_mask = jnp.ones_like(predictions['F']) * mask[:, jnp.newaxis,
            #                                                 jnp.newaxis] #* force_mask
            non_zero_f = jnp.count_nonzero(batch['F'])
            zero_f = jnp.count_nonzero(batch['F'] == 0)

            loss += g_f * error_fn(predictions['F'], batch['F'], mask=None, predef_weights=f_weight[:, jnp.newaxis,
                                                                                            jnp.newaxis]) * (zero_f + non_zero_f) / non_zero_f
        if 'p' in batch.keys():  # virial loss component
            # p_mask = jnp.ones_like(predictions['p']) * mask[:, jnp.newaxis,
            #                                                 jnp.newaxis]
            loss += g_p * error_fn(predictions['p'], batch['p'], mask=None, predef_weights=p_weight[:, jnp.newaxis,
                                                                                            jnp.newaxis])
        return loss
    return loss_fn_on_the_fly


def init_mae_fn(val_loader, nbrs_init, energy_fn_template, batch_size=1,
                batch_cache=1, virial_fn=None):
    """Returns a function that computes for each observable - energy, forces and
    virial (if applicable) - the individual mean absolute error on the
    validation set. These metrics are usually better interpretable than a
    (combined) MSE loss value.
    """
    single_prediction = init_single_prediction(nbrs_init, energy_fn_template,
                                               virial_fn)

    def abs_error(params, batch, mask):
        predictions = vmap(single_prediction, in_axes=(None, 0))(params,
                                                                 batch['R'])
        maes = {}
        if 'U' in batch.keys():  # energy loss component
            u_mask = jnp.ones_like(predictions['U']) * mask
            maes['energy'] = max_likelihood.mae_loss(predictions['U'],
                                                     batch['U'], u_mask)
        if 'F' in batch.keys():  # forces loss component
            f_mask = jnp.ones_like(predictions['F']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['forces'] = max_likelihood.mae_loss(predictions['F'],
                                                     batch['F'], f_mask)
        if 'p' in batch.keys():  # virial loss component
            p_mask = jnp.ones_like(predictions['p']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['pressure'] = max_likelihood.mae_loss(predictions['p'],
                                                       batch['p'], p_mask)
        return maes

    mean_abs_error, init_data_state = max_likelihood.val_loss_fn(
        abs_error, val_loader, device_count(), batch_size, batch_cache)

    return mean_abs_error, init_data_state


def init_mae_fn_on_the_fly(val_loader, energy_fn_template, batch_size=1,
                           batch_cache=1, virial_fn=None):
    """Returns a function that computes for each observable - energy, forces and
    virial (if applicable) - the individual mean absolute error on the
    validation set. These metrics are usually better interpretable than a
    (combined) MSE loss value.
    """
    single_prediction_on_the_fly = init_single_prediction_on_the_fly(energy_fn_template,
                                                                     virial_fn)

    def abs_error(params, batch, mask):
        # if mask is not None:
        #     sys.exit("In loss_fn_on_the_fly passed mask, but this is overwritten!"
        #              "If this occours figure where mask is passed.")
        #
        # # Convert species into mask: 0=False(padding) - 1,2,...=True(true atoms)
        # mask = jnp.array(batch['species'], dtype=bool)

        predictions = vmap(single_prediction_on_the_fly, in_axes=(None, 0, 0, 0, 0))(params,
                                                                            batch['R'],
                                                                            batch['box'],
                                                                            batch['species'],
                                                                            batch['precom_edge_mask'])
        maes = {}
        if 'U' in batch.keys():  # energy loss component
            u_mask = jnp.ones_like(predictions['U']) * mask
            maes['energy'] = max_likelihood.mae_loss(predictions['U'],
                                                     batch['U'], u_mask)
        if 'F' in batch.keys():  # forces loss component
            f_mask = jnp.ones_like(predictions['F']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['forces'] = max_likelihood.mae_loss(predictions['F'],
                                                     batch['F'], f_mask)
        if 'p' in batch.keys():  # virial loss component
            p_mask = jnp.ones_like(predictions['p']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['pressure'] = max_likelihood.mae_loss(predictions['p'],
                                                       batch['p'], p_mask)
        return maes

    mean_abs_error, init_data_state = max_likelihood.val_loss_fn(
        abs_error, val_loader, device_count(), batch_size, batch_cache)

    return mean_abs_error, init_data_state

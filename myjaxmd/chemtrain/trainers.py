"""This file contains several Trainer classes as a quickstart for users."""
import warnings

# from blackjax import nuts, stan_warmup
from coax.utils._jit import jit
from jax import value_and_grad, random, numpy as jnp
from jax_sgmc import data
import numpy as onp
import jax

from chemtrain import (util, force_matching, traj_util, reweighting,
                       probabilistic, max_likelihood, property_prediction)


class PropertyPrediction(max_likelihood.DataParallelTrainer):
    """Trainer for direct prediction of molecular properties."""
    def __init__(self, error_fn, model, init_params, optimizer, graph_dataset,
                 targets, batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        # TODO documentation
        checkpoint_path = 'output/property_prediction/' + str(checkpoint_folder)
        dataset = self._build_dataset(targets, graph_dataset)
        loss_fn = property_prediction.init_loss_fn(model, error_fn)

        super().__init__(dataset, loss_fn, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio,
                         convergence_criterion=convergence_criterion)

    @staticmethod
    def _build_dataset(targets, graph_dataset):
        return property_prediction.build_dataset(targets, graph_dataset)


class ForceMatching(max_likelihood.DataParallelTrainer):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.

    Virial data is pressure tensor, i.e. negative stress tensor

    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, energy_data=None, force_data=None,
                 virial_data=None, box_tensor=None, gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        dataset = self._build_dataset(position_data, energy_data, force_data,
                                      virial_data)

        virial_fn = force_matching.init_virial_fn(
            virial_data, energy_fn_template, box_tensor)
        loss_fn = force_matching.init_loss_fn(
            energy_fn_template, nbrs_init, gamma_f=gamma_f,
            gamma_p=gamma_p, virial_fn=virial_fn
        )

        super().__init__(dataset, loss_fn, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio,
                         convergence_criterion=convergence_criterion,
                         energy_fn_template=energy_fn_template)

        self.mae_fn, self.mae_init_state = force_matching.init_mae_fn(
            self.test_loader, nbrs_init, energy_fn_template,
            self.batch_size, batch_cache, virial_fn
        )

    @staticmethod
    def _build_dataset(position_data, energy_data=None, force_data=None,
                       virial_data=None):
        return force_matching.build_dataset(position_data, energy_data,
                                            force_data, virial_data)

    def evaluate_mae_testset(self):
        maes, self.mae_init_state = self.mae_fn(self.state.params,
                                                self.mae_init_state)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')


class ForceMatching_on_the_fly(max_likelihood.DataParallelTrainer_precomputed_dataset):
    """Force-matching trainer.

    # This implementation assumes a constant number of particles per box and
    # constant box sizes for each snapshot.
    # If this is not the case, please use the ForceMatchingPrecomputed trainer
    # based on padded sparse neighborlists.
    # Caution: Currently neighborlist overflow is not checked.
    # Make sure to build nbrs_init large enough.

    Virial data is pressure tensor, i.e. negative stress tensor

    """
    def __init__(self, init_params, energy_fn_template, boxes, species,
                 optimizer, position_data, precom_edge_mask=None, energy_data=None, force_data=None,
                 virial_data=None, force_mask=None, box_tensor=None, gamma_u=1., gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.7,
                 val_ratio=0.1, convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints', train_len=None, val_len=None,
                 predef_weights=None):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        # TODO-Done: Sebi Add boxes to _build_dataset. Change (_)build_dataset accordingly.
        # TODO-Done: Sebi Add species to _build_dataset. Change (_)build_dataset accordingly.
        dataset = self._build_dataset(position_data=position_data, box_data=boxes,
                                      species=species,  energy_data=energy_data,
                                      force_data=force_data, virial_data=virial_data,
                                      force_mask=force_mask, precom_edge_mask=precom_edge_mask,
                                      predef_weights=predef_weights)

        # # TODO-Attention: virial_fn compatibility is currently not checked for on_the_fly.
        # virial_fn = force_matching.init_virial_fn(
        #     virial_data, energy_fn_template, box_tensor)

        virial_fn_on_the_fly = force_matching.init_virial_fn_on_the_fly(
            virial_data, energy_fn_template)

        # TODO-Done: sebi Write force_matching.init_loss_fn_on_the_fly
        # loss_fn = force_matching.init_loss_fn_on_the_fly(
        #     energy_fn_template=energy_fn_template, gamma_f=gamma_f, gamma_p=gamma_p,
        #     virial_fn=virial_fn
        # )

        loss_fn = force_matching.init_loss_fn_on_the_fly(
            energy_fn_template=energy_fn_template, gamma_u=gamma_u, gamma_f=gamma_f, gamma_p=gamma_p,
            virial_fn=virial_fn_on_the_fly
        )

        super().__init__(dataset, loss_fn, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio,
                         convergence_criterion=convergence_criterion,
                         energy_fn_template=energy_fn_template,
                         train_len=train_len, val_len=val_len)

        # self.mae_fn, self.mae_init_state = force_matching.init_mae_fn(
        #     self.test_loader, nbrs_init, energy_fn_template,
        #     self.batch_size, batch_cache, virial_fn
        # )
        # TODO - sebi Write force_matching.init_mae_fn_on_the_fly
        self.mae_fn, self.mae_init_state = force_matching.init_mae_fn_on_the_fly(
            self.test_loader, energy_fn_template,
            self.batch_size, batch_cache, virial_fn_on_the_fly
        )

    @staticmethod
    def _build_dataset(position_data, box_data, species, energy_data=None, force_data=None,
                       virial_data=None, force_mask=None, precom_edge_mask=None,
                       predef_weights=None):
        return force_matching.build_dataset_on_the_fly(position_data=position_data, box=box_data,
                                                       species=species, energy_data=energy_data,
                                                       force_data=force_data, virial_data=virial_data,
                                                       force_mask=force_mask, precom_edge_mask=precom_edge_mask,
                                                       predef_weights=predef_weights)

    def evaluate_mae_testset(self):
        maes, self.mae_init_state = self.mae_fn(self.state.params,
                                                self.mae_init_state)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')


class Difftre(reweighting.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method."""
    def __init__(self, init_params, optimizer, reweight_ratio=0.9,
                 sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """Initializes a DiffTRe trainer instance.

        The implementation assumes a NVT ensemble in weight computation.
        The trainer initialization only sets the initial trainer state
        as well as checkpointing and save-functionality. For training,
        target state points with respective simulations need to be added
        via 'add_statepoint'.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes a new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the maximum loss
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average loss across the batch. For a
                                   single state point, both are equivalent.
                                   A criterion based on the rolling standatd
                                   deviation 'std' might be implemented in the
                                   future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        self.batch_losses, self.epoch_losses = [], []
        self.predictions = {}
        self.early_stop = max_likelihood.EarlyStopping(convergence_criterion)
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size,
            energy_fn_template=energy_fn_template)

    def add_statepoint(self, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, quantities,
                       reference_state, targets=None, ref_press=None,
                       loss_fn=None, initialize_traj=True):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. In case the default loss function should
        be employed, for each observable the 'target' dict containing
        a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target' needs to be provided.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness),
        a custom loss_fn needs to be defined. The signature of the loss_fn
        needs to be the following: It takes the trajectory of computed
        instantaneous quantities saved in a dict under its respective key of
        the quantities_dict. Additionally, it receives corresponding weights
        w_i to perform ensemble averages under the reweighting scheme. With
        these components, ensemble averages of more complex observables can
        be computed. The output of the function is (loss value, predicted
        ensemble averages). The latter is only necessary for post-processing
        the optimization process. See 'init_independent_mse_loss_fn' for
        an example implementation.

        Args:
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            quantities: Dict containing for each observables specified by the
                        key a corresponding function to compute it for each
                        snapshot using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                     another dict providing 'gamma' and 'target' for each
                     observable. Targets are only necessary when using the
                     'independent_loss_fn'.
            loss_fn: Custom loss function taking the trajectory of quantities
                     and weights and returning the loss and predictions;
                     Default None initializes an independent MSE loss, which
                     computes reweighting averages from snapshot-based
                     observables.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
        """

        # init simulation, reweighting functions and initial trajectory
        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           ref_press,
                                                           initialize_traj)

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = reweighting.init_default_loss_fn(targets)
        else:
            print('Using custom loss function. Ignoring "target" dict.')

        reweighting.checkpoint_quantities(quantities)

        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            weights, _ = weights_fn(params, traj_state)
            quantity_trajs = traj_util.quantity_traj(traj_state,
                                                     quantities,
                                                     params)
            loss, predictions = loss_fn(quantity_trajs, weights)
            return loss, predictions

        statepoint_grad_fn = jit(value_and_grad(difftre_loss, has_aux=True))

        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state)
            outputs, grad = statepoint_grad_fn(params, traj_state)
            loss_val, predictions = outputs
            return traj_state, grad, loss_val, predictions

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts
        grads, losses = [], []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]
            new_traj_state, curr_grad, loss_val, state_point_predictions = \
                grad_fn(self.params, self.trajectory_states[sim_key])

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = state_point_predictions
            grads.append(curr_grad)
            losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self._diverged = True  # ends training
                break

        self.batch_losses.append(sum(losses) / self.sim_batch_size)
        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)
        self.gradient_norm_history.append(util.tree_norm(batch_grad))

    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(f'\nEpoch {self._epoch}: Epoch loss = {epoch_loss:.5f}, Gradient '
              f'norm: {self.gradient_norm_history[-1]}, '
              f'Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        # print last scalar predictions
        for statepoint, prediction_series in self.predictions.items():
            last_predictions = prediction_series[self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    print(f'Statepoint {statepoint}: Predicted {quantity}:'
                          f' {value}')

        self._converged = self.early_stop.early_stopping(epoch_loss, thresh,
                                                         self.params)

    @property
    def best_params(self):
        return self.early_stop.best_params

    def move_to_device(self):
        super().move_to_device()
        self.early_stop.move_to_device()



# DiffTRe class for HCP Ti
class DifftreTiHCP(reweighting.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method."""
    def __init__(self, init_params, optimizer, reweight_ratio=0.9,
                 sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """Initializes a DiffTRe trainer instance.

        The implementation assumes a NVT ensemble in weight computation.
        The trainer initialization only sets the initial trainer state
        as well as checkpointing and save-functionality. For training,
        target state points with respective simulations need to be added
        via 'add_statepoint'.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes a new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the maximum loss
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average loss across the batch. For a
                                   single state point, both are equivalent.
                                   A criterion based on the rolling standatd
                                   deviation 'std' might be implemented in the
                                   future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        self.batch_losses, self.epoch_losses = [], []
        self.predictions = {}
        self.early_stop = max_likelihood.EarlyStopping(convergence_criterion)
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size,
            energy_fn_template=energy_fn_template)

    def add_statepoint(self, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, quantities,
                       reference_state, targets=None, ref_press=None,
                       loss_fn=None, initialize_traj=True, box_tensor=None, N=None,
                       target_dict=None):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. In case the default loss function should
        be employed, for each observable the 'target' dict containing
        a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target' needs to be provided.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness),
        a custom loss_fn needs to be defined. The signature of the loss_fn
        needs to be the following: It takes the trajectory of computed
        instantaneous quantities saved in a dict under its respective key of
        the quantities_dict. Additionally, it receives corresponding weights
        w_i to perform ensemble averages under the reweighting scheme. With
        these components, ensemble averages of more complex observables can
        be computed. The output of the function is (loss value, predicted
        ensemble averages). The latter is only necessary for post-processing
        the optimization process. See 'init_independent_mse_loss_fn' for
        an example implementation.

        Args:
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            quantities: Dict containing for each observables specified by the
                        key a corresponding function to compute it for each
                        snapshot using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                     another dict providing 'gamma' and 'target' for each
                     observable. Targets are only necessary when using the
                     'independent_loss_fn'.
            loss_fn: Custom loss function taking the trajectory of quantities
                     and weights and returning the loss and predictions;
                     Default None initializes an independent MSE loss, which
                     computes reweighting averages from snapshot-based
                     observables.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
        """

        # init simulation, reweighting functions and initial trajectory
        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           ref_press,
                                                           initialize_traj)

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = reweighting.init_default_loss_fn(targets)
        elif loss_fn == 'stiffness_loss':
            loss_fn = reweighting.init_stiffness_loss_function(energy_fn_template=energy_fn_template,
                                                               box_tensor=box_tensor, kbT=kbt, N=N,
                                                               target_dict=target_dict)
        elif loss_fn == 'stiffness_and_stress_loss':
            loss_fn = reweighting.init_stiffness_and_stress_loss_function(energy_fn_template=energy_fn_template,
                                                                          box_tensor=box_tensor, kbT=kbt, N=N,
                                                                          target_dict=target_dict)
        elif loss_fn == 'stiffness_and_pressure_loss':
            loss_fn = reweighting.init_stiffness_and_pressure_loss_function(energy_fn_template=energy_fn_template,
                                                                            box_tensor=box_tensor, kbT=kbt, N=N,
                                                                            target_dict=target_dict)
        else:
            print('Using custom loss function. Ignoring "target" dict.')
            exit(0)

        reweighting.checkpoint_quantities(quantities)

        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            weights, _ = weights_fn(params, traj_state)
            quantity_trajs = traj_util.quantity_traj(traj_state,
                                                     quantities,
                                                     params)
            loss, predictions = loss_fn(quantity_trajs, weights)
            return loss, predictions

        statepoint_grad_fn = jit(value_and_grad(difftre_loss, has_aux=True))

        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state)
            outputs, grad = statepoint_grad_fn(params, traj_state)
            loss_val, predictions = outputs
            return traj_state, grad, loss_val, predictions

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts
        grads, losses = [], []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]
            new_traj_state, curr_grad, loss_val, state_point_predictions = \
                grad_fn(self.params, self.trajectory_states[sim_key])

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = state_point_predictions
            grads.append(curr_grad)
            losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self._diverged = True  # ends training
                break

        self.batch_losses.append(sum(losses) / self.sim_batch_size)
        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)
        self.gradient_norm_history.append(util.tree_norm(batch_grad))

    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(f'\nEpoch {self._epoch}: Epoch loss = {epoch_loss:.5f}, Gradient '
              f'norm: {self.gradient_norm_history[-1]}, '
              f'Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        # print last scalar predictions
        # jax.debug.print("Predictions: {predictions}", predictions=self.predictions)
        for statepoint, prediction_series in self.predictions.items():
            last_predictions = prediction_series[self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    print(f'Statepoint {statepoint}: Predicted {quantity}:'
                          f' {value}')

        self._converged = self.early_stop.early_stopping(epoch_loss, thresh,
                                                         self.params)

    @property
    def best_params(self):
        return self.early_stop.best_params

    def move_to_device(self):
        super().move_to_device()
        self.early_stop.move_to_device()


class hybridTrainer():

    def __init__(self, trainer1, trainer2):
        self._trainer1 = trainer1
        self._trainer2 = trainer2

    def train(self, num_epochs):
        for i in range(num_epochs):
            self._trainer1.train(1, checkpoint_freq=None)
            # Set new params of trainer 2 based on params of trainer 1
            self._trainer2.params = self._trainer1.params

            self._trainer2.train(1, checkpoint_freq=None)
            # Set new params of trainer 1 based on params of trainer
            self._trainer1.params = self._trainer2.params



class hybridTrainer4temps_and_DFT():

    def __init__(self, trainerDiffTRe, trainerDFT):
        self._trainer0 = trainerDiffTRe
        self._trainer1 = trainerDFT


    def train(self, num_epochs, stiffness_path, pressure_path=None, stress_path=None):
        trainer_stiffness = [[], [], [], []]
        trainer_pressure = [[], [], [], []]
        trainer_stress = [[], [], [], []]
        num_epochs = int(num_epochs)
        for i in range(num_epochs):
            self._trainer0.train(1, checkpoint_freq=None)
            trainer_stiffness[0].append(onp.array(self._trainer0.predictions[0][i]['stiffness']))
            trainer_stiffness[1].append(onp.array(self._trainer0.predictions[1][i]['stiffness']))
            trainer_stiffness[2].append(onp.array(self._trainer0.predictions[2][i]['stiffness']))
            trainer_stiffness[3].append(onp.array(self._trainer0.predictions[3][i]['stiffness']))

            if 'pressure_scalar' in self._trainer0.predictions[0][i].keys():
                trainer_pressure[0].append(self._trainer0.predictions[0][i]['pressure_scalar'])
                trainer_pressure[1].append(self._trainer0.predictions[1][i]['pressure_scalar'])
                trainer_pressure[2].append(self._trainer0.predictions[2][i]['pressure_scalar'])
                trainer_pressure[3].append(self._trainer0.predictions[3][i]['pressure_scalar'])

            if 'stress' in self._trainer0.predictions[0][i].keys():
                trainer_pressure[0].append(self._trainer0.predictions[0][i]['stress'])
                trainer_pressure[1].append(self._trainer0.predictions[1][i]['stress'])
                trainer_pressure[2].append(self._trainer0.predictions[2][i]['stress'])
                trainer_pressure[3].append(self._trainer0.predictions[3][i]['stress'])

            self._trainer1.params = self._trainer0.params
            self._trainer1.train(1, checkpoint_freq=None)
            self._trainer0.params = self._trainer1.params

            if i % 10 == 0:
                temp_save = '../examples/output/figures/' + str(
                    i) + 'th_epoch_280423_stiffness_4temps_200epochs_23_323_623_923K_DFT_ec1emin10_presss1emin9_80_10ps_gf1emin2.npy'
                onp.save(temp_save, onp.array(trainer_stiffness))

                if 'pressure_scalar' in self._trainer0.predictions[0][i].keys():
                    temp_save_pressure = '../examples/output/figures/' + str(
                        i) + 'th_epoch_280423_pressure_4temps_200epochs_23_323_623_923K_DFT_ec1emin10_presss1emin9_gf1emin2.npy'
                    onp.save(temp_save_pressure, onp.array(trainer_pressure))

                if 'stress' in self._trainer0.predictions[0][i].keys():
                    temp_save_stress = '../examples/output/figures/' + str(
                        i) + 'th_epoch_280423_stress_4temps_200epochs_23_323_623_923K_DFT_ec1emin10_prress1emin9_80_10ps_gf1emin2.npy'
                    onp.save(temp_save_stress, onp.array(trainer_stress))

                temp_save_energy_params = '../examples/output/figures/' + str(
                    i) + 'th_epoch_280423_energyParams_4temps_200epochs_23_323_623_923K_DFT_ec1emin10_press1emin9_80_10ps_gf1emin2.pkl'
                self._trainer0.save_energy_params(file_path=temp_save_energy_params, save_format='.pkl')


        trainer_stiffness = onp.array(trainer_stiffness)
        onp.save(stiffness_path, trainer_stiffness)

        if 'pressure_scalar' in self._trainer0.predictions[0][0].keys():
            trainer_pressure = onp.array(trainer_pressure)
            onp.save(pressure_path, trainer_pressure)

        if 'stress' in self._trainer0.predictions[0][0].keys():
            trainer_stress = onp.array(trainer_stress)
            onp.save(stress_path, trainer_stress)







class template_trainer_DFT():

    def __init__(self, trainerDFT):
        self._trainer0 = trainerDFT


    def train(self, num_epochs):
        num_epochs = int(num_epochs)
        for i in range(num_epochs):
            self._trainer0.train(1, checkpoint_freq=None)

            if i % 10 == 0:
                temp_save_energy_params = '../examples/output/figures/' + str(
                    i) + 'th_epoch_300323_energyParams_DFT_1emin6_1emin2.pkl'
                self._trainer0.save_energy_params(file_path=temp_save_energy_params, save_format='.pkl')
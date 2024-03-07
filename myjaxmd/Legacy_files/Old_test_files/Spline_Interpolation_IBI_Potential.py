"""File to fit a tabulated spline to the IBI initial guess and then train it via gradient descent to match the final
IBI Potential. Can be used to find good hyperparameters for the related RDF lerarning problem.
We could add regularisation term to avoid oscillating forces, e.g. by punishing the difference of 2 or three neighboring
support points."""

import numpy as onp
from scipy import interpolate as sci_interpolate
import jax.numpy as np
from jax import random, jit, value_and_grad, vmap, grad, config
config.update('jax_numpy_rank_promotion','raise')  # to make sure unintentional broadcasting does not create bugs
# saver to keep this on! Weird broadcasting can cause strange behaviour
from jax.experimental import optimizers
import time
from jax_md.interpolate import InterpolatedUnivariateSpline

import torch  # use pytorch for data loading
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


from jax_md.util import *

"""Alternative: Use Potential of Mean Force as Prior and only learn difference --> possibly faster training 
                due to reduced magnitude differences in parameters
                """

class PotentialDataset(Dataset):
    """Tabulated Potential dataset."""
    def __init__(self, table_loc, r_values):
        """
        Args:
            location: Path to the espresso++ csv potential file
        """
        U_int, F_int = self.interpolate_tabulated_pot(table_loc)
        self.r_values = r_values
        self.U_values = U_int(self.r_values)
        self.F_values = F_int(self.r_values)
        self.U_std = np.std(self.U_values)
        self.F_std = np.std(self.F_values)

    def interpolate_tabulated_pot(self, location):
        """
        Assumes Espresso-style tabulated potential: r, U, F
        :param location: String describing location of tabulated potential
        :return: Scipy interpolations of U and F
        """
        pot_table = onp.loadtxt(location)
        U_int = sci_interpolate.interp1d(pot_table[:, 0], pot_table[:, 1], kind='cubic')
        F_int = sci_interpolate.interp1d(pot_table[:, 0], pot_table[:, 2], kind='cubic')
        return U_int, F_int

    def __len__(self):
        return len(self.r_values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'r': self.r_values[idx], 'U': self.U_values[idx], 'F': self.F_values[idx]}
        return sample

def predict_U_F(params, input):
    """Automatic batching of prediction function"""
    # in_axes: Dont paralelize over 1st argument, but parallelize over 0-axis of second argument
    # out_axes: similar definition to in_axes, default axis is 0
    # --> construct input as 2D array with inputs along axis 1 and batch samples along axis 0
    # define own vmap for batched prediction of forces, as grad only defined on scalars
    # 1 says to take gradient wrt. input index 1 (default 0)
    # beware! this yields output of shape (batch,) for U and (batch,1) for F!

    # need to recompute spline at each gradient update, i.e. at each batch:
    spline = InterpolatedUnivariateSpline(x_vals, params, k=degree)  # uses global parameters --> depreciated

    # interpolate returns a 1D [u_val] array, we need to reshape it to a scalar to apply gradient on it
    U_pred, F_pred = vmap(value_and_grad(spline), in_axes=0, out_axes=0)(input)
    return U_pred, -F_pred  # minus due to force being negative gradient of potential

def accuracy(params, my_dataloader):
    # returns MAE of U and F of testset
    MAE_U = 0.
    MAE_F = 0.
    for i, batched_sample in enumerate(my_dataloader):
        predicted_U, predicted_F = predict_U_F(params, batched_sample['r'].numpy())
        MAE_U += np.mean(np.abs(predicted_U - batched_sample['U'].numpy()))
        MAE_F += np.mean(np.abs(predicted_F - batched_sample['F'].numpy()))
    return MAE_U / len(my_dataloader), MAE_F / len(my_dataloader)

def loss(params, input, targets_U, targets_F, U_norm, F_norm):
    """Loss function that minimized relative squared error"""
    U, F = predict_U_F(params, input)
    loss_U_RMS = np.sqrt(np.mean((np.square(U - targets_U)))) / U_norm
    loss_F_RMS = np.sqrt(np.mean((np.square(F - targets_F)))) / F_norm
    # loss_F = np.mean(np.sqrt(np.square(directF - targets_F)) / np.abs(targets_F))
    return loss_U_RMS + loss_F_RMS

def relative_loss(params, input, targets_U, targets_F, U_norm, F_norm):
    """Loss function that minimized relative squared error"""
    U, F = predict_U_F(params, input)
    u_thresh = 0.05
    f_thresh = 0.1
    # careful when using where: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    # https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf
    # Jax MD has this save_where implemented! Use where necessary!
    U_target_sq = np.square(targets_U)
    thresholded_U_target = np.where(U_target_sq > u_thresh ** 2, np.where(U_target_sq > u_thresh ** 2, targets_U, u_thresh), u_thresh)  # TODO this does not work! generates NaN in gradient
    thresholded_F_target = np.where(np.square(targets_F) > f_thresh ** 2, targets_F, f_thresh)
    squared_U_err = np.square(U - targets_U)
    squared_F_err = np.square(F - targets_F)
    loss_U_rel = np.sqrt(np.mean(squared_U_err / thresholded_U_target))
    loss_F_rel = np.sqrt(np.mean(squared_F_err / thresholded_F_target))
    return loss_U_rel # + loss_F_rel

def update(step, params, x, targets_U, targets_F, opt_state, U_norm, F_norm):
    """ Compute the gradient, update the parameters and the opt_state and return loss for a batch"""
    # value, grads = value_and_grad(loss)(params, prior, x, targets_U, targets_F, U_norm, F_norm)
    value, grads = value_and_grad(loss)(params, x, targets_U, targets_F, U_norm, F_norm)
    # value, grads = value_and_grad(relative_loss)(params, x, targets_U, targets_F, U_norm, F_norm)
    opt_state = opt_update(step, grads, opt_state)  # update optimizer state after stepping
    return get_params(opt_state), opt_state, value


if __name__ == '__main__':
    # user input:

    table_loc = 'data/CG_potential_SPC_955.csv'
    IBI_prior_table = 'data/IBI_Initial_guess.csv'
    onset = 0.25  # training points are sampled starting from here
    cut = 1.
    delta_cut = 0.1  # define spline on a larger area to avoid oscillations at boundary
    resolution_training_data = 200
    resolution_spline = 100
    dx = cut / float(resolution_spline)

    x_vals = np.linspace(0., cut + delta_cut, resolution_spline)
    IBI_prior_array = onp.loadtxt(IBI_prior_table)

    # compute tabulated values at spline support points
    U_init_int = sci_interpolate.interp1d(IBI_prior_array[:, 0], IBI_prior_array[:, 1], kind='cubic')
    u_vals_init = U_init_int(x_vals)

    params = u_vals_init  # initial params defining spline
    degree = 3

    # fit spline to IBI initial guess for testing
    spline = InterpolatedUnivariateSpline(x_vals, params, k=degree)
    plt.plot(x_vals, spline(x_vals), label='Spline')
    plt.plot(IBI_prior_array[:, 0], IBI_prior_array[:, 1], linestyle='--', label='Original data')
    plt.ylim([-2.5, 2.5])
    plt.legend()
    plt.savefig('Figures/Spline_Initialization.png')

    # training params:
    num_epochs = 100
    step_size = 0.1
    batch = 500
    test_ratio = 0.1

    # Set up Data
    r_values = onp.linspace(onset, cut, resolution_training_data)
    dataset = PotentialDataset(table_loc, r_values)  # initialize dataset
    # train-test-split
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # set up automatic batching loader
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

    # initialize optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size)  # define adam
    cur_opt_state = opt_init(params)  # initialize adam

    # test_array = np.array([0.5, 0.5, 0.5]).reshape([-1,1])
    # U = batched_predict(params, test_array)

    # train the NN
    step = 0
    train_history = onp.zeros(num_epochs)
    test_history = onp.zeros(num_epochs)
    # prior function needs to be assigned static, otherwise cannot jit
    # only arrays and containers of arrays can be jitted non-statically
    update = jit(update)
    for epoch in range(num_epochs):
        start_time = time.time()
        for batched_sample in trainloader:
            params, cur_opt_state, loss_val = update(step, params, batched_sample['r'].numpy(),
                                                     batched_sample['U'].numpy(),
                                                     batched_sample['F'].numpy(), cur_opt_state, dataset.U_std, dataset.F_std)
            # print(params[100])
            step += 1  # current step of Adam (not sure if really necessary)
        epoch_time = time.time() - start_time

        # train_acc = accuracy(params, trainloader)
        test_acc = accuracy(params, testloader)
        # train_history[epoch] = onp.asarray(train_acc)[0]
        test_history[epoch] = onp.asarray(test_acc)[0]
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print('Loss = ', loss_val)
        # print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

    # plotting
    U_pred, F_pred = predict_U_F(params, r_values)

    plt.figure()
    plt.plot(train_history, label='Train_acc')
    plt.plot(test_history, label='Test_acc')
    plt.legend()
    plt.savefig('Figures/Train_history.png')

    plt.figure()
    plt.plot(r_values, U_pred, label='Spline')
    plt.plot(r_values, dataset.U_values, label='Tabulated', linestyle='--')
    plt.plot(x_vals, u_vals_init, label='Initial guess', linestyle='--')
    # plt.scatter(x_vals, params, label='Spline support')
    plt.ylim([-2.5, 2.5])
    plt.legend()
    plt.savefig('Figures/NN_potential.png')

    plt.figure()
    plt.plot(r_values, U_pred, label='NN')
    plt.plot(r_values, dataset.U_values, label='Tabulated')
    plt.ylim([-2, 2])
    plt.legend()
    plt.savefig('Figures/NN_potential_zoom.png')

    plt.figure()
    plt.plot(r_values, F_pred, label='NN Forces')
    plt.plot(r_values, dataset.F_values, label='Tabulated Forces')
    plt.legend()
    plt.savefig('Figures/NN_forces.png')

    plt.figure()
    plt.plot(r_values, F_pred, label='NN Forces')
    plt.plot(r_values, dataset.F_values, label='Tabulated Forces')
    plt.legend()
    plt.ylim([-60, 60])
    plt.savefig('Figures/NN_forces_zoom.png')

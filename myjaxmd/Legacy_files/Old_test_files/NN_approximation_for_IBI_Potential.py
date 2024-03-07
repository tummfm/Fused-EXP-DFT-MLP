"""Interpolation of tabulated potential via a FC-NN"""

import numpy as onp
from scipy import interpolate as sci_interpolate
import jax.numpy as np
from jax import random, jit, value_and_grad, vmap, grad, config
config.update('jax_numpy_rank_promotion','warn')  # to make sure unintentional broadcasting does not create bugs
# saver to keep this on! Weird broadcasting can cause strange behaviour
from jax.experimental import optimizers
from jax.nn import relu, selu, celu
import time
from jax_md import interpolate
from functools import partial

import torch  # use pytorch for data loading
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

def random_layer_params(m, n, key, scale=1e-2):
    """
    A helper function to randomly initialize weights and biases
    Weights connect layers, i.e. 3 Layer Network has calls this function twice
    :param m: incoming layer size
    :param n: outcoming layer size
    :param key: jax random key
    :param scale: Gaussian standard deviation
    :return: Gaussian initialized weights and biases with dimensions defined by m and n
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_FC_network_params(sizes, key):
    """Initialize all layers for a fully-connected neural network with sizes "sizes"""
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def predict(params, r):
    """
    prediction function of fully connected network for single input r
    Only for single input, batching is handled by vmap
    """
    activations = r
    #w_input, b_input = params[0]
    #outputs = np.dot(w, activations) + b
    for w, b in params[0:-1]:  # iterate hidden layers
        outputs = np.dot(w, activations) + b
        # outputs = w * activations + b
        # activations = np.tanh(outputs)
        # activations = relu(outputs)
        # activations = selu(outputs)
        activations = celu(outputs)

    final_w, final_b = params[-1]
    final_value = np.dot(final_w, activations) + final_b
    # not run through activation function in output node
    return final_value[0]  # transform (1,) d array into 0-d array as grad can only handle these types

def IBI_prior(u_spline):
    # bakes in u_spline values
    # Alternative:
    # def foo(a):
    #     x = do_stuff(a)  # x is array with shape (1,)
    #     return np.reshape(x, ())
    u_spline_scalar = lambda r: np.reshape(u_spline(r), ())  # interpolate returns a 1D [u_val] array, we need to reshape it to a scalar to apply gradient on it
    def prior_fun(x):
        return value_and_grad(u_spline_scalar)(x)
    return prior_fun

def batched_predict_U_F(params, input, prior=None):
    """Automatic batching of prediction function"""
    # in_axes: Dont paralelize over 1st argument, but parallelize over 0-axis of second argument
    # out_axes: similar definition to in_axes, default axis is 0
    # --> construct input as 2D array with inputs along axis 1 and batch samples along axis 0
    # define own vmap for batched prediction of forces, as grad only defined on scalars
    # 1 says to take gradient wrt. input index 1 (default 0)
    # beware! this yields output of shape (batch,) for U and (batch,1) for F!
    if prior is None:
        U_pred, F_pred = vmap(value_and_grad(predict, 1), in_axes=(None, 0), out_axes=0)(params, input)
        return U_pred, -F_pred  # minus due to force being negative gradient of potential
    else:
        U_pred, F_pred = vmap(value_and_grad(predict, 1), in_axes=(None, 0), out_axes=0)(params, input)
        U_prior, F_prior = vmap(prior, in_axes=0, out_axes=0)(input)
        return U_pred + U_prior, -F_prior - F_pred

def accuracy(params, my_dataloader, prior):
    # returns MAE of U and F of testset
    MAE_U = 0.
    MAE_F = 0.
    for i, batched_sample in enumerate(my_dataloader):
        predicted_U, predicted_F = batched_predict_U_F(params, batched_sample['r'].numpy().reshape([-1,1]), prior=prior)
        MAE_U += np.mean(np.abs(predicted_U - batched_sample['U'].numpy()))
        MAE_F += np.mean(np.abs(predicted_F - batched_sample['F'].numpy().reshape([-1,1])))
    return MAE_U / len(my_dataloader), MAE_F / len(my_dataloader)

def loss(params, prior, input, targets_U, targets_F, U_norm, F_norm):
    """Loss function that minimized relative squared error"""
    U, F = batched_predict_U_F(params, input, prior=prior)
    loss_U_RMS = np.sqrt(np.mean((np.square(U - targets_U)))) / U_norm
    loss_F_RMS = np.sqrt(np.mean((np.square(F - targets_F.reshape([-1,1]))))) / F_norm
    # loss_F = np.mean(np.sqrt(np.square(directF - targets_F)) / np.abs(targets_F))
    return loss_U_RMS + loss_F_RMS

def relative_loss(params, prior, input, targets_U, targets_F, U_norm, F_norm):
    """Loss function that minimized relative squared error"""
    U, F = batched_predict_U_F(params, input, prior=prior)
    u_thresh = 0.05
    f_thresh = 0.1
    # careful when using where: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    # https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf
    U_target_sq = np.square(targets_U)
    thresholded_U_target = np.where(U_target_sq > u_thresh ** 2, np.where(U_target_sq > u_thresh ** 2, targets_U, u_thresh), u_thresh)  # TODO this does not work! generates NaN in gradient
    thresholded_F_target = np.where(np.square(targets_F) > f_thresh ** 2, targets_F, f_thresh).reshape([-1, 1])
    squared_U_err = np.square(U - targets_U)
    squared_F_err = np.square(F - targets_F.reshape([-1,1]))
    loss_U_rel = np.sqrt(np.mean(squared_U_err / thresholded_U_target))
    # loss_F_rel = np.sqrt(np.mean(squared_F_err / thresholded_F_target))
    return loss_U_rel # + loss_F_rel

def update(step, params, prior, x, targets_U, targets_F, opt_state, U_norm, F_norm):
    """ Compute the gradient, update the parameters and the opt_state and return loss for a batch"""
    # value, grads = value_and_grad(loss)(params, prior, x, targets_U, targets_F, U_norm, F_norm)
    value, grads = value_and_grad(loss)(params, prior, x, targets_U, targets_F, U_norm, F_norm)
    opt_state = opt_update(step, grads, opt_state)  # update optimizer state after stepping
    # get_param is globally defined function --> OK since we do not change this function, only use it
    return get_params(opt_state), opt_state, value


if __name__ == '__main__':
    # TODO some problems with prior!! Has oscillatory forces that the NN has a hard time to correct! Fix this!
    #  Maybe compute Force directly, not via gradient of potential --> seems to be problematic at boundary of potential
    # user input:
    table_loc = 'data/CG_potential_SPC_955.csv'
    IBI_prior_table = 'data/IBI_Initial_guess.csv'
    onset = 0.25
    cut = 1.
    resolution = 1000
    IBI_prior_array = onp.loadtxt(IBI_prior_table)
    dx_IBI = (IBI_prior_array[-1, 0] - IBI_prior_array[0, 0]) / (IBI_prior_array.shape[0] - 1)
    u_spline = interpolate.spline(IBI_prior_array[:, 1], dx_IBI, degree=3)

    prior = IBI_prior(u_spline)
    #prior = None
    # NN params:
    # layer_sizes = [784, 512, 512, 10]
    layer_sizes = [1, 100, 50, 10, 1]  # size of input, output and hidden layers in between


    # training params:
    num_epochs = 300
    step_size = 0.001
    batch = 32
    test_ratio = 0.01

    # Set up Data
    r_values = onp.linspace(onset, cut, resolution)
    dataset = PotentialDataset(table_loc, r_values)  # initialize dataset
    # train-test-split
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # set up automatic batching loader
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

    # initialize optimizer
    params = init_FC_network_params(layer_sizes, random.PRNGKey(0))  # initialize NN
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
    update = jit(update, static_argnums=2)
    for epoch in range(num_epochs):
        start_time = time.time()
        for batched_sample in trainloader:
            params, cur_opt_state, loss_val = update(step, params, prior, batched_sample['r'].numpy().reshape([-1,1]),
                                                     batched_sample['U'].numpy(),
                                                     batched_sample['F'].numpy(), cur_opt_state, dataset.U_std, dataset.F_std)
            step += 1  # current step of Adam (not sure if really necessary)
        epoch_time = time.time() - start_time

        # train_acc = accuracy(params, trainloader)
        test_acc = accuracy(params, testloader, prior)
        # train_history[epoch] = onp.asarray(train_acc)[0]
        test_history[epoch] = onp.asarray(test_acc)[0]
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print('Loss = ', loss_val)
        # print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

    U_pred, F_pred = batched_predict_U_F(params, r_values.reshape(-1, 1), prior=prior)

    plt.figure()
    plt.plot(train_history, label='Train_acc')
    plt.plot(test_history, label='Test_acc')
    plt.legend()
    plt.savefig('Train_history.png')

    plt.figure()
    plt.plot(r_values, U_pred, label='NN')
    plt.plot(r_values, dataset.U_values, label='Tabulated')
    plt.legend()
    plt.savefig('NN_potential.png')

    plt.figure()
    plt.plot(r_values, U_pred, label='NN')
    plt.plot(r_values, dataset.U_values, label='Tabulated')
    plt.ylim([-2,2])
    plt.legend()
    plt.savefig('NN_potential_zoom.png')

    plt.figure()
    plt.plot(r_values, F_pred, label='NN Forces')
    plt.plot(r_values, dataset.F_values, label='Tabulated Forces')
    plt.legend()
    plt.savefig('NN_forces.png')

    plt.figure()
    plt.plot(r_values, F_pred, label='NN Forces')
    plt.plot(r_values, dataset.F_values, label='Tabulated Forces')
    plt.legend()
    plt.ylim([-60,60])
    plt.savefig('NN_forces_zoom.png')

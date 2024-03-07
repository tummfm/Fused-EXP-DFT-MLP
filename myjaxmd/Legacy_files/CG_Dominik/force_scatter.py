import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as onp

import jax
import jax.numpy as np

import haiku as hk

import time

import pickle

from jax import lax
from jax import random
from jax.api import jit, grad
# config.update("jax_enable_x64", True) # very slow!

from jax_md import space, partition

from chemtrain.jax_md_mod import custom_energy
from chemtrain.jax_md_mod import custom_nn

from functools import partial

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")

# ------------------------------ LOAD DATA ----------------------------------- #

R = np.array(onp.load("samples/conf_SPC_test.npy"))
F = np.array(onp.load("samples/forces_SPC_test.npy"))

R_init = R[0]

folder = "C" # folder in which plots and data are saved
method = "relative_entropy"
print(f"Method: {method}    Version: {folder}") 

#--------------------------- NEIGHBOR_FN --------------------------------------#

box_size = np.array([3.0, 3.0, 3.0])
r_cutoff = 0.5

displacement_fn, shift_fn = space.periodic(box_size)

neighbor_fn = partition.neighbor_list(displacement_fn, 
                                      box_size, 
                                      r_cutoff, 
                                      dr_threshold=0.05,
                                      capacity_multiplier=2.0)

neighbors_init = neighbor_fn(R_init, extra_capacity=0)

#--------------------------- ENERGY_FN ----------------------------------------#

mlp_init = {
    "w_init": custom_nn.OrthogonalVarianceScalingInit(scale=1.0),
    "b_init": hk.initializers.Constant(0.0)}

if method == "RDF_matching":
    prior_kwargs = {
        "sigma": 0.3165,
        "epsilon": 1.0,
        "exp": 12}
else:
    prior_kwargs = {
    "sigma": 0.3,
    "epsilon": 1.0,
    "exp": 12}

init_fn, energy_fn_GNN = custom_energy.dimenetpp_neighborlist(
    displacement_fn,
    R_init,
    neighbors_init,
    r_cutoff,
    embed_size=32,
    n_interaction_blocks=4,
    num_residual_before_skip=1,
    num_residual_after_skip=2,
    out_embed_size=None,
    type_embed_size=None,
    angle_int_embed_size=None,
    basis_int_embed_size=8,
    num_dense_out=3,
    num_rbf=6,
    num_sbf=7,
    activation=jax.nn.swish,
    envelope_p=6,
    init_kwargs=mlp_init,
    prior_potential=custom_energy.generic_repulsion,
    prior_kwargs=prior_kwargs,
    onset_prior_ratio=0.9,
    max_angle_multiplier=1.25,
    max_edge_multiplier=1.25)

key = random.PRNGKey(0)

with open(f"results/{method}/{folder}/data/params.pkl", "rb") as input:
        params = pickle.load(input)

@jit
def energy_fn(params, R, **kwargs):
    neighbors = neighbor_fn(R, neighbors_init)
    return energy_fn_GNN(params, R, neighbors)

#--------------------------- LOSS_FN ------------------------------------------#

grad_fn = grad(energy_fn, argnums=1)

force_fn = lambda params, R: -grad_fn(params, R)

@jit
def force_loss(params, R, F):
    # use of lax.map instead of vmap to avoid out of memory
    start_time = time.time()
    dforces = lax.map(partial(force_fn, params), R) - F
    return np.mean(np.sum(dforces ** 2, axis=(1,2)))

computed_forces = lax.map(partial(force_fn, params), R)

onp.save(f"results/{method}/{folder}/data/F_test.npy", computed_forces)
print("Finished")

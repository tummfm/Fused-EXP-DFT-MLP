import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as onp

import jax
import jax.numpy as np

import matplotlib.pyplot as plt

import haiku as hk

import pickle

from functools import partial

from jax.api import jit, vmap, grad

from jax_md import space, partition

from chemtrain.jax_md_mod import custom_energy
from chemtrain.jax_md_mod import custom_nn

import warnings
warnings.filterwarnings("ignore")
print("IMPORTANT: You have Warning Messages disabled!")

#--------------------------- GENERATE POSITIONS -------------------------------#
max_distance = 2.0
# compute energy/force every step
steps = 999
# increase distance between atoms by step_size every step
step_size = max_distance / steps

# empty container for all position vector pairs
R = []
for i in range(steps):
    atom_1 = np.array([1, 1, 1])
    atom_2 = np.array([1 + (i+1) * step_size , 1, 1])

    R.append([atom_1, atom_2])

R = np.array(R)
R_init = R[0]

method = "RDF_matching"
folder = "B"
print(f"Method: {method}    Version: {folder}")

#--------------------------- LOAD ENERGY FN -----------------------------------#
r_cutoff = 0.5
box_size = np.array(max_distance + 2)

displacement_fn, shift_fn = space.periodic(box_size)

neighbor_fn = partition.neighbor_list(displacement_fn,
                                      box_size,
                                      r_cutoff,
                                      dr_threshold=0.05,
                                      capacity_multiplier=2.0)
neighbors_init = neighbor_fn(R_init)

mlp_init = {
    "w_init": custom_nn.OrthogonalVarianceScalingInit(scale=1.0),
    "b_init": hk.initializers.Constant(0.0)}
if method == "RDF_matching":
    prior_kwargs = {
        "sigma": 0.3165, 
        "epsilon": 1.0, # TODO
        "exp": 12} 
else:
    prior_kwargs = {
        "sigma": 0.3,  #0.3165 for reweighting 
        "epsilon": 1.0, # TODO
        "exp": 12} 


init_fn, gnn_energy_fn = custom_energy.dimenetpp_neighborlist(displacement_fn,
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
                                                              num_dense_out=3,  # TODO
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

@jit
def energy_fn(params, R, **kwargs):
    neighbors = neighbor_fn(R, neighbor_list=neighbors_init)
    return gnn_energy_fn(params, R, neighbor=neighbors)

with open(f"results/{method}/{folder}/data/params.pkl", "rb") as input:
    params = pickle.load(input)

#--------------------------- COMPUTE ------------------------------------------#
# energy_fn = partial(energy_fn, params)
# force_fn = grad(energy_fn)

grad_fn = grad(energy_fn, argnums=1)
grad_fn = partial(grad_fn, params)

energy_fn = partial(energy_fn, params)
force_fn = lambda R: -grad_fn(R)

energy_fn = vmap(energy_fn)
force_fn = vmap(force_fn)

potential = energy_fn(R)
forces = force_fn(R)
forces = forces[:, 1, 0] # there are only forces in x-direction

#--------------------------- SAVE DATA ----------------------------------------#
onp.savetxt(f"results/{method}/{folder}/data/pair_energy.csv", potential)
onp.savetxt(f"results/{method}/{folder}/data/pair_force.csv", forces)
#--------------------------- LOAD REFERENCE -----------------------------------#
reference = onp.loadtxt("data/CG_potential_SPC_955.csv")
reference_potential = reference[:, 1]
reference_forces = reference[:, 2]

#--------------------------- CREATE PLOTS -------------------------------------#
plt.figure(facecolor="w")
x_vals = np.linspace(0.0, max_distance, num=steps)
plt.plot(x_vals, potential, label="learned")
plt.plot(x_vals, reference_potential, label="reference")
plt.legend()
plt.xlabel("Distance r [nm]")
plt.ylim([-5, 5])
plt.ylabel("Energy E [kJ $\mathrm{mol^{-1}}$]")
# plt.savefig(f"{path}/plots/pair_energy.svg")
plt.show()

plt.figure(facecolor="w")
x_vals = np.linspace(0.0, max_distance, num=steps)
plt.plot(x_vals, forces, label="learned")
plt.plot(x_vals, reference_forces, label="reference")
plt.legend()
plt.xlabel("Distance r [nm]")
plt.ylim([-75, 75])
plt.ylabel("Force F [kJ $\mathrm{mol^{-1} \ nm^{-1}}$]")
# plt.savefig(f"{path}/plots/pair_force.svg")
plt.show()
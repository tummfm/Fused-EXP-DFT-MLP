import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax.numpy as np

import numpy as onp

from jax_md import space, partition, util, dataclasses

from chemtrain.jax_md_mod import custom_quantity

#--------------------------- LOAD DATA ----------------------------------------#
position_data = np.array(onp.load("samples/conf_SPC_1k.npy"))

#--------------------------- SETUP NEIGHBOR_FN --------------------------------#
box_size = np.array([3.0, 3.0, 3.0])
r_cutoff = 0.9 # TODO what cutoff when no energy_fn?
               # TODO high cutoff leads to oom?

displacement_fn, shift_fn = space.periodic(box_size)

neighbor_fn = partition.neighbor_list(displacement_fn,
                                      box_size,
                                      r_cutoff,
                                      dr_threshold=0.2,
                                      capacity_multiplier=1.25)
                                      # TODO what parameter when no energy_fn?

#--------------------------- RDF PARAMS ---------------------------------------#
rdf_discretization_params = custom_quantity.rdf_discretization(rdf_cut=1.0,
                                                               nbins=300,
                                                               rdf_start=0.0)
rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization_params
rdf_params = custom_quantity.RDFParams(reference_rdf=np.array([]),
                                       rdf_bin_centers=rdf_bin_centers,
                                       rdf_bin_boundaries=rdf_bin_boundaries,
                                       sigma_RDF=sigma_RDF)

#--------------------------- ADF PARAMS ---------------------------------------#
adf_discretization_params = custom_quantity.adf_discretization(nbins_theta=150)
adf_bin_centers, sigma_ADF = adf_discretization_params
adf_params = custom_quantity.ADFParams(reference_adf=np.array([]),
                                       adf_bin_centers=adf_bin_centers,
                                       sigma_ADF=sigma_ADF,
                                       r_outer=0.318,
                                       r_inner=0.0)

#--------------------------- INITIALIZE ---------------------------------------#
# define pseudo state for RDF and ADF functions
Array = util.Array
@dataclasses.dataclass
class PseudoState:
    position: Array

R_init = position_data[0]
neighbors_init = neighbor_fn(R_init)

rdf_fn = custom_quantity.init_rdf(displacement_fn, rdf_params, box_size)
adf_fn = custom_quantity.init_adf_nbrs(displacement_fn,
                                       adf_params,
                                       smoothing_dr=0.01,
                                       r_init=R_init,
                                       nbrs_init=neighbors_init)
# empty list to store adf/rdfs
rdfs = []
adfs = []

#--------------------------- COMPUTE RDF/ADF ----------------------------------#
# must use for-loop: map, scan etc. only work on proper pytrees
for i, R in enumerate(position_data):

    print(f"{i+1}. Iteration")
    
    # for some reason R is a numpy ndarray but needs to be jax deviceArray
    # jax arrays probably dont properly support normal "for ... in ..." loops
    R = np.array(R)

    state = PseudoState(R)
    neighbors = neighbor_fn(R)

    rdf = rdf_fn(state)
    adf = adf_fn(state, neighbors)

    rdfs.append(rdf)
    adfs.append(adf)

rdf = np.mean(np.array(rdfs), axis=0)
adf = np.mean(np.array(adfs), axis=0)

onp.savetxt("data/SPC_FW_RDF.csv", rdf)
onp.savetxt("data/SPC_FW_ADF.csv", adf)



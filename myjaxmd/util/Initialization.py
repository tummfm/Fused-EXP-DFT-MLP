from functools import partial
import chex

import haiku as hk
import jax.numpy as np
from jax import random, checkpoint
from scipy import interpolate as sci_interpolate
import pickle

from chemtrain import traj_quantity, layers, neural_networks
from chemtrain.jax_md_mod import custom_energy, custom_space, custom_quantity
import numpy as onp
from jax_md import util, simulate, partition, space

Array = util.Array


@chex.dataclass
class InitializationClass:
    """A dataclass containing initialization information.

    Notes:
      careful: dataclasses.astuple(InitializationClass) sometimes
      changes type from jnp.Array to onp.ndarray

    Attributes:
        R_init: Initial Positions
        box: Simulation box size
        kbT: Target thermostat temperature times Boltzmann constant
        mass: Particle masses
        dt: Time step size
        species: Species index for each particle
        ref_press: Target pressure for barostat
        temperature: Thermostat temperature; only used for computation of
                     thermal expansion coefficient and heat capacity
    """
    R_init: Array
    box: Array
    kbT: float
    masses: Array
    dt: float
    species: Array = None
    ref_press: float = 1.
    temperature: float = None


def select_target_RDF(target_rdf, RDF_start=0., nbins=300):
    if target_rdf == 'LJ':
        reference_rdf = util.f32(onp.loadtxt('data/LJ_reference_RDF.csv'))
        RDF_cut = 1.5
    elif target_rdf == 'SPC':
        reference_rdf = onp.loadtxt('data/water_models/SPC_955_RDF.csv')
        RDF_cut = 1.0
    elif target_rdf == 'SPC_FW':
        reference_rdf = onp.loadtxt('data/water_models/SPC_FW_RDF.csv')
        RDF_cut = 1.0
        raise NotImplementedError
    elif target_rdf == 'Water_Ox':
        reference_rdf = onp.loadtxt('data/experimental/O_O_RDF.csv')
        RDF_cut = 1.0
    else:
        raise ValueError('The reference rdf ' + target_rdf + ' is not implemented.')

    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = custom_quantity.rdf_discretization(RDF_cut, nbins, RDF_start)
    rdf_spline = sci_interpolate.interp1d(reference_rdf[:, 0], reference_rdf[:, 1], kind='cubic')
    reference_rdf = util.f32(rdf_spline(rdf_bin_centers))
    rdf_struct = custom_quantity.RDFParams(reference_rdf, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF)
    return rdf_struct


def select_target_ADF(target_adf, r_outer, r_inner=0., nbins_theta=150):
    if target_adf == 'Water_Ox':
        reference_adf = onp.loadtxt('data/experimental/O_O_O_ADF.csv')
    else:
        raise ValueError('The reference adf ' + target_adf + ' is not implemented.')

    adf_bin_centers, sigma_ADF = custom_quantity.adf_discretization(nbins_theta)

    adf_spline = sci_interpolate.interp1d(reference_adf[:, 0], reference_adf[:, 1], kind='cubic')
    reference_adf = util.f32(adf_spline(adf_bin_centers))

    adf_struct = custom_quantity.ADFParams(reference_adf, adf_bin_centers, sigma_ADF, r_outer, r_inner)
    return adf_struct


def build_quantity_dict(R_init, box_tensor, displacement, energy_fn_template,
                        nbrs, target_dict, init_class):
    targets = {}
    compute_fns = {}
    kJ_mol_nm3_to_bar = 16.6054

    if 'kappa' in target_dict or 'alpha' in target_dict or 'cp' in target_dict:
        compute_fns['volume'] = custom_quantity.volume_npt
        if 'alpha' in target_dict or 'cp' in target_dict:
            compute_fns['energy'] = custom_quantity.energy_wrapper(
                energy_fn_template)

    if 'rdf' in target_dict:
        rdf_struct = target_dict['rdf']
        rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)
        rdf_dict = {'target': rdf_struct.reference, 'gamma': 1.,
                    'traj_fn': traj_quantity.init_traj_mean_fn('rdf')}
        targets['rdf'] = rdf_dict
        compute_fns['rdf'] = rdf_fn
    if 'adf' in target_dict:
        adf_struct = target_dict['adf']
        adf_fn = custom_quantity.init_adf_nbrs(
            displacement, adf_struct, smoothing_dr=0.01, r_init=R_init,
            nbrs_init=nbrs)
        adf_target_dict = {'target': adf_struct.reference, 'gamma': 1.,
                           'traj_fn': traj_quantity.init_traj_mean_fn('adf')}
        targets['adf'] = adf_target_dict
        compute_fns['adf'] = adf_fn
    if 'pressure' in target_dict:
        pressure_fn = custom_quantity.init_pressure(energy_fn_template,
                                                    box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure'], 'gamma': 1.e-7,
            'traj_fn': traj_quantity.init_traj_mean_fn('pressure')}  # 5.e-6 good alternative
        targets['pressure'] = pressure_target_dict
        compute_fns['pressure'] = pressure_fn
    if 'pressure_scalar' in target_dict:
        pressure_fn = custom_quantity.init_pressure(energy_fn_template,
                                                    box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure_scalar'], 'gamma': 5.e-6,
            'traj_fn': traj_quantity.init_traj_mean_fn('pressure_scalar')}
        targets['pressure_scalar'] = pressure_target_dict
        compute_fns['pressure_scalar'] = pressure_fn
    if 'density' in target_dict:
        density_dict = {
            'target': target_dict['density'], 'gamma': 1.e-3,  # 1.e-5
            'traj_fn': traj_quantity.init_traj_mean_fn('density')
        }
        targets['density'] = density_dict
        compute_fns['density'] = custom_quantity.density
    if 'kappa' in target_dict:
        def compress_traj_fn(quantity_trajs):
            volume_traj = quantity_trajs['volume']
            kappa = traj_quantity.isothermal_compressibility_npt(volume_traj,
                                                                 init_class.kbT)
            return kappa

        comp_dict = {
            'target': target_dict['kappa'],
            'gamma': 1. / (5.e-5 * kJ_mol_nm3_to_bar),
            'traj_fn': compress_traj_fn
        }
        targets['kappa'] = comp_dict
    if 'alpha' in target_dict:
        def thermo_expansion_traj_fn(quantity_trajs):
            alpha = traj_quantity.thermal_expansion_coefficient_npt(
                quantity_trajs['volume'], quantity_trajs['energy'],
                init_class.temperature, init_class.kbT, init_class.ref_press)
            return alpha

        alpha_dict = {
            'target': target_dict['alpha'], 'gamma': 1.e4,
            'traj_fn': thermo_expansion_traj_fn
        }
        targets['alpha'] = alpha_dict

    if 'cp' in target_dict:
        n_particles, dim = R_init.shape
        # assuming no reduction, e.g. due to rigid bonds
        n_dof = dim * n_particles

        def cp_traj_fn(quantity_trajs):
            cp = traj_quantity.specific_heat_capacity_npt(
                quantity_trajs['volume'], quantity_trajs['energy'],
                init_class.temperature, init_class.kbT, init_class.ref_press,
                n_dof)
            return cp

        cp_dict = {
            'target': target_dict['cp'], 'gamma': 10.,
            'traj_fn': cp_traj_fn
        }
        targets['cp'] = cp_dict

    return compute_fns, targets


def build_quantity_dict_ti_hcp(R_init, box_tensor, energy_fn_template, target_dict, init_class):
    targets = {}
    compute_fns = {}

    N = R_init.shape[0]
    kbT = init_class.kbT

    # Initialize observable function
    stress_fn = custom_quantity.init_virial_stress_tensor(energy_fn_template,
                                                          box_tensor)
    born_stiffness_fn, sigma_born, sigma_tensor_prod, stiffness_tensor_fn = \
        custom_quantity.init_stiffness_tensor_stress_fluctuation(energy_fn_template,
                                                                 box_tensor, kbT, N)

    if 'stiffness' in target_dict:
        compute_fns['born_stress'] = checkpoint(sigma_born)
        compute_fns['born_stiffness'] = checkpoint(born_stiffness_fn)
        compute_fns['stress'] = checkpoint(stress_fn)

    pressure_scalar_fn = custom_quantity.init_pressure(energy_fn_template,
                                                       box_tensor)

    if 'pressure_scalar' in target_dict:
        compute_fns['pressure_scalar'] = checkpoint(pressure_scalar_fn)

    return compute_fns, targets


def default_x_vals(r_cut, delta_cut):
    return np.linspace(0.05, r_cut + delta_cut, 100, dtype=np.float32)


def select_model(model, R_init, displacement, box, model_init_key, species=None,
                 x_vals=None, fractional=True, kbt_dependent=False, precom_edge_mask_init=None,
                 precom_max_edges=None, precom_max_triplets=None, **energy_kwargs):

    """Pass precomputed max_edges and max_triplets to decrease memory requirement and improve performance."""

    if model == 'LJ':
        r_cut = 0.9
        init_params = np.array([0.2, 1.2], dtype=np.float32)  # initial guess
        lj_neighbor_energy = partial(
            custom_energy.customn_lennard_jones_neighbor_list, displacement, box,
            r_onset=0.8, r_cutoff=r_cut, dr_threshold=0.2,
            capacity_multiplier=1.25, fractional=fractional)
        neighbor_fn, _  = lj_neighbor_energy(sigma=init_params[0], epsilon=init_params[1])
        nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN

        def energy_fn_template(energy_params):
            # we only need to re-create energy_fn, neighbor function can be re-used
            energy = lj_neighbor_energy(sigma=energy_params[0], epsilon=energy_params[1],
                                                   initialize_neighbor_list=False)
            return energy


    elif model == 'Tabulated':
        # TODO: change initial guess to generic LJ or random initialization
        r_cut = 0.9
        delta_cut = 0.1
        if x_vals is None:
            x_vals = default_x_vals(r_cut, delta_cut)

        # load PMF initial guess
        # pmf_init = False  # for IBI
        pmf_init = False
        if pmf_init:
            # table_loc = 'data/tabulated_potentials/CG_potential_SPC_955.csv'
            table_loc = 'data/tabulated_potentials/IBI_initial_guess.csv'
            tabulated_array = onp.loadtxt(table_loc)
            # compute tabulated values at spline support points
            U_init_int = sci_interpolate.interp1d(tabulated_array[:, 0], tabulated_array[:, 1], kind='cubic')
            init_params = np.array(U_init_int(x_vals), dtype=np.float32)
        else:
            # random initialisation + prior
            init_params = 0.1 * random.normal(model_init_key, x_vals.shape)
            init_params = np.array(init_params, dtype=np.float32)
            prior_fn = custom_energy.generic_repulsion_neighborlist(
                displacement, sigma=0.3165, epsilon=1., exp=12,
                initialize_neighbor_list=False, r_onset=0.9 * r_cut,
                r_cutoff=r_cut
            )


        tabulated_energy = partial(custom_energy.tabulated_neighbor_list, displacement, x_vals,
                                   box_size=box, r_onset=(r_cut - 0.2), r_cutoff=r_cut,
                                   dr_threshold=0.2, capacity_multiplier=1.25)
        neighbor_fn, _ = tabulated_energy(init_params)

        nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN

        if pmf_init:
            def energy_fn_template(energy_params):
                energy = tabulated_energy(energy_params, initialize_neighbor_list=False)
                return energy
        else:  # with prior
            def energy_fn_template(energy_params):
                tab_energy = tabulated_energy(energy_params, initialize_neighbor_list=False)
                def energy(R, neighbor, **dynamic_kwargs):
                    return tab_energy(R, neighbor, **dynamic_kwargs) + prior_fn(R, neighbor=neighbor, **dynamic_kwargs)
                return energy

    elif model == 'PairNN':
        r_cut = 3.  # 3 sigma in LJ units
        hidden_layers = [64, 64]  # with 32 higher best force error

        neighbor_fn = partition.neighbor_list(displacement, box[0], r_cut,
                                              dr_threshold=0.5,
                                              capacity_multiplier=1.5,
                                              fractional_coordinates=fractional)
        nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)
        prior_fn = custom_energy.generic_repulsion_neighborlist(
            displacement,
            sigma=0.7,
            epsilon=1.,
            exp=12,
            initialize_neighbor_list=False,
            r_onset=0.9 * r_cut,
            r_cutoff=r_cut
        )

        init_fn, gnn_energy = neural_networks.pair_interaction_nn(displacement,
                                                                  r_cut,
                                                                  hidden_layers)
        if isinstance(model_init_key, list):
            init_params = [init_fn(key, R_init, neighbor=nbrs_init)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init)

        def energy_fn_template(energy_params):
            gnn_energy_fix = partial(gnn_energy, energy_params, species=species)

            def energy(R, neighbor, **dynamic_kwargs):
                return gnn_energy_fix(R, neighbor, **dynamic_kwargs) + prior_fn(
                    R, neighbor=neighbor, **dynamic_kwargs)
            return energy


    elif model == 'CGDimeNet':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        # former prior: sigma=0.3 eps=1.
        # SPC / SPC/FW: sigma=0.3165, eps=0.65 * 4 = 2.6
        # TODO fix cut-off
        # TODO bug in cell_list prevents use of 3D box --> only works for cube here
        neighbor_fn = partition.neighbor_list(displacement, box[0], r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)
        # TODO can we re-enable after update?
        nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN
        prior_fn = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=0.3165, epsilon=1., exp=12,
            initialize_neighbor_list=False, r_onset=0.9 * r_cut, r_cutoff=r_cut
        )

        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
            displacement, r_cut, n_species, R_init, nbrs_init,
            kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        )

        if isinstance(model_init_key, list):
            init_params = [init_fn(key, R_init, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init,
                                  species=species, **energy_kwargs)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            gnn_energy = partial(GNN_energy, energy_params, species=species)
            def energy(R, neighbor, **dynamic_kwargs):
                return gnn_energy(R, neighbor, **dynamic_kwargs) + prior_fn(R, neighbor=neighbor, **dynamic_kwargs)
            return energy

    elif model == 'SW':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        # former prior: sigma=0.3 eps=1.
        # SPC / SPC/FW: sigma=0.3165, eps=0.65 * 4 = 2.6
        # TODO fix cut-off
        # TODO bug in cell_list prevents use of 3D box --> only works for cube here
        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)
        # TODO can we re-enable after update?
        nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN
        # prior_fn = custom_energy.generic_repulsion_neighborlist(
        #     displacement, sigma=0.3165, epsilon=1., exp=12,
        #     initialize_neighbor_list=False, r_onset=0.9 * r_cut, r_cutoff=r_cut
        # )

        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
            displacement, r_cut, n_species, R_init, nbrs_init,
            kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        )

        if isinstance(model_init_key, list):
            init_params = [init_fn(key, R_init, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init,
                                  species=species, **energy_kwargs)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            gnn_energy = partial(GNN_energy, energy_params, species=species)
            def energy(R, neighbor, **dynamic_kwargs):
                return gnn_energy(R, neighbor, **dynamic_kwargs)  # + prior_fn(R, neighbor=neighbor, **dynamic_kwargs)
            return energy


    elif model == 'TiDimeNet':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        # former prior: sigma=0.3 eps=1.
        # SPC / SPC/FW: sigma=0.3165, eps=0.65 * 4 = 2.6
        # TODO fix cut-off
        # TODO bug in cell_list prevents use of 3D box --> only works for cube here
        # neighbor_fn = partition.neighbor_list(displacement, box[0], r_cut,
        #                                       dr_threshold=0.05,
        #                                       capacity_multiplier=2.,
        #                                       fractional_coordinates=fractional,
        #                                       disable_cell_list=True)
        # TODO can we re-enable after update?
        # nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN
        # prior_fn = custom_energy.generic_repulsion_neighborlist(
        #     displacement, sigma=0.3165, epsilon=1., exp=12,
        #     initialize_neighbor_list=False, r_onset=0.9 * r_cut, r_cutoff=r_cut
        # )

        # init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
        #     displacement, r_cut, n_species, R_init, nbrs_init,
        #     kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        # )

        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist_on_the_fly(r_cutoff=r_cut,
                                                                                n_species=n_species,
                                                                                box_sample=box,
                                                                                position_sample=R_init,
                                                                                species_sample=species,
                                                                                precom_max_edges=precom_max_edges,
                                                                                precom_max_triplets=precom_max_triplets,
                                                                                embed_size=32,
                                                                                init_kwargs=mlp_init)

        # Set unused neighbor_fn and nbrs_init to fit into general select_model() setupt
        neighbor_fn = None
        nbrs_init = None

        # if isinstance(model_init_key, list):
        #     init_params = [init_fn(key, R_init, neighbor=nbrs_init,
        #                            species=species, **energy_kwargs)
        #                    for key in model_init_key]
        # else:
        #     init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init,
        #                           species=species, **energy_kwargs)
        if isinstance(model_init_key, list):
            init_params = [init_fn(key, positions=R_init, box=box, species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, positions=R_init, box=box, species=species,
                                  precom_edge_mask=precom_edge_mask_init, **energy_kwargs)


        # this pattern allows changing the energy parameters on-the-fly
        # def energy_fn_template(energy_params):
        #     gnn_energy = partial(GNN_energy, energy_params, species=species)
        #     def energy(R, neighbor, **dynamic_kwargs):
        #         return gnn_energy(R, neighbor, **dynamic_kwargs) + prior_fn(R, neighbor=neighbor, **dynamic_kwargs)
        #     return energy


        def energy_fn_template(energy_params):
            gnn_energy = partial(GNN_energy, energy_params)

            def energy(position, box, species, precom_edge_mask, **dynamic_kwargs):
                return gnn_energy(position, box, species, precom_edge_mask, **dynamic_kwargs)
            return energy

    elif model == 'TiDimeNet_Prior':
        r_cut = 5.0
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        # former prior: sigma=0.3 eps=1.
        # SPC / SPC/FW: sigma=0.3165, eps=0.65 * 4 = 2.6
        # TODO fix cut-off
        # TODO bug in cell_list prevents use of 3D box --> only works for cube here
        # neighbor_fn = partition.neighbor_list(displacement, box[0], r_cut,
        #                                       dr_threshold=0.05,
        #                                       capacity_multiplier=2.,
        #                                       fractional_coordinates=fractional,
        #                                       disable_cell_list=True)
        # TODO can we re-enable after update?
        # nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN
        # prior_fn = custom_energy.generic_repulsion_neighborlist(
        #     displacement, sigma=0.3165, epsilon=1., exp=12,
        #     initialize_neighbor_list=False, r_onset=0.9 * r_cut, r_cutoff=r_cut
        # )

        # init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
        #     displacement, r_cut, n_species, R_init, nbrs_init,
        #     kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        # )

        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist_on_the_fly(r_cutoff=r_cut,
                                                                                n_species=n_species,
                                                                                box_sample=box,
                                                                                position_sample=R_init,
                                                                                species_sample=species,
                                                                                init_kwargs=mlp_init)

        # Set unused neighbor_fn and nbrs_init to fit into general select_model() setupt
        neighbor_fn = None
        nbrs_init = None

        # if isinstance(model_init_key, list):
        #     init_params = [init_fn(key, R_init, neighbor=nbrs_init,
        #                            species=species, **energy_kwargs)
        #                    for key in model_init_key]
        # else:
        #     init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init,
        #                           species=species, **energy_kwargs)
        if isinstance(model_init_key, list):
            init_params = [init_fn(key, positions=R_init, box=box, species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, positions=R_init, box=box, species=species,
                                  precom_edge_mask=precom_edge_mask_init, **energy_kwargs)


        # this pattern allows changing the energy parameters on-the-fly
        # def energy_fn_template(energy_params):
        #     gnn_energy = partial(GNN_energy, energy_params, species=species)
        #     def energy(R, neighbor, **dynamic_kwargs):
        #         return gnn_energy(R, neighbor, **dynamic_kwargs) + prior_fn(R, neighbor=neighbor, **dynamic_kwargs)
        #     return energy

        # Prior is a generic repulsive potential
        # prior_fn = custom_energy.generic_repulsion_neighborlist_on_the_fly(r_cutoff=r_cut, r_onset=0.9*r_cut)
        prior_fn = custom_energy.meam_pair_interaction_on_the_fly(r_cutoff=r_cut, r_onset=0.9*r_cut)

        def energy_fn_template(energy_params):
            gnn_energy = partial(GNN_energy, energy_params)

            def energy(position, box, species, precom_edge_mask, **dynamic_kwargs):
                return gnn_energy(position, box, species, precom_edge_mask, **dynamic_kwargs) + \
                       prior_fn(position, box, species, precom_edge_mask)
            return energy

    elif model == 'TiDimeNetDiffTRe':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            # 'w_init': hk.initializers.VarianceScaling(scale=0.5),
            # 'b_init': hk.initializers.VarianceScaling(0.1),
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=2.5,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(R_init, extra_capacity=0)  # create neighborlist for init of GNN

        # init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
        #     displacement, r_cut, n_species, R_init, nbrs_init,
        #     kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init
        # )

        init_fn, GNN_energy = neural_networks.dimenetpp_neighborlist(
            displacement, r_cut, n_species, positions_test=R_init, neighbor_test=nbrs_init,
            max_edge_multiplier=2.25, max_triplet_multiplier=2.25, kbt_dependent=kbt_dependent, embed_size=32,
            init_kwargs=mlp_init
        )

        if isinstance(model_init_key, list):
            init_params = [init_fn(key, R_init, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, R_init, neighbor=nbrs_init,
                                  species=species, **energy_kwargs)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            gnn_energy = partial(GNN_energy, energy_params, species=species)
            def energy(R, neighbor, **dynamic_kwargs):
                return gnn_energy(R, neighbor, **dynamic_kwargs)
            return energy

    else:
        raise ValueError('The model' + model + 'is not implemented.')

    return energy_fn_template, neighbor_fn, init_params, nbrs_init


def initialize_simulation(init_class, model, target_dict=None, x_vals=None,
                          key_init=0, fractional=True, integrator='Nose_Hoover',
                          wrapped=True, kbt_dependent=False):

    R_init = init_class.R_init
    key = random.PRNGKey(key_init)
    model_init_key, simuation_init_key = random.split(key, 2)

    box = init_class.box
    box_tensor, scale_fn = custom_space.init_fractional_coordinates(box)
    if fractional:  # need to scale coordinates to unit hypercube
        R_init = scale_fn(R_init)

    displacement, shift = space.periodic_general(box_tensor, fractional_coordinates=fractional, wrapped=wrapped)

    energy_kwargs = {}
    if kbt_dependent:
        energy_kwargs['kT'] = init_class.kbT  # dummy input to allow init of kbt_embedding

    energy_fn_template, neighbor_fn, init_params, nbrs = \
        select_model(model, R_init, displacement, box, model_init_key, init_class.species,
                     x_vals, fractional, kbt_dependent, **energy_kwargs)

    energy_fn_init = energy_fn_template(init_params)

    # setup simulator
    if integrator == 'Nose_Hoover':
        simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift, dt=init_class.dt, kT=init_class.kbT,
                                        chain_length=3, chain_steps=1)
    elif integrator == 'Langevin':
        simulator_template = partial(simulate.nvt_langevin, shift=shift, dt=init_class.dt, kT=init_class.kbT, gamma=5.)
    elif integrator == 'NPT':
        chain_kwargs = {'chain_steps': 1}
        simulator_template = partial(simulate.npt_nose_hoover, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbT,
                                     pressure=init_class.ref_press,
                                     barostat_kwargs=chain_kwargs,
                                     thermostat_kwargs=chain_kwargs)
    else:
        raise NotImplementedError('Integrator string not recognized!')

    init, apply_fn = simulator_template(energy_fn_init)
    # init = jit(init)  # avoid throwing initialization NaN for debugging other NaNs

    # box only used in NPT: needs to be box tensor as 1D box leads to error as
    # box is erroneously mapped over N dimensions (usually only for eps, sigma)
    state = init(simuation_init_key, R_init, mass=init_class.masses,
                 neighbor=nbrs, box=box_tensor, **energy_kwargs)
    sim_state = (state, nbrs)  # store neighbor list together with current simulation state

    if target_dict is None:
        target_dict = {}
    compute_fns, targets = build_quantity_dict(R_init, box_tensor, displacement,
                                               energy_fn_template, nbrs,
                                               target_dict, init_class)

    simulation_funs = (simulator_template, energy_fn_template, neighbor_fn)
    return sim_state, init_params, simulation_funs, compute_fns, targets


def initialize_simulation_ti_hcp(init_class, model, target_dict=None, x_vals=None,
                                 key_init=0, fractional=True, integrator='Nose_Hoover',
                                 wrapped=True, kbt_dependent=False, precom_edge_mask=None,
                                 load_pretrained_model_path=None, load_energy_params=None):

    # TODO: Only started, not yet done with implementation!
    R_init = init_class.R_init
    key = random.PRNGKey(key_init)
    model_init_key, simuation_init_key = random.split(key, 2)

    box = init_class.box
    box_tensor = box
    scale_fn = custom_space.fractional_coordinates_triclinic_box(box)
    if fractional:  # need to scale coordinates to unit hypercube
        R_init = scale_fn(R_init)

    displacement, shift = space.periodic_general(box_tensor, fractional_coordinates=fractional, wrapped=wrapped)

    # energy_kwargs = {}
    # if kbt_dependent:
    #     energy_kwargs['kT'] = init_class.kbT  # dummy input to allow init of kbt_embedding

    # energy_fn_template, neighbor_fn, init_params, nbrs = \
    #     select_model(model=model, R_init=R_init, displacement=displacement, box=box, model_init_key=model_init_key,
    #                  species=init_class.species, x_vals=x_vals, fractional=fractional, kbt_dependent=kbt_dependent,
    #                  precom_edge_mask_init=precom_edge_mask, **energy_kwargs)

    energy_fn_template, neighbor_fn, init_params, nbrs = \
        select_model(model=model, R_init=R_init, displacement=displacement, box=box, model_init_key=model_init_key,
                     species=init_class.species, x_vals=x_vals, fractional=fractional, kbt_dependent=kbt_dependent)

    # Overwrite init_params from previously trained model
    if load_pretrained_model_path is not None:
        with open(load_pretrained_model_path, 'rb') as pickle_file:
            trainer_loaded = pickle.load(pickle_file)
            init_params = trainer_loaded.best_params
            if load_energy_params is not None:
                with open(load_energy_params, 'rb') as pickle_file:
                    params = pickle.load(pickle_file)
                    trainer_loaded.params = params
                    init_params = params
                    # trainer_loaded.best_params = params

        fn = lambda module_name, name, arr: np.array(arr)
        init_params = hk.data_structures.map(fn, init_params)



    energy_fn_init = energy_fn_template(init_params)

    # setup simulator
    if integrator == 'Langevin':
        simulator_template = partial(simulate.nvt_langevin, shift_fn=shift, dt=init_class.dt, kT=init_class.kbT, gamma=4.)
    # elif integrator == 'Nose_Hoover':
    #     simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift, dt=init_class.dt, kT=init_class.kbT,
    #                                     chain_length=3, chain_steps=1)
    else:
        raise NotImplementedError('Integrator string not recognized!')

    init, apply_fn = simulator_template(energy_fn_init)
    # init = jit(init)  # avoid throwing initialization NaN for debugging other NaNs

    # box only used in NPT: needs to be box tensor as 1D box leads to error as
    # box is erroneously mapped over N dimensions (usually only for eps, sigma)
    # TODO: Don't think neighobr=nbrs is acutally used on the fly. Delete from arguments.
    # state = init(simuation_init_key, R_init, mass=init_class.masses,
    #              neighbor=nbrs, box=box_tensor, species=init_class.species, precom_edge_mask=precom_edge_mask,
    #              **energy_kwargs)

    state = init(simuation_init_key, R_init, mass=init_class.masses,
                 neighbor=nbrs)

    sim_state = (state, nbrs)  # store neighbor list together with current simulation state

    if target_dict is None:
        print("This is wrong. Target_dict cannot be none")
        exit(1)

    compute_fns, targets = build_quantity_dict_ti_hcp(R_init, box_tensor, energy_fn_template,
                                                      target_dict, init_class)

    simulation_funs = (simulator_template, energy_fn_template, neighbor_fn)
    return sim_state, init_params, simulation_funs, compute_fns, targets

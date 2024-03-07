"""A collection of functions evaluating quantiities of trajectories.
For easiest integration into chemtain, functions should be compatible with
traj_util.quantity_traj. Functions provided to quantity_traj need to take the
state and additional kwargs.
"""
from functools import partial

from coax.utils._jit import jit
from jax import grad, vmap, lax, jacrev, jacfwd, numpy as jnp
from jax.scipy.stats import norm
from jax_md import space, util, dataclasses, quantity, simulate
import numpy as onp

from chemtrain import sparse_graph

Array = util.Array


def energy_wrapper(energy_fn_template):
    """Wrapper around energy_fn to allow energy computation via
    traj_util.quantity_traj.
    """
    def energy(state, neighbor, energy_params, **kwargs):
        energy_fn = energy_fn_template(energy_params)
        return energy_fn(state.position, neighbor=neighbor, **kwargs)
    return energy


def temperature(state, **unused_kwargs):
    """Temperature function that is consistent with quantity_traj interface.
        JaxMD>2.0: Use momentum instead of velocity."""
    return quantity.temperature(momentum=state.momentum, mass=state.mass)


def _dyn_box(reference_box, **kwargs):
    """Gets box dynamically from kwargs, if provided, otherwise defaults to
    reference. Ensures that a box is provided and deletes from kwargs.
    """
    box = kwargs.pop('box', reference_box)
    assert box is not None, ('If no reference box is given, needs to be '
                             'given as kwarg "box".')
    return box, kwargs


def volume_npt(state, **unused_kwargs):
    """Returns volume of a single snapshot in the NPT ensemble, e.g. for use in
     DiffTRe learning of thermodynamic fluctiations in chemtrain.traj_quantity.
     """
    dim = state.position.shape[-1]
    box = simulate.npt_box(state)
    volume = quantity.volume(dim, box)
    return volume


def _canonicalized_masses(state):
    if state.mass.ndim == 0:
        masses = jnp.ones(state.position.shape[0]) * state.mass
    else:
        masses = state.mass
    return masses


def density(state, **unused_kwargs):
    """Returns density of a single snapshot of the NPT ensemble."""
    masses = _canonicalized_masses(state)
    total_mass = jnp.sum(masses)
    volume = volume_npt(state)
    return total_mass / volume


# TODO distinct classes and discretization functions don't seem optimal
#  --> possible refactor


@dataclasses.dataclass
class RDFParams:
    """Hyperparameters to initialize the radial distribution function (RDF).

    Attributes:
    reference_rdf: The target rdf; initialize with None if no target available
    rdf_bin_centers: The radial positions of the centers of the rdf bins
    rdf_bin_boundaries: The radial positions of the edges of the rdf bins
    sigma_RDF: Standard deviation of smoothing Gaussian
    """
    reference: Array
    rdf_bin_centers: Array
    rdf_bin_boundaries: Array
    sigma: Array


def rdf_discretization(rdf_cut, nbins=300, rdf_start=0.):
    """Computes dicretization parameters for initialization of RDF function.

    Args:
        rdf_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in radial direction
        rdf_start: Minimal distance after which particle pairs are considered

    Returns:
        Arrays with radial positions of bin centers, bin edges and the standard
        deviation of the Gaussian smoothing kernel.

    """
    dx_bin = (rdf_cut - rdf_start) / float(nbins)
    rdf_bin_centers = jnp.linspace(rdf_start + dx_bin / 2.,
                                   rdf_cut - dx_bin / 2.,
                                   nbins)
    rdf_bin_boundaries = jnp.linspace(rdf_start, rdf_cut, nbins + 1)
    sigma_rdf = jnp.array(dx_bin)
    return rdf_bin_centers, rdf_bin_boundaries, sigma_rdf



def _ideal_gas_density(particle_density, bin_boundaries):
    """Returns bin densities that would correspond to an ideal gas."""
    r_small = bin_boundaries[:-1]
    r_large = bin_boundaries[1:]
    bin_volume = (4. / 3.) * jnp.pi * (r_large**3 - r_small**3)
    bin_weights = bin_volume * particle_density
    return bin_weights


def init_rdf(displacement_fn, rdf_params, reference_box=None):
    """Initializes a function that computes the radial distribution function
    (RDF) for a single state.

    Args:
        displacement_fn: Displacement function
        rdf_params: RDFParams defining the hyperparameters of the RDF
        reference_box: Simulation box. Can be provided here for constant boxes
                       or on-the-fly as kwarg 'box', e.g. for NPT ensemble

    Returns:
        A function taking a simulation state and returning the instantaneous RDF
    """
    _, bin_centers, bin_boundaries, sigma = dataclasses.astuple(rdf_params)
    distance_metric = space.canonicalize_displacement_or_metric(displacement_fn)
    bin_size = jnp.diff(bin_boundaries)

    def pair_corr_fun(position, box):
        """Returns instantaneous pair correlation function while ensuring
        each particle pair contributes exactly 1.
        """
        n_particles = position.shape[0]
        metric = partial(distance_metric, box=box)
        metric = space.map_product(metric)
        dr = metric(position, position)
        # neglect same particles i.e. distance = 0.
        dr = jnp.where(dr > util.f32(1.e-7), dr, util.f32(1.e7))

        #  Gaussian ensures that discrete integral over distribution is 1
        exp = jnp.exp(util.f32(-0.5) * (dr[:, :, jnp.newaxis] - bin_centers)**2
                      / sigma**2)
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma**2)
        pair_corr_per_particle = util.high_precision_sum(gaussian_distances,
                                                         axis=1)  # sum nbrs
        mean_pair_corr = util.high_precision_sum(pair_corr_per_particle,
                                                 axis=0) / n_particles
        return mean_pair_corr

    def rdf_compute_fun(state, **kwargs):
        box, _ = _dyn_box(reference_box, **kwargs)
        # Note: we cannot use neighbor list since RDF cutoff and
        # neighbor list cut-off don't coincide in general
        n_particles, spatial_dim = state.position.shape
        total_vol = quantity.volume(spatial_dim, box)
        particle_density = n_particles / total_vol
        mean_pair_corr = pair_corr_fun(state.position, box)
        # RDF is defined to relate the particle densities to an ideal gas.
        rdf = mean_pair_corr / _ideal_gas_density(particle_density,
                                                  bin_boundaries)
        return rdf
    return rdf_compute_fun


def kinetic_energy_tensor(state):
    """Computes the kinetic energy tensor of a single snapshot.

    Args:
        state: Jax_md simulation state

    Returns:
        Kinetic energy tensor
    """
    average_velocity = jnp.mean(state.velocity, axis=0)
    thermal_excitation_velocity = state.velocity - average_velocity
    diadic_velocity_product = vmap(lambda v: jnp.outer(v, v))
    velocity_tensors = diadic_velocity_product(thermal_excitation_velocity)
    return util.high_precision_sum(state.mass * velocity_tensors, axis=0)


def virial_potential_part(energy_fn, state, nbrs, box_tensor, **kwargs):
    """Interaction part of the virial pressure tensor for a single snaphot
    based on the formulation of Chen at al. (2020). See
    init_virial_stress_tensor. for details."""
    energy_fn_ = lambda pos, neighbor, box: energy_fn(
        pos, neighbor=neighbor, box=box, **kwargs)  # for grad
    position = state.position  # in unit box if fractional coordinates used
    negative_forces, box_gradient = grad(energy_fn_, argnums=[0, 2])(
        position, nbrs, box_tensor)
    position = space.transform(box_tensor, position)  # back to real positions
    force_contribution = jnp.dot(negative_forces.T, position)

    # Old box contribution, when row vectors were used
    # box_contribution = jnp.dot(box_gradient.T, box_tensor)

    # Corrected box contribution for column edge vectors instead of row edge vectors.
    box_contribution = jnp.dot(box_gradient, box_tensor.T)
    return force_contribution + box_contribution


def virial_potential_part_on_the_fly(energy_fn, state, box_tensor, species, precom_edge_mask):
    """Interaction part of the virial pressure tensor for a single snaphot
    based on the formulation of Chen at al. (2020). See
    init_virial_stress_tensor. for details."""

    energy_fn_ = lambda pos, box, species_lambda, precom_edge_mask_lambda: energy_fn(position=pos, box=box,
                                                                                     species=species_lambda,
                                                                                     precom_edge_mask=precom_edge_mask_lambda)  # for grad

    position = state.position  # in unit box if fractional coordinates used

    negative_forces, box_gradient = grad(energy_fn_, argnums=[0, 1])(
        position, box_tensor, species, precom_edge_mask)

    position = space.transform(box_tensor, position)  # back to real positions
    force_contribution = jnp.dot(negative_forces.T, position)

    # Old box contirbution, when row vectors were used
    # box_contribution = jnp.dot(box_gradient.T, box_tensor)

    # Box contribution when column vectors are used.
    box_contribution = jnp.dot(box_gradient, box_tensor.T)

    return force_contribution + box_contribution


def init_virial_stress_tensor(energy_fn_template, ref_box_tensor=None,
                              include_kinetic=True, pressure_tensor=False):
    """Initializes a function that computes the virial stress tensor for a
    single state.

    This function is applicable to arbitrary many-body interactions, even
    under periodic boundary conditions. This implementation is based on the
    formulation of Chen et al. (2020), which is well-suited for vectorized,
    differentiable MD libararies. This function requires that `energy_fn`
    takes a `box` keyword argument, usually alongside `periodic_general`
    boundary conditions.

    Chen et al. "TensorAlloy: An automatic atomistic neural network program
    for alloys". Computer Physics Communications 250 (2020): 107057

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for the NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.
        pressure_tensor: If False (default), returns the stress tensor. If True,
                         returns the pressure tensor, i.e. the negative stress
                         tensor.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        virial stress tensor.
    """
    if pressure_tensor:
        pressure_sign = -1.
    else:
        pressure_sign = 1.

    def virial_stress_tensor_neighborlist(state, neighbor, energy_params,
                                          **kwargs):
        # Note: this workaround with the energy_template was needed to keep
        #       the function jitable when changing energy_params on-the-fly
        # TODO function to transform box to box-tensor
        box, kwargs = _dyn_box(ref_box_tensor, **kwargs)
        energy_fn = energy_fn_template(energy_params)
        virial_tensor = virial_potential_part(energy_fn, state, neighbor, box,
                                              **kwargs)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        if include_kinetic:
            kinetic_tensor = -1 * kinetic_energy_tensor(state)
            return pressure_sign * (kinetic_tensor + virial_tensor) / volume
        else:
            return pressure_sign * virial_tensor / volume

    return virial_stress_tensor_neighborlist


def init_virial_stress_tensor_on_the_fly(energy_fn_template,
                                         include_kinetic=True, pressure_tensor=False,
                                         only_virial=True):
    """Initializes a function that computes the virial stress tensor for a
    single state.

    This function is applicable to arbitrary many-body interactions, even
    under periodic boundary conditions. This implementation is based on the
    formulation of Chen et al. (2020), which is well-suited for vectorized,
    differentiable MD libararies. This function requires that `energy_fn`
    takes a `box` keyword argument, usually alongside `periodic_general`
    boundary conditions.

    Chen et al. "TensorAlloy: An automatic atomistic neural network program
    for alloys". Computer Physics Communications 250 (2020): 107057

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for the NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.
        pressure_tensor: If False (default), returns the stress tensor. If True,
                         returns the pressure tensor, i.e. the negative stress
                         tensor.
        only_virial:    If only_virial is True, then this function returns virial
                        tensor. Differes from pressure tensor as we don't divide by
                        box volume.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        virial stress tensor.
    """
    if pressure_tensor:
        pressure_sign = -1.
    else:
        pressure_sign = 1.

    def virial_stress_tensor_neighborlist_on_the_fly(state, box, species, precom_edge_mask, energy_params):
        # Note: this workaround with the energy_template was needed to keep
        #       the function jitable when changing energy_params on-the-fly
        # TODO function to transform box to box-tensor
        # box, kwargs = _dyn_box(ref_box_tensor, **kwargs)
        energy_fn = energy_fn_template(energy_params)
        virial_tensor = virial_potential_part_on_the_fly(energy_fn=energy_fn, state=state, box_tensor=box,
                                                         species=species, precom_edge_mask=precom_edge_mask)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        if only_virial:
            return pressure_sign * virial_tensor
        elif include_kinetic:
            kinetic_tensor = -1 * kinetic_energy_tensor(state)
            return pressure_sign * (kinetic_tensor + virial_tensor) / volume
        else:
            return pressure_sign * virial_tensor / volume

    return virial_stress_tensor_neighborlist_on_the_fly


def init_pressure(energy_fn_template, ref_box_tensor=None,
                  include_kinetic=True):
    """Initializes a function that computes the pressure for a single state.

    This function is applicable to arbitrary many-body interactions, even
    under periodic boundary conditions. See `init_virial_stress_tensor`
    for details.

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        pressure.
    """
    # pressure is negative hydrostatic stress
    stress_tensor_fn = init_virial_stress_tensor(
        energy_fn_template, ref_box_tensor, include_kinetic=include_kinetic,
        pressure_tensor=True
    )

    def pressure_neighborlist(state, neighbor, energy_params, **kwargs):
        pressure_tensor = stress_tensor_fn(state, neighbor, energy_params,
                                           **kwargs)
        return jnp.trace(pressure_tensor) / 3.
    return pressure_neighborlist


def energy_under_strain(epsilon, energy_fn, box_tensor, state, neighbor,
                        **kwargs):
    """Potential energy of a state after applying linear strain epsilon."""
    # Note: When computing the gradient, we deal with infinitesimally
    #       small strains. Linear strain theory is therefore valid and
    #       additionally tan(gamma) = gamma. These assumptions are used
    #       computing the box after applying the stain.
    # strained_box = jnp.dot(box_tensor, jnp.eye(box_tensor.shape[0]) + epsilon)
    strained_box = jnp.dot(jnp.eye(box_tensor.shape[0]) + epsilon, box_tensor)
    energy = energy_fn(state.position, neighbor=neighbor, box=strained_box,
                       **kwargs)
    return energy


def init_sigma_born(energy_fn_template, ref_box_tensor=None):
    """Initialiizes a function that computes the Born contribution to the
    stress tensor.

    sigma^B_ij = d U / d epsilon_ij

    Can also be computed to compute the stress tensor at kbT = 0, when called
    on the state of minimum energy. This function requires that `energy_fn`
    takes a `box` keyword argument, usually alongside `periodic_general`
    boundary conditions.

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for the NPT ensemble.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        Born contribution to the stress tensor.
    """
    def sigma_born(state, neighbor, energy_params, **kwargs):
        box, kwargs = _dyn_box(ref_box_tensor, **kwargs)
        spatial_dim = box.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

        energy_fn = energy_fn_template(energy_params)
        sigma_b = jacrev(energy_under_strain)(
            epsilon0, energy_fn, box, state, neighbor, **kwargs)
        return sigma_b / volume
    return sigma_born


def init_stiffness_tensor_stress_fluctuation(energy_fn_template, box_tensor,
                                             kbt, n_particles):
    """Initializes all functions necessary to compute the elastic stiffness
    tensor via the stress fluctuation method in the NVT ensemble.

    The provided functions compute all necessary instantaneous properties
    necessary to compute the elastic stiffness tensor via the stress fluctuation
     method. However, for compatibility with DiffTRe, (weighted) ensemble
     averages need to be computed manually and given to the stiffness_tensor_fn
     for final computation of the stiffness tensor. For an example usage see
     the diamond notebook. The implementation follows the formulation derived by
     Van Workum et al., "Isothermal stress and elasticity tensors for ions and
     point dipoles using Ewald summations", PHYSICAL REVIEW E 71, 061102 (2005).

     # TODO provide sample usage

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        box_tensor: The transformation T of general periodic boundary
                    conditions. As the stress-fluctuation method is only
                    applicable to the NVT ensemble, the box_tensor needs to be
                    provided here as a constant, not on-the-fly.
        kbt: Temperature in units of the Boltzmann constant
        n_particles: Number of particles in the box

    Returns: A tuple of 3 functions:
        born_term_fn: A function computing the Born contribution to the
                      stiffness tensor for a single snapshot
        sigma_born: A function computing the Born contribution to the stress
                    tensor for a single snapshot
        sigma_tensor_prod: A function computing sigma^B_ij * sigma^B_kl given
                           a trajectory of sigma^B_ij
        stiffness_tensor_fn: A function taking ensemble averages of C^B_ijkl,
                             sigma^B_ij and sigma^B_ij * sigma^B_kl and
                             returning the resulting stiffness tensor.
    """
    # TODO this function simplifies a lot if split between per-snapshot
    #  and per-trajectory functions
    spatial_dim = box_tensor.shape[-1]
    volume = quantity.volume(spatial_dim, box_tensor)
    epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

    def born_term_fn(state, neighbor, energy_params, **kwargs):
        """Born contribution to the stiffness tensor:
        C^B_ijkl = d^2 U / d epsilon_ij d epsilon_kl
        """
        energy_fn = energy_fn_template(energy_params)
        born_stiffness_contribution = jacfwd(jacrev(energy_under_strain))(
            epsilon0, energy_fn, box_tensor, state, neighbor, **kwargs)
        return born_stiffness_contribution / volume

    @vmap
    def sigma_tensor_prod(sigma):
        """A function that computes sigma_ij * sigma_kl for a whole trajectory
        to be averaged afterwards.
        """
        return jnp.einsum('ij,kl->ijkl', sigma, sigma)

    def stiffness_tensor_fn(mean_born, mean_sigma, mean_sig_ij_sig_kl):
        """Computes the stiffness tensor given ensemble averages of
        C^B_ijkl, sigma^B_ij and sigma^B_ij * sigma^B_kl.
        """
        sigma_prod = jnp.einsum('ij,kl->ijkl', mean_sigma, mean_sigma)
        delta_ij = jnp.eye(spatial_dim)
        delta_ik_delta_jl = jnp.einsum('ik,jl->ijkl', delta_ij, delta_ij)
        delta_il_delta_jk = jnp.einsum('il,jk->ijkl', delta_ij, delta_ij)
        # Note: maybe use real kinetic energy of trajectory rather than target
        #       kbt?
        kinetic_term = n_particles * kbt / volume * (
                delta_ik_delta_jl + delta_il_delta_jk)
        delta_sigma = mean_sig_ij_sig_kl - sigma_prod
        return mean_born - volume / kbt * delta_sigma + kinetic_term

    sigma_born = init_sigma_born(energy_fn_template, box_tensor)

    return born_term_fn, sigma_born, sigma_tensor_prod, stiffness_tensor_fn


def init_stiffness_tensor_stress_fluctuation_0K(energy_fn_template, box_tensor):
    """Initializes all functions necessary to compute the elastic stiffness
    tensor via the stress fluctuation method in the NVT ensemble.

    The provided functions compute all necessary instantaneous properties
    necessary to compute the elastic stiffness tensor via the stress fluctuation
    method at 0K. However, for compatibility with DiffTRe, (weighted) ensemble
    averages need to be computed manually and given to the stiffness_tensor_fn
    for final computation of the stiffness tensor. For an example usage see
    the diamond notebook. The implementation follows the formulation derived by
    Van Workum et al., "Isothermal stress and elasticity tensors for ions and
    point dipoles using Ewald summations", PHYSICAL REVIEW E 71, 061102 (2005).

    # TODO provide sample usage

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        box_tensor: The transformation T of general periodic boundary
                    conditions. As the stress-fluctuation method is only
                    applicable to the NVT ensemble, the box_tensor needs to be
                    provided here as a constant, not on-the-fly.

    Returns: A tuple of 3 functions:
        born_term_fn_0K: A function computing the Born contribution to the
                      stiffness tensor for a single snapshot
        stiffness_tensor_fn_0K: A function taking ensemble averages of C^B_ijkl,
                             sigma^B_ij and sigma^B_ij * sigma^B_kl and
                             returning the resulting stiffness tensor.
    """
    # TODO this function simplifies a lot if split between per-snapshot
    #  and per-trajectory functions
    spatial_dim = box_tensor.shape[-1]
    volume = quantity.volume(spatial_dim, box_tensor)
    epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

    def born_term_fn_0K(state, neighbor, energy_params, **kwargs):
        """Born contribution to the stiffness tensor:
        C^B_ijkl = d^2 U / d epsilon_ij d epsilon_kl
        """
        energy_fn = energy_fn_template(energy_params)
        born_stiffness_contribution = jacfwd(jacrev(energy_under_strain))(
            epsilon0, energy_fn, box_tensor, state, neighbor, **kwargs)
        return born_stiffness_contribution / volume

    def stiffness_tensor_fn_0K(mean_born):
        """Computes the stiffness tensor at 0K given ensemble averages of
        C^B_ijkl.
        """
        return mean_born

    return born_term_fn_0K, stiffness_tensor_fn_0K


def stiffness_tensor_components_cubic_crystal(stiffness_tensor):
    """Computes the 3 independent elastic stiffness components of a cubic
    crystal from the whole stiffness tensor.

    The number of independent components in a general stiffness tensor is 21
    for isotropic pressure. For a cubic crystal, these 21 parameters only take
    3 distinct values: c11, c12 and c44. We compute these values from averages
    using all 21 components for variance reduction purposes.

    Args:
        stiffness_tensor: The full (3, 3, 3, 3) elastic stiffness tensor

    Returns:
        A (3,) ndarray containing (c11, c12, c44)
    """
    # TODO likely there exists a better formulation via Einstein notation
    c = stiffness_tensor
    c11 = (c[0, 0, 0, 0] + c[1, 1, 1, 1] + c[2, 2, 2, 2]) / 3.
    c12 = (c[0, 0, 1, 1] + c[1, 1, 0, 0] + c[0, 0, 2, 2] + c[2, 2, 0, 0]
           + c[1, 1, 2, 2] + c[2, 2, 1, 1]) / 6.
    c44 = (c[0, 1, 0, 1] + c[1, 0, 0, 1] + c[0, 1, 1, 0] + c[1, 0, 1, 0] +
           c[0, 2, 0, 2] + c[2, 0, 0, 2] + c[0, 2, 2, 0] + c[2, 0, 2, 0] +
           c[2, 1, 2, 1] + c[1, 2, 2, 1] + c[2, 1, 1, 2] + c[1, 2, 1, 2]) / 12.

    print("C11 components: ", c[0, 0, 0, 0], ", ", c[1, 1, 1, 1], ", ",c[2, 2, 2, 2])
    print("C12 components: ", c[0, 0, 1, 1], ", ", c[1, 1, 0, 0], ", ", c[0, 0, 2, 2], ", ", c[2, 2, 0, 0]
          , ", ", c[1, 1, 2, 2], ", ", c[2, 2, 1, 1])
    print("C44 components: ", c[0, 1, 0, 1], ", ", c[1, 0, 0, 1], ", ", c[0, 1, 1, 0], ", ", c[1, 0, 1, 0], ", ",
          c[0, 2, 0, 2], ", ", c[2, 0, 0, 2], ", ", c[0, 2, 2, 0], ", ", c[2, 0, 2, 0], ", ",
          c[2, 1, 2, 1], ", ", c[1, 2, 2, 1], ", ", c[2, 1, 1, 2], ", ", c[1, 2, 1, 2])


    return jnp.array([c11, c12, c44])


def cij_triclinic_cell(stiffness_tensor):
    """Function takes in 3x3x3x3 stiffness tensor and return 6x6 Voigt notation"""

    c = stiffness_tensor
    # Voigt notation from https://de.wikipedia.org/wiki/Voigtsche_Notation
    # Red entries have one component
    C_11 = c[0, 0, 0, 0]
    C_22 = c[1, 1, 1, 1]
    C_33 = c[2, 2, 2, 2]
    C_12 = c[0, 0, 1, 1]
    C_13 = c[0, 0, 2, 2]
    C_21 = c[1, 1, 0, 0]
    C_23 = c[1, 1, 2, 2]
    C_31 = c[2, 2, 0, 0]
    C_32 = c[2, 2, 1, 1]

    # Blue entries have two components
    # Top right
    C_14 = (c[0, 0, 1, 2] + c[0, 0, 2, 1]) / 2
    C_15 = (c[0, 0, 0, 2] + c[0, 0, 2, 0]) / 2
    C_16 = (c[0, 0, 0, 1] + c[0, 0, 1, 0]) / 2
    C_24 = (c[1, 1, 1, 2] + c[1, 1, 2, 1]) / 2
    C_25 = (c[1, 1, 0, 2] + c[1, 1, 2, 0]) / 2
    C_26 = (c[1, 1, 0, 1] + c[1, 1, 1, 0]) / 2
    C_34 = (c[2, 2, 1, 2] + c[2, 2, 2, 1]) / 2
    C_35 = (c[2, 2, 0, 2] + c[2, 2, 2, 0]) / 2
    C_36 = (c[2, 2, 0, 1] + c[2, 2, 1, 0]) / 2
    # Bottom left
    C_41 = (c[1, 2, 0, 0] + c[2, 1, 0, 0]) / 2
    C_42 = (c[1, 2, 1, 1] + c[2, 1, 1, 1]) / 2
    C_43 = (c[1, 2, 2, 2] + c[2, 1, 2, 2]) / 2
    C_51 = (c[0, 2, 0, 0] + c[2, 0, 0, 0]) / 2
    C_52 = (c[0, 2, 1, 1] + c[2, 0, 1, 1]) / 2
    C_53 = (c[0, 2, 2, 2] + c[2, 0, 2, 2]) / 2
    C_61 = (c[0, 1, 0, 0] + c[1, 0, 0, 0]) / 2
    C_62 = (c[0, 1, 1, 1] + c[1, 0, 1, 1]) / 2
    C_63 = (c[0, 1, 2, 2] + c[1, 0, 2, 2]) / 2

    # Black entries have 4 components
    C_44 = (c[1, 2, 1, 2] + c[1, 2, 2, 1] + c[2, 1, 1, 2] + c[2, 1, 2, 1]) / 4
    C_45 = (c[1, 2, 0, 2] + c[1, 2, 2, 0] + c[2, 1, 0, 2] + c[2, 1, 2, 0]) / 4
    C_46 = (c[1, 2, 0, 1] + c[1, 2, 1, 0] + c[2, 1, 0, 1] + c[2, 1, 1, 0]) / 4
    C_54 = (c[0, 2, 1, 2] + c[0, 2, 2, 1] + c[2, 0, 1, 2] + c[2, 0, 2, 1]) / 4
    C_55 = (c[0, 2, 0, 2] + c[0, 2, 2, 0] + c[2, 0, 0, 2] + c[2, 0, 2, 0]) / 4
    C_56 = (c[0, 2, 0, 1] + c[0, 2, 1, 0] + c[2, 0, 0, 1] + c[2, 0, 1, 0]) / 4
    C_64 = (c[0, 1, 1, 2] + c[0, 1, 2, 1] + c[1, 0, 1, 2] + c[1, 0, 2, 1]) / 4
    C_65 = (c[0, 1, 0, 2] + c[0, 1, 2, 0] + c[1, 0, 0, 2] + c[1, 0, 2, 0]) / 4
    C_66 = (c[0, 1, 0, 1] + c[0, 1, 1, 0] + c[1, 0, 0, 1] + c[1, 0, 1, 0]) / 4

    mat = onp.array([[C_11, C_12, C_13, C_14, C_15, C_16], [C_21, C_22, C_23, C_24, C_25, C_26],
                     [C_31, C_32, C_33, C_34, C_35, C_36], [C_41, C_42, C_43, C_44, C_45, C_46],
                     [C_51, C_52, C_53, C_54, C_55, C_56], [C_61, C_62, C_63, C_64, C_65, C_66]])

    return mat


def stiffness_tensor_components_hexagonal_crystal(stiffness_tensor):
    """ Computes 5 independent elastic stiff components of a hexagonal
    crystal from the whole stiffness tensor

    For a hexagonal structure the 5 components are c11, c33, c44, c12, c13

    Args:
        stiffness_tensor: The full(3,3,3,3) elastic stiffness tensor

    Returns:
          A (5,) ndarray containing (c11, c33, c44, c12, c13)
        """
    c = stiffness_tensor
    c11 = (c[0, 0, 0, 0] + c[1, 1, 1, 1]) / 2.
    c33 = c[2, 2, 2, 2]
    c44 = (c[0, 2, 0, 2] + c[2, 0, 0, 2] + c[0, 2, 2, 0] + c[2, 0, 2, 0] +
           c[2, 1, 2, 1] + c[1, 2, 2, 1] + c[2, 1, 1, 2] + c[1, 2, 1, 2]) / 8.
    c12 = (c[0, 0, 1, 1] + c[1, 1, 0, 0]) / 2.
    c13 = (c[0, 0, 2, 2] + c[2, 2, 0, 0] + c[1, 1, 2, 2] + c[2, 2, 1, 1]) / 4.


    # convert_from_GPa_to_kJ_mol_nm_3 = 10 ** 3 / 1.66054
    # print('Target C11: ', 175)
    # print('C_0000: ', c[0, 0, 0, 0] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_1111: ', c[1, 1, 1, 1] / convert_from_GPa_to_kJ_mol_nm_3)
    #
    # print('Target C33: ', 188)
    # print('C_2222: ', c[2, 2, 2, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    #
    # print('Target C44: ', 58)
    # print('C_0202: ', c[0, 2, 0, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_2002: ', c[2, 0, 0, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_0220: ', c[0, 2, 2, 0] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_2020: ', c[2, 0, 2, 0] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_2121: ', c[2, 1, 2, 1] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_1221: ', c[1, 2, 2, 1] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_2112: ', c[2, 1, 1, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_1212: ', c[1, 2, 1, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    #
    # print('Target C12: ', 95)
    # print('C_0011: ', c[0, 0, 1, 1] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_1100: ', c[1, 1, 0, 0] / convert_from_GPa_to_kJ_mol_nm_3)
    #
    # print('Target C13: ', 72)
    # print('C_0022: ', c[0, 0, 2, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_2200: ', c[2, 2, 0, 0] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_1122: ', c[1, 1, 2, 2] / convert_from_GPa_to_kJ_mol_nm_3)
    # print('C_2211: ', c[2, 2, 1, 1] / convert_from_GPa_to_kJ_mol_nm_3)
    # #
    # #
    # #
    # print('Should be equal: ', c[0, 0, 0, 0], ' ', c[1, 1, 1, 1])
    # print('Should be equal: ', c[0, 2, 0, 2], ' ', c[2, 0, 0, 2], ' ', c[0, 2, 2, 0], ' ', c[2, 0, 2, 0], ' ',
    #       c[2, 1, 2, 1], ' ', c[1, 2, 2, 1], ' ', c[2, 1, 1, 2], ' ', c[1, 2, 1, 2])
    # print('Should be equal: ', c[0, 0, 1, 1], ' ', c[1, 1, 0, 0])
    # print('Should be equal: ', c[0, 0, 2, 2], ' ', c[2, 2, 0, 0], ' ', c[1, 1, 2, 2], ' ', c[2, 2, 1, 1])
    #
    # for i in range(0,3):
    #     for j in range(0,3):
    #         for k in range(0,3):
    #             for l in range(0,3):
    #                 print('c[{}][{}][{}][{}]: {} '.format(i, j, k, l, c[i,j,k,l]/convert_from_GPa_to_kJ_mol_nm_3))

    return jnp.array([c11, c33, c44, c12, c13])
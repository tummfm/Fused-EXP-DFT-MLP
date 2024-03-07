from jax_md import space, energy, simulate, quantity
from jax import custom_jvp, jit, random, lax, grad
import jax.numpy as np
import numpy as onp


"""new implementation of periodic_general from issue 116"""

periodic_displacement = space.periodic_displacement
pairwise_displacement = space.pairwise_displacement
periodic_shift = space.periodic_shift

f32 = np.float32


def inverse(box):
    if np.isscalar(box) or box.size == 1:
        return 1 / box
    elif box.ndim == 1:
        return 1 / box
    elif box.ndim == 2:
        return np.linalg.inv(box)

    raise ValueError()


def get_free_indices(n):
    return ''.join([chr(ord('a') + i) for i in range(n)])


def base_transform(box, R):
    if np.isscalar(box) or box.size == 1:
        return R * box
    elif box.ndim == 1:
        indices = get_free_indices(R.ndim - 1) + 'i'
        return np.einsum(f'i,{indices}->{indices}', box, R)
    elif box.ndim == 2:
        free_indices = get_free_indices(R.ndim - 1)
        left_indices = free_indices + 'j'
        right_indices = free_indices + 'i'
        return np.einsum(f'ij,{left_indices}->{right_indices}', box, R)
    raise ValueError()


@custom_jvp
def transform_without_tangents(box, R):
    return base_transform(box, R)


@transform_without_tangents.defjvp
def transform_without_tangents_jvp(primals, tangents):
    box, R = primals
    dbox, dR = tangents

    return (transform_without_tangents(box, R),
            dR + transform_without_tangents(dbox, R))


def transform(box, R, fractional_coordinates=True):
    if not fractional_coordinates:
        return base_transform(box, R)
    return transform_without_tangents(box, R)


def periodic_general(box, fractional_coordinates=True, wrapped=True):
    inv_box = inverse(box)

    def displacement_fn(Ra, Rb, **kwargs):
        _box, _inv_box = box, inv_box

        if 'box' in kwargs:
            _box = kwargs['box']

            if not fractional_coordinates:
                _inv_box = inverse(_box)

        if 'new_box' in kwargs:
            _box = kwargs['new_box']

        if not fractional_coordinates:
            Ra = transform(_inv_box, Ra)
            Rb = transform(_inv_box, Rb)

        dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
        return transform(_box, dR, fractional_coordinates=fractional_coordinates)

    def u(R, dR):
        if wrapped:
            return periodic_shift(f32(1.0), R, dR)
        return R + dR

    def shift_fn(R, dR, **kwargs):
        if not fractional_coordinates and not wrapped:
            return R + dR

        _box, _inv_box = box, inv_box
        if 'box' in kwargs:
            _box = kwargs['box']
            _inv_box = inverse(_box)

        if 'new_box' in kwargs:
            _box = kwargs['new_box']

        dR = transform(_inv_box, dR, fractional_coordinates=fractional_coordinates)
        if not fractional_coordinates:
            R = transform(_inv_box, R)

        R = u(R, dR)

        if not fractional_coordinates:
            R = transform(_box, R)

        return R

    return displacement_fn, shift_fn


"""LJ system adapted from nve_neighbor_list jupyter notebook"""

Nx = particles_per_side = 80
spacing = np.float32(1.25)
side_length = Nx * spacing

R = onp.stack([onp.array(r) for r in onp.ndindex(Nx, Nx)]) * spacing
R = np.array(R, np.float64)


box = np.ones(2) * side_length  # standard definition of rectangular box
box_tensor = np.array([[box[0], 0.], [0., box[1]]])  # same box, only represented as tensor

standard_displacement, standard_shift = space.periodic(box)
general_displacement, general_shift = periodic_general(box_tensor, fractional_coordinates=False)

standard_energy_fn = jit(energy.lennard_jones_pair(standard_displacement))
energy_fn_general = jit(energy.lennard_jones_pair(general_displacement))

# energies are initially the same
print('E_standard = {}'.format(standard_energy_fn(R)))
print('E_general = {}'.format(energy_fn_general(R)))

# initial forces match closely
print('Force difference:', np.mean(-grad(standard_energy_fn)(R) + grad(energy_fn_general)(R)))


init_fn, apply_fn = simulate.nve(standard_energy_fn, standard_shift, 1e-3)  # works, temperatures stable below 1.
init_fn, apply_fn = simulate.nve(energy_fn_general, general_shift, 1e-3)  # temperatures quickly diverge


state = init_fn(random.PRNGKey(0), R)
body_fn = lambda _, state: (apply_fn(state))

step = 0
while step < 30:
  state = lax.fori_loop(0, 100, body_fn, state)
  print('Temperature at step', step, ':', quantity.temperature(state.velocity, state.mass))
  step += 1

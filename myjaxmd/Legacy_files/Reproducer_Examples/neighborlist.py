import jax.numpy as jnp
from jax_md import space, partition

# all particles interacting within larger box
box = jnp.eye(3) * 3
r_cut = 0.5
_positions = jnp.linspace(1., 1.2, 20)
positions = jnp.stack([_positions, _positions, _positions], axis=1)

displacement, _ = space.periodic(box)

neighbor_fn = partition.neighbor_list(displacement, box, r_cut, 0.1*r_cut,
                                      capacity_multiplier=1.5)

nbrs = neighbor_fn.allocate(positions)

print(nbrs.idx)
new_nbrs = nbrs.update(positions)




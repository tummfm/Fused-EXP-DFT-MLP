from jax import vmap, numpy as jnp, jit, ops
from jax_md import space

analyze_diffusion = False


def mean_squared_displacements(trajectory, num_lags, subsample=1):
    """Computes Mean Squared Displacement (MSD) values from a given trajectory.

    Note that the trajectory needs to be "unwrapped" such that periodic BCs
    do not artificially constrain diffusion. The series of time lags is defined
    implicitly by Delta lag = subsample * time per printout.

    Args:
        trajectory: Trajectory of states
        num_lags: Number of time lags to compute; determines the maximum lag
        subsample: Only consider every subsample state of the trajectory

    Returns:
        An array of MSD values for different time lags averaged over all
        particles in the box
    """
    # trajectory needs to be unwrapped
    positions = trajectory.position[::subsample]  # only every subsampled state
    assert positions.shape[0] > num_lags, (f'The trajectory is not long enough '
                                           f'for the specified number of lags: '
                                           f'{positions.shape[0]} positions vs '
                                           f'{num_lags} lags.')
    # TODO most likely this will not work in fractional coordinates!
    non_periodic_displacement = vmap(vmap(space.pairwise_displacement))
    lags = jnp.arange(1, num_lags + 1)

    @jit
    def msd_fn(r1, r2):
        displacements = non_periodic_displacement(r1, r2)
        squared_distances = space.square_distance(displacements)
        msd = jnp.mean(squared_distances)
        return msd

    msds = []
    for lag in lags:
        r1 = positions[:-lag]
        r2 = positions[lag:]
        msd = msd_fn(r1, r2)
        msds.append(msd)

    return jnp.array(msds)


def self_diffusion(msd_values, dt, dim, start_idx=0, end_idx=None):
    """Computes the self-diffusion coefficient D by linear regression on
    mean squared displacement (MSD) values.

    start_idx and end_idx are useful to carve out the most representative part
    of MSD values to consider during regression. Most commonly, the error in
    MSD values is smallest in some "middle" range of lags.

    Args:
        msd_values:  values for linearly increasing time lags
        dt: Difference in time lag between adjacent MSD values
        dim: Dimension of the simulation
        start_idx: Starting index of MSD values are considered in the regression
        end_idx: End index of MSD values are considered in the regression

    Returns:
        Self-diffusion coefficient
    """
    n_max = msd_values.shape[0]
    time_lags = jnp.arange(1, n_max + 1) * dt
    considered_msds = msd_values[start_idx:end_idx]
    considered_lags = time_lags[start_idx:end_idx]
    assert len(considered_lags) > 0, ('No MSD values left for regression. '
                                      'Check start and end indices.')

    # solve MSD(i * dt) = a + (i * dt) b
    system_matrix = jnp.ones((considered_lags.size, 2))  # intercept
    system_matrix.at[ops.index[:, 1]].set(considered_lags)
    solution, residuals, _, _ = jnp.linalg.lstsq(system_matrix, considered_msds)
    b = solution[1]
    return b / (2 * dim)


if analyze_diffusion:
    subsample = 100  # only consider every 100th printout state of the trajectory as MSD time lag: Delta_t lag = 10 ps
    dimension = 3
    delta_t = print_every * subsample
    traj = traj_state[1]
    msds = mean_squared_displacements(traj, 20, subsample=subsample)
    # skip first 100 ps lags in regression as these can include ballistic trajectories biasing diffusion estimation
    diffusion = self_diffusion(msds, delta_t, dim=dimension, start_idx=0)
    lags = jnp.arange(1, msds.shape[0] + 1) * delta_t

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(lags, lags * diffusion * 2 * dimension, label='Regression Result', linestyle='--', color='k')
    plt.scatter(lags, msds, marker='x', label='Measured MSD')
    plt.xlabel(r'$\Delta t$ in ps')
    plt.ylabel(r'$MSD$ in $nm^2$')
    plt.legend()
    plt.savefig('Mean_squared_displacement.png')

    print('Diffusion in nm^2 / ps:', diffusion, 'In my m / ms:', diffusion * 1000)
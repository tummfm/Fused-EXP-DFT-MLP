import jax.numpy as np
from jax import lax
import pickle
import numpy as onp

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from scipy import interpolate as sci_interpolate
from chemtrain import traj_quantity


def box_density(R_snapshot, bin_edges, axis=0):
    # assumes all particles are wrapped into the same box
    profile, _ = np.histogram(R_snapshot[:, axis], bins=bin_edges)
    profile *= (profile.shape[0] / R_snapshot.shape[0]) # norm via n_bins and n_particles
    return profile


def get_bin_centers_from_edges(bin_edges):
    """To get centers from bin edges as generated from np.histogram"""
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    return bin_centers


def plot_density(file_name, n_bins=50):
    with open(file_name, 'rb') as f:
        R_traj_list, box = pickle.load(f)

    R_traj = R_traj_list[10]  # select one trajectory from all trajectories over optimization
    bin_edges = np.linspace(0., box[0], n_bins + 1)
    bin_centers = get_bin_centers_from_edges(bin_edges)
    compute_box_density = partial(box_density, bin_edges=bin_edges)
    density_snapshots = lax.map(compute_box_density, R_traj)
    density = np.mean(density_snapshots, axis=0)

    file_name = file_name[:-4]
    plt.figure()
    plt.plot(bin_centers, density)
    plt.ylabel('Normalizes Density')
    plt.xlabel('x')
    plt.savefig(file_name + '.png')


def visualize_time_series(file_name):
    with open(file_name, 'rb') as f:
        plot_dict = pickle.load(f)
    x_vals = plot_dict['x_vals']
    reference = plot_dict['reference']
    time_series = plot_dict['series']

    fig, ax = plt.subplots(figsize=(5, 3))
    series_line = ax.plot(x_vals, reference, label='Predicted')[0]
    ax.plot(x_vals, reference, label='Reference')
    ax.legend()
    # ax.set(xlim=(-3, 3), ylim=(-1, 1))

    def animate(i):
        series_line.set_ydata(time_series[i])
        ax.set_title('Epoche ' + str(i))

    file_name = file_name[:-4]
    anim = FuncAnimation(fig, animate, interval=200, frames=len(time_series) - 1)
    anim.save(file_name + '.gif', writer='imagemagick')


def plot_initial_and_predicted_rdf(rdf_bin_centers, g_average_final, model, visible_device, reference_rdf=None,
                                   g_average_init=None, after_pretraining=False):
    if after_pretraining:
        pretrain_str = '_after_pretrain'
    else:
        pretrain_str = ''

    plt.figure()
    plt.plot(rdf_bin_centers, g_average_final, label='predicted')
    if reference_rdf is not None:
        plt.plot(rdf_bin_centers, reference_rdf, label='reference')
    if g_average_init is not None:
        plt.plot(rdf_bin_centers, g_average_init, label='initial guess')
    plt.legend()
    plt.savefig('output/figures/difftre_predicted_RDF_' + model + str(visible_device) + pretrain_str + '.png')
    return

def plot_initial_and_predicted_adf(adf_bin_centers, predicted_adf_final, model, visible_device, reference_adf=None,
                                   adf_init=None, after_pretraining=False):
    if after_pretraining:
        pretrain_str = '_after_pretrain'
    else:
        pretrain_str = ''

    plt.figure()
    plt.plot(adf_bin_centers, predicted_adf_final, label='predicted')
    if reference_adf is not None:
        plt.plot(adf_bin_centers, reference_adf, label='reference')
    if adf_init is not None:
        plt.plot(adf_bin_centers, adf_init, label='initial guess')
    plt.legend()
    plt.savefig('output/figures/difftre_predicted_ADF_' + model + str(visible_device) + pretrain_str + '.png')
    return

def plot_pressure_history(pressure_history, model, visible_device, reference_pressure=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Pressure in kJ/ (mol nm^3)')
    ax1.plot(pressure_history, label='Predicted')
    if reference_pressure is not None:
        ax1.axhline(y=reference_pressure, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_pressure_history_' + model + str(visible_device) + '.png')
    return

def plot_density_history(density_history, model, visible_device, reference_density=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Density in g / l')
    ax1.plot(density_history, label='Predicted')
    if reference_density is not None:
        ax1.axhline(y=reference_density, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_density_history_' + model + str(visible_device) + '.png')
    return

def plot_alpha_history(prediction_history, model, visible_device, reference=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('alpha in 1 / K')
    ax1.plot(prediction_history, label='Predicted')
    if reference is not None:
        ax1.axhline(y=reference, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_alpha_history_' + model + str(visible_device) + '.png')
    return

def plot_kappa_history(prediction_history, model, visible_device, reference=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('kappa in 1 / (kJ / mol nm**3)')
    ax1.plot(prediction_history, label='Predicted')
    if reference is not None:
        ax1.axhline(y=reference, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_kappa_history_' + model + str(visible_device) + '.png')
    return

def plot_cp_history(prediction_history, model, visible_device, reference=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('c_p in kJ / mol K')
    ax1.plot(prediction_history, label='Predicted')
    if reference is not None:
        ax1.axhline(y=reference, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_cp_history_' + model + str(visible_device) + '.png')
    return

def plot_loss_and_gradient_history(loss_history, visible_device,
                                   gradient_history=None):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Loss')
    ax1.plot(loss_history, color=color, label='Loss')
    if gradient_history is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.semilogy(gradient_history, label='Gradient norm', color=color)
        ax2.set_ylabel('Gradient norm', color=color)  # we already handled the x-label with ax1
    plt.savefig('output/figures/difftre_train_history' + str(visible_device) + '.png')
    return


def plot_and_save_tabulated_potential(x_vals, params, init_params, model, visible_device):
    U_int = sci_interpolate.interp1d(x_vals, params, kind='cubic')
    spline_predicted = np.array(U_int(x_vals))
    result_array = onp.array([x_vals, spline_predicted]).T
    onp.savetxt('output/predicted_tabulated_potential.csv', result_array)

    plt.figure()
    plt.plot(x_vals, init_params, label='initial guess')
    plt.plot(x_vals, params, label='predicted table points')
    plt.plot(x_vals, spline_predicted, label='predicted spline')
    plt.ylim([-2.5, 5.])
    plt.legend()
    plt.savefig('output/figures/difftre_predicted_Potential_' + model + str(visible_device) + '.png')


def debug_npt_ensemble(init_traj_state, pressure_target, kbt, target_density,
                       mass):
    n_particles = init_traj_state.sim_state[0].position.shape[0]
    N = init_traj_state.aux['energy'].shape[0]
    plt.figure()
    plt.plot(init_traj_state.aux['energy'])
    plt.ylabel('Energy in kJ/mol')
    plt.savefig('NPT_energy_distribution.png')
    plt.figure()
    plt.plot(init_traj_state.aux['pressure'])
    plt.ylabel('Pressure in kJ/mol nm**3')
    plt.savefig('NPT_pressure_distribution.png')
    mean_pressure = np.mean(init_traj_state.aux['pressure'], axis=0)
    std_pressure = np.std(init_traj_state.aux['pressure'], axis=0)
    print(f'Mean pressure: {mean_pressure}. Target: {pressure_target}.'
          f'Statistical uncertainty = {std_pressure / np.sqrt(N)}')

    volume_traj = traj_quantity.volumes(init_traj_state)
    kappa = traj_quantity.isothermal_compressibility_npt(volume_traj, kbt)
    print(f'Isothermal compressibility {kappa} / kJ / mol nm**3  equals '
          f'{kappa * 16.6054} / bar')

    volumes = traj_quantity.volumes(init_traj_state)
    mean_volume = np.mean(volumes, axis=0)
    std_volume = np.std(volumes, axis=0)
    print(f'Mean volume = {mean_volume} nm**3. Statistical uncertainty = '
          f'{std_volume / np.sqrt(N)}\n '
          f'Mean density = {mass * n_particles / mean_volume}. '
          f'Target = {target_density}')

    plt.figure()
    plt.plot(volumes)
    plt.savefig('NPT_volume_distribution.png')


if __name__ == '__main__':

    # visualize_time_series('RDFs_Tabulated.pkl')
    plot_density('output/Trajectories/Traj_GNN3.pkl')

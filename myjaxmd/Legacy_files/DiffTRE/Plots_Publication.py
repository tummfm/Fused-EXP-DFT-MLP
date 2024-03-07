import numpy as np
import matplotlib.pyplot as plt
from chemtrain.jax_md_mod.custom_energy import generic_repulsion

def plot_prior_potential():
    r_vals = np.linspace(0.1, 0.6, 1000)
    eps = 1.
    sigma = 0.3165
    exp = 12
    u_vals = generic_repulsion(r_vals, sigma, eps, exp)
    plt.figure()
    plt.plot(r_vals, u_vals, color='#3c5488ff')
    plt.ylim([0, 10])
    plt.savefig('Figures/Publication/Prior_repulsion.eps')
    plt.savefig('Figures/Publication/Prior_repulsion.png')

if __name__ == '__main__':
    plot_prior_potential()

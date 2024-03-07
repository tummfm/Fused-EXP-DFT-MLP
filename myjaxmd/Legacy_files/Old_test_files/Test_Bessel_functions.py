import jax.numpy as np
import matplotlib.pyplot as plt



class SmoothingEnvelope():
    """A function that is 1 at 0 and has root of multiplicity 3 at 1 as defined in DimeNet.
    Apply on d/c [0, 1] to enable a smooth cut-off."""
    def __init__(self, p=6.):
        self.p = p

        self.a = -(p + 1.) * (p + 2.) / 2.
        self.b = p * (p + 2.)
        self.c = -p * (p + 1.) / 2.


    def __call__(self, inputs):
        envelope_val = 1. + self.a * inputs**self.p + self.b * inputs**(self.p + 1.) + self.c * inputs**(self.p + 2.)
        return np.where(inputs < 1., envelope_val, 0.)


class RadialBesselLayer():
    """A Layer that computes the Radial Bessel Function representation of pairwise distances"""

    def __init__(self, cutoff, num_radial=16, envelope_p=6, name='BesselRepresentation'):
        self.inv_cutoff = 1. / cutoff
        self.envelope = SmoothingEnvelope(p=envelope_p)
        self.num_radial = [num_radial]
        self.RBF_scale = np.sqrt(2. / cutoff)

    def __call__(self, distances):
        distances = np.expand_dims(distances, -1)  # to broadcast to num_radial
        scaled_distances = distances * self.inv_cutoff
        envelope_vals = self.envelope(scaled_distances)

        frequencies = np.pi * np.arange(1, self.num_radial[0] + 1, dtype=np.float32)
        RBF_vals = self.RBF_scale * np.sin(frequencies * scaled_distances) / distances
        return envelope_vals * RBF_vals

r_cutoff = 0.35

x_vals = np.linspace(0., 1., 100)
x_vals_inside_cut = np.linspace(0., r_cutoff, 100)

envelope = SmoothingEnvelope(p=6)
envelope_vals = envelope(x_vals)

bessel_representation = RadialBesselLayer(r_cutoff, num_radial=8, envelope_p=6)
RBF_edges = bessel_representation(x_vals_inside_cut)

plt.figure()
plt.plot(x_vals, envelope_vals, label='Envelope Function')
plt.legend()
plt.savefig('Figures/envelope.png')

plt.figure()
for i in range(RBF_edges.shape[1]):
    plt.plot(x_vals, RBF_edges[:, i], label='RBF_' + str(i))
plt.xlabel('d/c')
plt.legend()
plt.savefig('Figures/RBF_visualisation.png')

import numpy as onp
from scipy import interpolate as sci_interpolate

def convert_ADF_experimental_data():
    reference_ADF = onp.loadtxt('data/O_O_O_ADF_sin.csv')
    deg_vals_ADF = reference_ADF[:, 0]
    theta_ADF = deg_vals_ADF * (onp.pi / 180.)
    ADF_over_sin_theta = reference_ADF[:, 1]
    ADF = ADF_over_sin_theta * onp.sin(theta_ADF)

    # norm such that integral is 1
    integral = onp.trapz(ADF, theta_ADF)
    ADF /= integral

    result_array = onp.array([theta_ADF, ADF]).T
    onp.savetxt('data/O_O_O_ADF.csv', result_array)
    return

def equalize_RDF_grid():
    nbins = 250
    r_end = 1.
    dx_bin = r_end / float(nbins)
    rdf_bin_centers = onp.linspace(dx_bin / 2., r_end - dx_bin / 2., nbins)

    rdf_raw_data = onp.loadtxt('data/O_O_RDF_raw.csv')
    rdf_spline = sci_interpolate.interp1d(rdf_raw_data[:, 0], rdf_raw_data[:, 1], kind='cubic')
    rdf_fine_data = rdf_spline(rdf_bin_centers)

    result_array = onp.array([rdf_bin_centers, rdf_fine_data]).T
    onp.savetxt('data/O_O_RDF.csv', result_array)
    return

if __name__ == '__main__':
    convert_ADF_experimental_data()
    equalize_RDF_grid()


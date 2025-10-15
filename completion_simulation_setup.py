import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import product
import pandas as pd
from astropy.table import Table

def ABMag_to_fnu(mag):

    fnu = 10**((mag + 48.6)/-2.5)

    return fnu

def one_sigma_error():

    errors = np.array([3.07609681e-30, 1.88799135e-31, 3.56451133e-31, 2.53512863e-31,
                        1.87068214e-30, 5.49540874e-31, 2.08929613e-30, 1.95884467e-30,
                        2.16770410e-30, 8.55066713e-33, 8.31763771e-33])
    return errors 

def Muv_to_F_lambda_cgs(M1500, z):
    """
    Convert observed m_1500 to M_1500 using luminosity distance and redshift.
    
    Parameters:
    - m1500 : float or array
        Apparent magnitude at 1500 Å
    - z : float or array
        Redshift

    Returns:
    - M1500 : float or array
        Absolute magnitude at 1500 Å
    """
    DL = cosmo.luminosity_distance(z).to(u.pc).value  # in parsecs
    #m1500 = -2.5*np.log10(f_nu/3631) #f_nu in Jy
    #M1500 = m1500 - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + z)

    f_nu_jy = 3631 * 10**((M1500 + 5 * np.log10(DL / 10) -  2.5 * np.log10(1 + z))/ -2.5 )

    f_nu_cgs = f_nu_jy * 1e-23  # Convert from Jy to cgs

    f_lambda_cgs = f_nu_cgs * (3e18 / (1500**2))  # Convert to F_lambda in cgs

    return f_lambda_cgs


def get_spectra_flambda(beta_uv, beta_opt, wav_break=3500, wav_rest_range=(912, 40000), 
                           norm_wav=1500):

    # Wavelength grid (rest-frame)
    wav_rest = np.linspace(wav_rest_range[0], wav_rest_range[1], 2000)  # Å

    # Power-law spectrum
    flux_lambda_rest = np.where(
        wav_rest <= wav_break,
        (wav_rest / norm_wav)**beta_uv,
        (wav_break / norm_wav)**beta_uv * (wav_rest / wav_break)**beta_opt
    )

    return flux_lambda_rest, wav_rest

def plot_restframe_spectra(wav, flux):

    fig, ax = plt.subplots()
    ax.plot(wav, flux)
    ax.set_xscale('log')
    return fig, ax

def generate_mock_spectrum(Muv, redshift, beta_uv, beta_opt, 
                           wav_break=3500, wav_rest_range=(912, 40000), 
                           norm_wav=1500):
    """
    Generate a mock rest-frame + redshifted spectrum using UV and optical slopes.
    """
    # Wavelength grid (rest-frame)
    wav_rest = np.linspace(wav_rest_range[0], wav_rest_range[1], 2000)  # Å

    # Power-law spectrum
    flux_rest = np.where(
        wav_rest <= wav_break,
        (wav_rest / norm_wav)**beta_uv,
        (wav_break / norm_wav)**beta_uv * (wav_rest / wav_break)**beta_opt
    )

    #flux_rest = (wav_rest**2 / 3e18) * flux_lambda_rest  # Convert to F_nu

    # Normalize to Muv
    norm_flux_flambda =  Muv_to_F_lambda_cgs(Muv, redshift) 


    flux_rest *= norm_flux_flambda

    flux_rest_fnu = flux_rest * (wav_rest**2 / 3e18)  # Convert to F_nu

    # Redshift to observed frame
    wav_obs = wav_rest * (1 + redshift)
    flux_obs = flux_rest_fnu / (1 + redshift)

    return wav_obs, flux_obs

def read_filters():

    base = 'Filters'

    filters = ['CFHT_MegaCam.u.dat', 'Subaru_HSC.g.dat', 'Subaru_HSC.r.dat',
               'Subaru_HSC.i.dat', 'Subaru_HSC.z.dat', 'Subaru_HSC.y.dat',
               'Euclid_NISP.Y.dat', 'Euclid_NISP.J.dat', 'Euclid_NISP.H.dat',
               'Spitzer_IRAC.I1.dat', 'Spitzer_IRAC.I2.dat']
    
    paths = [f'{base}/{f}' for f in filters]

    return paths

def read_filter_data(paths):
    """
    Read filter data from a file.
    
    Parameters:
    - filter_file : str
        Path to the filter file.
    
    Returns:
    - wav : np.ndarray
        Wavelengths in Angstrom

    """

    transmission_curves = [np.loadtxt(path) for path in paths] 

    return transmission_curves

def map_transmission_curves_to_grid(wave_low, wave_high, transmission_curves):

    """
    Map transmission curves to a wavelength grid.
    
    Parameters:
    - wave_low : float
        Lower wavelength limit in Angstroms.
    - wave_high : float
        Upper wavelength limit in Angstroms.
    
    Returns:
    - wav_grid : np.ndarray
        Wavelength grid in Angstroms.
    """
    
    wav_grid = np.linspace(wave_low, wave_high, 10000)  # 1000 points between low and high
    all_filters = []

    for data in transmission_curves:
        wav = data[:, 0]
        transmission = data[:, 1]
        interp = np.interp(wav_grid, wav, transmission, left=0, right=0)
        all_filters.append(interp)
    
    return all_filters, wav_grid

filter_centers = np.array([3665.89, 4850.98, 6265.24, 7767.72, 8917.86, 9789.78, 10866.24, 13766.64, 17825.15, 35572.59, 45049.28])

def compute_fluxes(filter_grid, interp_fluxes, grid_wav):

    fluxes = []
    for transmission in filter_grid:  # shape (n_filters, n_wav)
        numerator = np.trapezoid(interp_fluxes * transmission, grid_wav)
        denominator = np.trapezoid(transmission, grid_wav)
        f = numerator/denominator
        fluxes.append(f)
    return np.array(fluxes)

def compute_interpolated_fluxes(Muv, z, beta_uv, beta_opt, grid_wav):
    wav_obs, flux_obs = generate_mock_spectrum(Muv, z, beta_uv, beta_opt)
    interp_fluxes = np.interp(grid_wav, wav_obs, flux_obs, left = 0, right = 0)
    return interp_fluxes

def photometry(Muv, z, beta_uv, beta_opt, filter_trans, grid_wav, plot = None):
    
    flux = compute_interpolated_fluxes(Muv, z, beta_uv, beta_opt, grid_wav)
    photom = compute_fluxes(filter_trans, flux, grid_wav)
    if plot:

        filter_centers = np.array([3665.89, 4850.98, 6265.24, 7767.72, 8917.86, 9789.78, 10866.24, 13766.64, 17825.15, 35572.59, 45049.28])
        plt.plot(grid_wav, flux)
        plt.scatter(filter_centers, photom)
        for filt in filter_trans:
            plt.plot(grid_wav, filt/np.amax(filt) * np.mean(flux), alpha=0.5)
        plt.show()
    return np.array(photom)


def procedure(Muv, z, beta_uv, beta_opt, plot = None):
    paths = read_filters()
    transmission_curves = read_filter_data(paths)
    filter_trans, grid_wav = map_transmission_curves_to_grid(1000, 60000, transmission_curves)
    photom = photometry(Muv, z, beta_uv, beta_opt, filter_trans, grid_wav, plot = plot)
    return photom


def perturb_photometry(photom, errors, num_realization):


    perturb = np.random.normal(0, scale=errors, size = (num_realization, len(errors)))

    perturb_photom = photom+perturb

    return perturb_photom

def make_photom_cat(perturb_photom, errors, flux_cols, fluxerr_cols):
    cols = ['id',
            'cfht_u_flux_ujy',
            'cfht_u_flux_ujy_err',
            'hsc_g_flux_ujy',
            'hsc_g_flux_ujy_err',
            'hsc_r_flux_ujy',
            'hsc_r_flux_ujy_err',
            'hsc_i_flux_ujy',
            'hsc_i_flux_ujy_err',
            'hsc_z_flux_ujy',
            'hsc_z_flux_ujy_err',
            'hsc_y_flux_ujy',
            'hsc_y_flux_ujy_err',
            'nisp_y_flux_ujy',
            'nisp_y_flux_ujy_err',
            'nisp_j_flux_ujy',
            'nisp_j_flux_ujy_err',
            'nisp_h_flux_ujy',
            'nisp_h_flux_ujy_err',
            'irac_ch1_flux_ujy',
            'irac_ch1_flux_ujy_err',
            'irac_ch2_flux_ujy',
            'irac_ch2_flux_ujy_err']
    
    DF = pd.DataFrame(perturb_photom, columns = flux_cols)

    for i, val in enumerate(errors):
        DF[fluxerr_cols[i]] = val

    DF['id'] = np.arange(1, DF.shape[0] + 1)

    return Table.from_pandas(DF[cols])

def run_procedure(args):
    Muv, z, beta_uv, beta_opt = args
    return procedure(Muv, z, beta_uv, beta_opt, plot=False)

if __name__ == "__main__":
    import h5py

    np.random.seed(42)

    uniform_Muv = np.random.uniform(-22, -18, 50)
    uniform_z = np.random.uniform(1.5, 4, 50)
    uniform_beta_uv = np.random.uniform(-2.8, -0.37, 50)
    uniform_beta_opt = np.random.uniform(0, 3, 50)


    # Generate all combinations
    param_grid = list(product(uniform_Muv, uniform_z, uniform_beta_uv, uniform_beta_opt))

    results = []

    with Pool(5) as pool:
        results = list(tqdm(pool.imap(run_procedure, param_grid), 
                            total=len(param_grid), 
                            desc="Running procedure"))
        
    results = np.array(results)
    param_array = np.array(param_grid)

    with h5py.File('photometry_output.h5', 'w') as f:
        f.create_dataset('photometry', data=results, compression='gzip')
        f.create_dataset('param_combinations', data=param_array)


# # Example usage
# Muv = -20


# z = 3
# beta_uv = -2
# beta_opt = 0.5

# wav, flux = generate_mock_spectrum(Muv, z, beta_uv, beta_opt)

# plt.plot(wav, flux)
# plt.xlabel("Observed Wavelength [Å]")
# plt.ylabel("Flux [erg/s/cm²/Å]")
# plt.title(f"Mock Spectrum (Muv={Muv}, z={z}, βuv={beta_uv}, βopt={beta_opt})")
# plt.grid()
# plt.xlim(3000, 20000)
# plt.yscale("log")
# plt.show()

# import h5py
# import pandas as pd

# with h5py.File("photometry_output.h5", "r") as f:
#     grp = f["perturbed_photometry"]
#     data = grp['data'][:]
#     columns = grp['columns'][:].astype(str)
#     index = grp['index'][:].astype(int)
#     #pd.DataFrame(group[:], columns = group.columns, index = group.index)
#     #data_array = group[:]

# df_reconstructed = pd.DataFrame(data=data, columns=columns, index=index)
# df_reconstructed['ID'] = df_reconstructed.ID.astype(int)


#     probabilities = [],
#     bins = np.array([[1.5,4]]),
#     binnames = ['1.5 to 4'],
#     for i in bins:
#         idx = np.where( (x >= i[0]) & (x <= i[1]) )[0]
#         x_subset = x[idx]
#         y_subset = y[idx]
#         probabilities.append(np.trapz(y_subset,x_subset))
#     binindex = probabilities.index(max(probabilities))
#     redshift = binnames[binindex]



# from astropy.table import Table
# import pandas as pd
# from scipy.integrate import trapezoid
# zarr = pd.read_csv('z_arr_TONE.csv')
# pzarr = Table.read('LRD_SIM_Perturbed_Photometry_PZ_Arrays.fits.gz')

# zmask = (zarr > 1.5) & (zarr < 4)


# ID = 97554
# row = pzarr[pzarr['ID'] == ID]
# y = row['Pz'].value


# def integrate_pz(pzarr, ID):
#     row = pzarr[pzarr['ID'] == ID]
#     y = row['Pz'].value[0]
#     z = zarr[zmask]  
#     return trapezoid(y = y[zmask], x = z)


# def integrate_pz_index(pzarr, index):
#     row = pzarr[index]
#     y = row['Pz']
#     z = zarr[zmask]  
#     return trapezoid(y = y[zmask], x = z)

from astropy.table import Table
import pandas as pd
from scipy.integrate import trapezoid
zarr = pd.read_csv('z_arr_TONE.csv')
pzarr = Table.read('LRD_SIM_Perturbed_Photometry_PZ_Arrays.fits.gz')
zarr = zarr['z_arr'].values
zmask = (zarr > 1.5) & (zarr < 4)

def integrate_pz_index(pzarr, index):
    row = pzarr[index]
    y = row['Pz'].value[0]
    z = zarr[zmask]  
    return trapezoid(y = y[zmask], x = z)

oscars_int = []
for i in range(len(pzarr)):
    oscars_int.append(integrate_pz_index(pzarr, i))

def hannah_pz(pz_tab, idx):
    x = zarr
    y = pzarr[idx]['Pz']
    
    probabilities = []
    bins = np.array([[1.5,4]])
    binnames = ['1.5 to 4']
    for i in bins:
        idx = np.where( (x >= i[0]) & (x <= i[1]) )[0]
        x_subset = x[idx]
        y_subset = y[idx]
        probabilities.append(np.trapz(y_subset,x_subset))
    binindex = probabilities.index(max(probabilities))
    redshift = binnames[binindex]
    return redshift, binindex, probabilities[0]

hannah_ints = []
for i in range(len(pzarr)):
    hannah_ints.append(hannah_pz(pzarr, i)[-1])
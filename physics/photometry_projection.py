import numpy as np


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


def compute_fluxes(filter_grid, interp_fluxes, grid_wav):

    fluxes = []
    for transmission in filter_grid:  # shape (n_filters, n_wav)
        numerator = np.trapezoid(interp_fluxes * transmission, grid_wav)
        denominator = np.trapezoid(transmission, grid_wav)
        f = numerator/denominator
        fluxes.append(f)
    return np.array(fluxes)

def photometry(Muv, z, beta_uv, beta_opt, filter_trans, grid_wav, plot = None):
    
    flux = compute_interpolated_fluxes(Muv, z, beta_uv, beta_opt, grid_wav)
    photom = compute_fluxes(filter_trans, flux, grid_wav)
    return np.array(photom)
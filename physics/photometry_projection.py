import numpy as np

def map_transmission_curves_to_grid(filter_wave_grid, transmission_curves):

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
    
    all_filters = []

    for data in transmission_curves:
        wav = data[:, 0]
        transmission = data[:, 1]
        interp = np.interp(filter_wave_grid, wav, transmission, left=0, right=0)
        all_filters.append(interp)
    
    return all_filters


def compute_fluxes(filter_grid, interp_fluxes, grid_wav):

    fluxes = []
    for transmission in filter_grid:  # shape (n_filters, n_wav)
        numerator = np.trapz(interp_fluxes * transmission* grid_wav, grid_wav)
        denominator = np.trapz(transmission*grid_wav, grid_wav)
        f = numerator/denominator
        fluxes.append(f)
    return np.array(fluxes)

def photometry(flux, filter_trans, grid_wav, plot = None):
    
    photom = compute_fluxes(filter_trans, flux, grid_wav)
    return photom

def perturb_photometry(photom, errors, num_realization):


    perturb = np.random.normal(0, scale=errors, size = (num_realization, len(errors)))

    perturb_photom = photom+perturb

    return perturb_photom
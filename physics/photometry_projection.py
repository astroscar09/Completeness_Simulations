import numpy as np

def map_transmission_curves_to_grid(filter_wave_grid, transmission_curves):
    """
    Interpolate a list of filter transmission curves onto a common
    wavelength grid.

    Parameters
    ----------
    filter_wave_grid : array-like
        Wavelength grid where the transmissions should be evaluated.
    transmission_curves : list of ndarray
        Each element should be an ``(N,2)`` array with wavelength in
        the first column and transmission in the second.

    Returns
    -------
    all_filters : list of ndarray
        Interpolated transmission values for each filter on
        ``filter_wave_grid``.
    """
    
    all_filters = []

    for data in transmission_curves:
        wav = data[:, 0]
        transmission = data[:, 1]
        interp = np.interp(filter_wave_grid, wav, transmission, left=0, right=0)
        all_filters.append(interp)
    
    return all_filters


def compute_fluxes(filter_grid, interp_fluxes, grid_wav):
    """
    Compute photometric fluxes by integrating a spectrum over filter
    transmission curves.

    Parameters
    ----------
    filter_grid : sequence of array-like
        Transmission curves sampled on ``grid_wav`` (one per filter).
    interp_fluxes : array-like
        Spectral flux values sampled on ``grid_wav``.
    grid_wav : array-like
        Wavelength grid corresponding to ``interp_fluxes`` and
        ``filter_grid``.

    Returns
    -------
    fluxes : np.ndarray
        Computed flux for each filter.
    """

    fluxes = []
    for transmission in filter_grid:  # shape (n_filters, n_wav)
        numerator = np.trapz(interp_fluxes * transmission* grid_wav, grid_wav)
        denominator = np.trapz(transmission*grid_wav, grid_wav)
        f = numerator/denominator
        fluxes.append(f)
    return np.array(fluxes)

def photometry(flux, filter_trans, grid_wav, plot = None):
    """
    Convenience wrapper around ``compute_fluxes`` for obtaining photometry.

    Parameters
    ----------
    flux : array-like
        Spectrum values on ``grid_wav``.
    filter_trans : sequence
        List of transmission curves on ``grid_wav``.
    grid_wav : array-like
        Common wavelength grid for ``flux`` and ``filter_trans``.
    plot : ignored
        Placeholder parameter (not currently used).

    Returns
    -------
    photom : np.ndarray
        Fluxes computed for each filter.
    """
    
    photom = compute_fluxes(filter_trans, flux, grid_wav)
    return photom

def perturb_photometry(photom, errors, num_realization):
    """Add Gaussian noise to photometry.

    Parameters
    ----------
    photom : array-like
        Photometric fluxes.
    errors : array-like
        Flux uncertainties.
    num_realization : int
        Number of noisy realizations.
    """

    perturb = np.random.normal(0, scale=errors, size = (num_realization, len(errors)))

    perturb_photom = photom+perturb

    return perturb_photom
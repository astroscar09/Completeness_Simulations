import numpy as np
from .cosmology_scaling import *


def generate_restframe_spectrum(
                                    beta_uv: float,
                                    beta_opt: float,
                                    wav_break: float,
                                    norm_wav: float,
                                    wav_range: tuple, 
                                    wav_grid_points: int,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a broken power‑law spectrum in the rest frame.

    Parameters
    ----------
    beta_uv : float
        Spectral slope shortward of ``wav_break``.
    beta_opt : float
        Spectral slope longward of ``wav_break``.
    wav_break : float
        Break wavelength (Å).
    norm_wav : float
        Normalization wavelength (Å) used in the power law.
    wav_range : tuple
        (min, max) rest‑frame wavelength range (Å).
    wav_grid_points : int
        Number of grid points to sample between ``wav_range``.

    Returns
    -------
    flux_lambda_rest : ndarray
        Flux density per unit wavelength on the rest frame grid.
    wav_rest : ndarray
        Wavelength grid (Å) corresponding to ``flux_lambda_rest``.
    """

    # Wavelength grid (rest-frame)
    wav_rest = np.linspace(wav_range[0], wav_range[1], wav_grid_points)  # Å

    # Power-law spectrum
    flux_lambda_rest = np.where(
        wav_rest <= wav_break,
        (wav_rest / norm_wav)**beta_uv,
        (wav_break / norm_wav)**beta_uv * (wav_rest / wav_break)**beta_opt
    )

    return flux_lambda_rest, wav_rest


def generate_mock_spectrum(Muv, redshift, beta_uv, beta_opt, wav_rest,
                           wav_break=3500,
                           norm_wav=1500):
    """
    Generate a mock rest-frame + redshifted spectrum using UV and optical slopes.
    """
    # Wavelength grid (rest-frame)
    # wav_rest = np.linspace(wav_rest_range[0], wav_rest_range[1], wav_grid_points)  # Å

    # Power-law spectrum
    flux_rest = np.where(
        wav_rest <= wav_break,
        (wav_rest / norm_wav)**beta_uv,
        (wav_break / norm_wav)**beta_uv * (wav_rest / wav_break)**beta_opt
    )

    # Normalize to Muv
    norm_flux_flambda =  Muv_to_F_lambda_cgs(Muv, redshift) 

    flux_rest *= norm_flux_flambda

    # Redshift to observed frame
    wav_obs = wav_rest * (1 + redshift)
    flux_obs_flam = flux_rest / (1 + redshift)

    return wav_obs, flux_obs_flam

def compute_interpolated_spectra(wav_obs, flux_obs, filter_grid_wav):
    
    interp_fluxes = np.interp(filter_grid_wav, wav_obs, flux_obs, left = 0, right = 0)
    
    return interp_fluxes
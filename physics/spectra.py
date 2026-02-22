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

    # Wavelength grid (rest-frame)
    wav_rest = np.linspace(wav_range[0], wav_range[1], wav_grid_points)  # Å

    # Power-law spectrum
    flux_lambda_rest = np.where(
        wav_rest <= wav_break,
        (wav_rest / norm_wav)**beta_uv,
        (wav_break / norm_wav)**beta_uv * (wav_rest / wav_break)**beta_opt
    )

    return flux_lambda_rest, wav_rest


def generate_mock_spectrum(Muv, redshift, beta_uv, beta_opt, 
                           wav_break=3500, wav_rest_range=(912, 40000), 
                           wav_grid_points = 2000,
                           norm_wav=1500):
    """
    Generate a mock rest-frame + redshifted spectrum using UV and optical slopes.
    """
    # Wavelength grid (rest-frame)
    wav_rest = np.linspace(wav_rest_range[0], wav_rest_range[1], wav_grid_points)  # Å

    # Power-law spectrum
    flux_rest = np.where(
        wav_rest <= wav_break,
        (wav_rest / norm_wav)**beta_uv,
        (wav_break / norm_wav)**beta_uv * (wav_rest / wav_break)**beta_opt
    )

    # Normalize to Muv
    norm_flux_flambda =  Muv_to_F_lambda_cgs(Muv, redshift) 

    flux_rest *= norm_flux_flambda

    flux_rest_fnu = flux_rest * (wav_rest**2 / 3e18)  # Convert to F_nu

    # Redshift to observed frame
    wav_obs = wav_rest * (1 + redshift)
    flux_obs = flux_rest_fnu / (1 + redshift)

    return wav_obs, flux_obs
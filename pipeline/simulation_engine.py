from physics.spectra import generate_mock_spectrum, compute_interpolated_spectra
from physics.photometry_projection import compute_fluxes
from functools import partial
from itertools import product

def generate_param_combinations(Muv, z, beta_uv, beta_opt):

    param_grid = list(product(Muv, z, beta_uv, beta_opt))

    return param_grid

def run_single_simulation(
    params,
    spec_wav_grid,
    wave_break,
    norm_wave, 
    filter_wave_grid, 
    all_filters   
):
    """
    Execute one forward-model simulation.
    """
    Muv, z, beta_uv, beta_opt = params

    wav_obs, flux_obs_flam = generate_mock_spectrum(Muv, z, beta_uv, beta_opt, 
                                                    spec_wav_grid,
                                                    wave_break, 
                                                    norm_wave)
    
    interp_fluxes    = compute_interpolated_spectra(wav_obs, flux_obs_flam, filter_wave_grid)

    phot_flam = compute_fluxes(all_filters, interp_fluxes, filter_wave_grid)

    return phot_flam #wav_obs, flux_obs_flam, 

def make_worker_function(   spec_wav_grid,
                            wave_break,
                            norm_wave, 
                            filter_wave_grid, 
                            all_filters):

    worker = partial(
                        run_single_simulation,
                        spec_wav_grid=spec_wav_grid,
                        wave_break=wave_break,
                        norm_wave=norm_wave,
                        filter_wave_grid=filter_wave_grid,
                        all_filters=all_filters
                    )
    
    return worker
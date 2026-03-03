from io_utils.filter_reader import read_filters, read_filter_data
from io_utils.yaml_handler import load_config
from physics.spectra import generate_mock_spectrum, compute_interpolated_spectra
from physics.photometry_projection import map_transmission_curves_to_grid, compute_fluxes
from physics.data_processesing import perturb_photometry, make_photom_cat, convert_flambda_to_fnu, convert_fnu_to_microjansky, Fnu_to_ABMag
from pipeline.experiment_launcher import run_experiment_sweep
from pipeline.simulation_engine import run_single_simulation
import numpy as np
from plotting.plotting_utils import plot_setup, plot_spectrum

cfg = load_config("config/simulation.yaml")

spec_wav_grid = np.linspace(
    cfg["spectrum"]["wav_rest_min"],
    cfg["spectrum"]["wav_rest_max"],
    cfg["spectrum"]["wav_grid_points"]
)

wave_break = cfg['spectrum']['wav_break']
norm_wave = cfg['spectrum']['norm_wav']

paths, filter_flux_names, filt_centers = read_filters()
transmission_curves = read_filter_data(paths)

wave_min = cfg['filter_grid_bounds']['min']
wave_max = cfg['filter_grid_bounds']['max']
num_grids = cfg['filter_grid_bounds']['wav_grid_points']

filter_wave_grid = np.linspace(wave_min, wave_max, num_grids) 
all_filters      = map_transmission_curves_to_grid(filter_wave_grid, transmission_curves)

#Muv_min = cfg['parameter_grid']['Muv']['min']
#Muv_max = cfg['parameter_grid']['Muv']['max']
#Muv_spacing = cfg['parameter_grid']['Muv']['spacing']
#Muv = np.arange(Muv_min, Muv_max+Muv_spacing, Muv_spacing)

#z_min = cfg['parameter_grid']['z']['min']
#z_max = cfg['parameter_grid']['z']['max']
#z_spacing = cfg['parameter_grid']['z']['spacing']
#z = np.arange(z_min, z_max+z_spacing, z_spacing)

#beta_uv_min = cfg['parameter_grid']['beta_uv']['min']
#beta_uv_max = cfg['parameter_grid']['beta_uv']['max']
#beta_uv_spacing = cfg['parameter_grid']['beta_uv']['spacing']
#beta_uv = np.arange(beta_uv_min, beta_uv_max+beta_uv_spacing, beta_uv_spacing)

#beta_opt_min = cfg['parameter_grid']['beta_opt']['min']
#beta_opt_max = cfg['parameter_grid']['beta_opt']['max']
#beta_opt_spacing = cfg['parameter_grid']['beta_opt']['spacing']
#beta_opt = np.arange(beta_opt_min, beta_opt_max+beta_opt_spacing, beta_opt_spacing)

Muv = -18.9 #
redshift = 2 

beta_uv = -1.2227562
beta_opt = 0.02761



wav_obs, flux_obs_flam, phot_flam = run_single_simulation((Muv,
                                                            redshift,
                                                            beta_uv,
                                                            beta_opt),
                                                            spec_wav_grid,
                                                            wave_break,
                                                            norm_wave, 
                                                            filter_wave_grid, 
                                                            all_filters)

phot_fnu = phot_flam * (filt_centers**2 / 3e18)

fig, ax = plot_setup()
plot_spectrum(wav_obs, flux_obs_flam, ax, show = True)

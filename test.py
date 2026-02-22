from io_utils.filter_reader import read_filters, read_filter_data
from io_utils.yaml_handler import load_config
from physics.spectra import generate_restframe_spectrum, generate_mock_spectrum
from physics.photometry_projection import photometry
from pipeline.experiment_launcher import run_experiment_sweep
from pipeline.simulation_engine import run_single_simulation
import numpy as np
import matplotlib.pyplot as plt


cfg = load_config("config/simulation.yaml")

wav_grid = np.linspace(
    cfg["spectrum"]["wav_rest_min"],
    cfg["spectrum"]["wav_rest_max"],
    cfg["spectrum"]["wav_grid_points"]
)

wave_min = cfg['spectrum']['wav_rest_min']
wave_max = cfg['spectrum']['wav_rest_max']
wav_grid_points = cfg['spectrum']['wav_grid_points']
wave_break = cfg['spectrum']['wav_break']
norm_wave = cfg['spectrum']['norm_wav']

Muv = -21
redshift = 2.5 #cfg['spectrum']['z'][]

beta_uv = cfg['parameter_grid']['beta_uv']['min']
beta_opt = cfg['parameter_grid']['beta_opt']['min']

flux_lambda_rest, wav_rest = generate_restframe_spectrum(beta_uv,
                                                         beta_opt,
                                                         wave_break,
                                                         norm_wave,
                                                         (wave_min, wave_max), 
                                                          wav_grid_points)


wav_obs, flux_obs = generate_mock_spectrum(Muv, redshift, beta_uv, beta_opt, 
                                            wave_break, (wave_min, wave_max), 
                                            (wav_grid_points),
                                            norm_wave)

plt.figure(figsize = (10, 5))
plt.loglog(wav_rest, flux_lambda_rest)
plt.loglog(wav_obs, flux_obs)
#plt.xscale('log')
plt.show()




from io_utils.filter_reader import read_filters, read_filter_data
from io_utils.yaml_handler import load_config
from physics.photometry_projection import map_transmission_curves_to_grid
from physics.data_processesing import perturb_photometry, make_photom_cat, convert_flambda_to_fnu, convert_fnu_to_microjansky, make_param_table, merge_tables_horizontally
from pipeline.experiment_launcher import run_experiment_sweep
from pipeline.simulation_engine import generate_param_combinations, make_worker_function
import numpy as np




def main():

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

    Muv_min = cfg['parameter_grid']['Muv']['min']
    Muv_max = cfg['parameter_grid']['Muv']['max']
    Muv_spacing = cfg['parameter_grid']['Muv']['spacing']
    Muv = np.arange(Muv_min, Muv_max+Muv_spacing, Muv_spacing)

    z_min = cfg['parameter_grid']['z']['min']
    z_max = cfg['parameter_grid']['z']['max']
    z_spacing = cfg['parameter_grid']['z']['spacing']
    z = np.arange(z_min, z_max+z_spacing, z_spacing)

    beta_uv_min = cfg['parameter_grid']['beta_uv']['min']
    beta_uv_max = cfg['parameter_grid']['beta_uv']['max']
    beta_uv_spacing = cfg['parameter_grid']['beta_uv']['spacing']
    beta_uv = np.arange(beta_uv_min, beta_uv_max+beta_uv_spacing, beta_uv_spacing)

    beta_opt_min = cfg['parameter_grid']['beta_opt']['min']
    beta_opt_max = cfg['parameter_grid']['beta_opt']['max']
    beta_opt_spacing = cfg['parameter_grid']['beta_opt']['spacing']
    beta_opt = np.arange(beta_opt_min, beta_opt_max+beta_opt_spacing, beta_opt_spacing)

    param_grid = np.array(generate_param_combinations(Muv, z, beta_uv, beta_opt))

    param_tab = make_param_table(param_grid)

    num_realizations = len(param_grid)

    worker = make_worker_function(   spec_wav_grid,
                                    wave_break,
                                    norm_wave, 
                                    filter_wave_grid, 
                                    all_filters)

    ncores = cfg['num_cores']
    results = run_experiment_sweep(param_grid, worker, ncores)

    photom_fnu = convert_flambda_to_fnu(np.array(results), filt_centers)

    photom_errors = np.array(cfg['photometry_errors'])

    perturb_phot_fnu = perturb_photometry(photom_fnu, photom_errors, num_realizations)
    
    perturb_phot_muJy = convert_fnu_to_microjansky(perturb_phot_fnu)
    errors_muJy = convert_fnu_to_microjansky(photom_errors)

    error_cols = [f'{x}_err' for x in filter_flux_names]
    
    phot_cat = make_photom_cat(perturb_phot_muJy, errors_muJy, filter_flux_names, error_cols)

    final_phot_cat = merge_tables_horizontally(phot_cat, param_tab)

    return final_phot_cat
    




if __name__ == '__main__':
    main()


# Muv = -18.9 #
# redshift = 2 

# beta_uv = -1.2227562
# beta_opt = 0.02761



# wav_obs, flux_obs_flam, phot_flam = run_single_simulation((Muv,
#                                                             redshift,
#                                                             beta_uv,
#                                                             beta_opt),
#                                                             spec_wav_grid,
#                                                             wave_break,
#                                                             norm_wave, 
#                                                             filter_wave_grid, 
#                                                             all_filters)

# phot_fnu = phot_flam * (filt_centers**2 / 3e18)


# fig, ax = plot_setup()
# plot_spectrum(wav_obs, flux_obs_flam, ax, show = True)

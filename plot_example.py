"""
Standalone demo: generate a single mock spectrum and visualize the full
forward model pipeline (spectrum + noiseless photometry + noisy realization
+ filter transmission curves).

Run from the repo root:
    python plot_example.py
"""

import numpy as np

from io_utils.yaml_handler import load_config
from io_utils.filter_reader import read_filters, read_filter_data
from physics.spectra import generate_mock_spectrum, compute_interpolated_spectra
from physics.photometry_projection import map_transmission_curves_to_grid, compute_fluxes
from physics.data_processesing import (
    convert_flambda_to_fnu,
    convert_fnu_to_microjansky,
)
from plotting.plotting_utils import plot_example_spectrum

# ---------------------------------------------------------------------------
# 1. Load config
# ---------------------------------------------------------------------------
cfg = load_config("config/test.yaml")

spec_cfg = cfg["spectrum"]
grid_cfg = cfg["filter_grid_bounds"]
errors_cgs = np.array(cfg["photometry_errors"])  # erg/s/cm^2/Hz

# ---------------------------------------------------------------------------
# 2. Example parameters
# ---------------------------------------------------------------------------
Muv = -16.0
z = 2.0
beta_uv = -1.5
beta_opt = 2.0

# ---------------------------------------------------------------------------
# 3. Build rest-frame wavelength grid and generate mock spectrum
# ---------------------------------------------------------------------------
wav_rest = np.linspace(spec_cfg["wav_rest_min"], spec_cfg["wav_rest_max"],
                       spec_cfg["wav_grid_points"])

wav_obs, flux_obs_flam = generate_mock_spectrum(
    Muv, z, beta_uv, beta_opt,
    wav_rest,
    wav_break=spec_cfg["wav_break"],
    norm_wav=spec_cfg["norm_wav"],
)

# Convert spectrum to F_nu µJy
flux_fnu_cgs = convert_flambda_to_fnu(flux_obs_flam, wav_obs)
flux_fnu_ujy = convert_fnu_to_microjansky(flux_fnu_cgs)

# ---------------------------------------------------------------------------
# 4. Load filters
# ---------------------------------------------------------------------------
paths, filter_names, filter_centers = read_filters()
transmission_curves = read_filter_data(paths)

# ---------------------------------------------------------------------------
# 5. Build filter wavelength grid and map transmission curves
# ---------------------------------------------------------------------------
filter_wave_grid = np.linspace(
    grid_cfg["min"], grid_cfg["max"], grid_cfg["wav_grid_points"]
)
all_filters = map_transmission_curves_to_grid(filter_wave_grid, transmission_curves)

# ---------------------------------------------------------------------------
# 6. Compute noiseless photometry
# ---------------------------------------------------------------------------
interp_fluxes = compute_interpolated_spectra(wav_obs, flux_obs_flam, filter_wave_grid)
phot_flam = compute_fluxes(all_filters, interp_fluxes, filter_wave_grid)
phot_fnu = convert_flambda_to_fnu(phot_flam, filter_centers)
phot_ujy = convert_fnu_to_microjansky(phot_fnu)

# ---------------------------------------------------------------------------
# 7. One noisy realization
# ---------------------------------------------------------------------------
errors_ujy = convert_fnu_to_microjansky(errors_cgs)
perturbed_ujy = phot_ujy + np.random.normal(0, errors_ujy)

# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------
plot_example_spectrum(
    wav_obs=wav_obs,
    flux_fnu_ujy=flux_fnu_ujy,
    filter_centers=filter_centers,
    phot_ujy=phot_ujy,
    perturbed_ujy=perturbed_ujy,
    errors_ujy=errors_ujy,
    filter_wave_grid=filter_wave_grid,
    all_filters=all_filters,
    filter_names=filter_names,
    redshift = z,
    savepath="example_spectrum.png",
)

print("Saved example_spectrum.png")

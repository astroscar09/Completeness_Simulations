# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This project generates synthetic galaxy photometry catalogs to assess survey completeness for LRD (Little Red Dot)-like galaxy searches. Mock spectra are simulated across a multidimensional grid of astrophysical parameters, projected through observational filter bandpasses, and noise-perturbed to produce realistic photometric catalogs. These catalogs are run through an external selection pipeline, after which `completeness_correction.py` computes recovery fractions.

## Running the Simulation

The primary entry point is `test_script.py`, which reads from `config/simulation.yaml`:

```bash
python test_script.py
```

To run a quick test with a coarser parameter grid, temporarily swap the config path in `test_script.py` to point at `config/test.yaml` (which uses larger grid spacings, e.g., Muv spacing 2.5 vs 0.2).

The legacy monolithic script (`completion_simulation_setup.py`) contains an older single-file version of the same workflow and is not the primary code path.

Parallelism is controlled by `num_cores` in the YAML config.

## Architecture

The simulation follows a linear pipeline:

1. **Config loading** (`io_utils/yaml_handler.py`) — reads `config/simulation.yaml` for parameter grid bounds/spacings, spectrum settings, photometry errors, and `num_cores`.

2. **Filter setup** (`io_utils/filter_reader.py`, `config/filters.yaml`) — loads 11 filter transmission curves from `Filters/` (CFHT u, HSC grizy, Euclid NISP YJH, Spitzer IRAC ch1/ch2) and interpolates them onto a common wavelength grid via `physics/photometry_projection.py`.

3. **Parameter grid** (`pipeline/simulation_engine.py`) — builds a Cartesian product over `Muv × z × beta_uv × beta_opt` via `generate_param_combinations`.

4. **Forward model** (per grid point, parallelized):
   - `physics/spectra.py` — generates a broken power-law spectrum in the rest frame (`generate_mock_spectrum`), normalized using `physics/cosmology_scaling.py::Muv_to_F_lambda_cgs` (Planck18 cosmology), then redshifts it to observed frame.
   - `physics/photometry_projection.py` — integrates the spectrum through each filter (`compute_fluxes`), returning F_lambda per filter.

5. **Parallelization** (`pipeline/experiment_launcher.py`) — `run_experiment_sweep` uses `multiprocessing.Pool` with `pool.imap`.

6. **Post-processing** (`physics/data_processesing.py`) — converts F_lambda → F_nu → µJy, adds Gaussian noise (`perturb_photometry`), and assembles an Astropy Table catalog (`make_photom_cat`, `merge_tables_horizontally`).

7. **Completeness analysis** (`completeness_correction.py`) — reads the simulation output catalog and external selection results (`simulation_results.csv`), merges them, and computes recovery fractions binned over the 4D parameter space into a `completeness_array`.

## Key Configuration Parameters

All in `config/simulation.yaml` (coarse test version in `config/test.yaml`):

- `parameter_grid`: min/max/spacing for Muv, z, beta_uv, beta_opt
- `spectrum`: wavelength grid bounds and break/normalization wavelengths (break at 3500 Å, norm at 1500 Å)
- `filter_grid_bounds`: common wavelength grid for filter projection (1000–60000 Å)
- `photometry_errors`: 11 per-filter 1σ noise values in cgs F_nu units
- `num_cores`: number of parallel workers

## Module Map

| Module | Responsibility |
|---|---|
| `physics/spectra.py` | Broken power-law spectrum generation and redshifting |
| `physics/cosmology_scaling.py` | `Muv_to_F_lambda_cgs` — absolute magnitude → flux normalization |
| `physics/photometry_projection.py` | Filter interpolation and flux integration |
| `physics/data_processesing.py` | Unit conversions, noise perturbation, catalog building |
| `pipeline/simulation_engine.py` | Parameter grid generation and worker function factory |
| `pipeline/experiment_launcher.py` | Multiprocessing pool execution |
| `io_utils/filter_reader.py` | Loads filter paths/names/centers from `config/filters.yaml` |
| `io_utils/yaml_handler.py` | Thin wrapper around `yaml.safe_load` |
| `io_utils/hdf5_handler.py` | HDF5 save/load utilities (not used in current main path) |
| `completeness_correction.py` | Post-simulation completeness fraction computation |
| `plotting/plotting_utils.py` | Visualization helpers; `plotting/style.mplstyle` for plot styling |

## Spectrum Model

Spectra are broken power-laws in F_lambda:
- For λ ≤ 3500 Å: `(λ/1500)^beta_uv`
- For λ > 3500 Å: `(3500/1500)^beta_uv × (λ/3500)^beta_opt`

Normalized at 1500 Å using the luminosity-distance-based conversion from M_UV to F_lambda (cgs). Output photometry is in µJy.

When running code here use 'conda activate research'

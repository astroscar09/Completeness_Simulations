import numpy as np
import pandas as pd
from astropy.table import Table, hstack
from itertools import product

def ABMag_to_fnu(mag):

    fnu = 10**((mag + 48.6)/-2.5)

    return fnu

def Fnu_to_ABMag(fnu):

    return -2.5*np.log10(fnu) - 48.6

def merge_tables_horizontally(tab1, tab2):
    return hstack((tab1, tab2))

def make_param_table(params):

    return Table(params, names = ['Muv', 'z', 'beta_uv', 'beta_opt'])


def perturb_photometry(photom, errors, num_realization):


    perturb = np.random.normal(0, scale=errors, size = (num_realization, len(errors)))

    perturb_photom = photom+perturb

    return perturb_photom

def make_photom_cat(perturb_photom, errors, flux_cols, fluxerr_cols):
    
    final_flux_cols = []

    for x, y in zip(flux_cols, fluxerr_cols):
        final_flux_cols.append(x)
        final_flux_cols.append(y)

    cols = ['id'] + final_flux_cols
    
    DF = pd.DataFrame(perturb_photom, columns = flux_cols)

    for i, val in enumerate(errors):
        DF[fluxerr_cols[i]] = val

    DF['id'] = np.arange(1, DF.shape[0] + 1)

    return Table.from_pandas(DF[cols])


def convert_fnu_to_microjansky(data):

    return data*1e29

def convert_flambda_to_fnu(data, wavelength):

    c_A_s = 3e18
    fnu_data = data * (wavelength**2/c_A_s)

    return fnu_data

def convert_m_to_beta(m):

    beta = -m/2.5 - 2

    return beta

def beta_function(wave, beta, C):

    return -2.5 * (beta + 2)*np.log10(wave) + C

def make_param_grid(Muv_grid, z_grid, beta_uv_grid, beta_opt_grid):
    param_grid = list(product(Muv_grid, z_grid, beta_uv_grid, beta_opt_grid))

    return param_grid

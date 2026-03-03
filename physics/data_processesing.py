import numpy as np
import pandas as pd
from astropy.table import Table, hstack
from itertools import product

def ABMag_to_fnu(mag):
    """
    Convert an AB magnitude to flux density $f_{\nu}$.

    Parameters
    ----------
    mag : float or array-like
        AB magnitude value(s).

    Returns
    -------
    fnu : float or ndarray
        Flux density corresponding to the input magnitude(s).
    """

    fnu = 10**((mag + 48.6)/-2.5)

    return fnu

def Fnu_to_ABMag(fnu):
    """
    Convert flux density $f_{\nu}$ back to an AB magnitude.

    Parameters
    ----------
    fnu : float or array-like
        Flux density value(s).

    Returns
    -------
    mag : float or ndarray
        Corresponding AB magnitude(s).
    """

    return -2.5*np.log10(fnu) - 48.6

def merge_tables_horizontally(tab1, tab2):
    """
    Horizontally concatenate two Astropy tables.

    Parameters
    ----------
    tab1, tab2 : astropy.table.Table
        Tables to be joined.

    Returns
    -------
    merged : astropy.table.Table
        Result of ``hstack((tab1, tab2))``.
    """
    return hstack((tab1, tab2))

def make_param_table(params):
    """
    Construct an Astropy ``Table`` from a sequence of parameter tuples.

    Parameters
    ----------
    params : sequence
        Iterable of tuples/lists containing values for ``Muv``, ``z``,
        ``beta_uv`` and ``beta_opt``.

    Returns
    -------
    table : astropy.table.Table
        Table with columns ``['Muv','z','beta_uv','beta_opt']``.
    """

    return Table(params, names = ['Muv', 'z', 'beta_uv', 'beta_opt'])


def perturb_photometry(photom, errors, num_realization):
    """
    Generate realizations of photometry by adding Gaussian noise.

    Parameters
    ----------
    photom : array-like
        Original photometric fluxes.
    errors : array-like
        1‑sigma uncertainties associated with ``photom``.
    num_realization : int
        Number of noisy realizations to create.

    Returns
    -------
    perturb_photom : ndarray
        Array of shape ``(num_realization, len(errors))`` containing the
        perturbed flux values.
    """

    perturb = np.random.normal(0, scale=errors, size = (num_realization, len(errors)))

    perturb_photom = photom+perturb

    return perturb_photom

def make_photom_cat(perturb_photom, errors, flux_cols, fluxerr_cols):
    """
    Build an Astropy table representing a photometric catalog from
    perturbed fluxes and their errors.

    Parameters
    ----------
    perturb_photom : ndarray
        Array of perturbed flux realizations of shape
        ``(n_realizations, n_filters)``.
    errors : array-like
        Flux uncertainties corresponding to each filter.
    flux_cols : list of str
        Column names for the flux measurements.
    fluxerr_cols : list of str
        Column names for the flux error measurements.

    Returns
    -------
    photom_table : astropy.table.Table
        Table containing ``id`` plus alternating flux and flux-error columns
        in the order specified by ``flux_cols``/``fluxerr_cols``.
    """
    
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
    """
    Convert flux density in cgs units to microjansky.

    Parameters
    ----------
    data : float or array-like
        Flux density values in erg/s/cm^2/Hz (cgs).

    Returns
    -------
    muJy : float or ndarray
        Flux density in microjansky.
    """

    return data*1e29

def convert_flambda_to_fnu(data, wavelength):
    """
    Convert flux per unit wavelength to flux per unit frequency.

    Parameters
    ----------
    data : float or array-like
        Flux density in erg/s/cm^2/Å.
    wavelength : float or array-like
        Wavelength(s) at which ``data`` is defined (Å).

    Returns
    -------
    fnu_data : float or ndarray
        Flux density in erg/s/cm^2/Hz.
    """

    c_A_s = 3e18
    fnu_data = data * (wavelength**2/c_A_s)

    return fnu_data

def convert_m_to_beta(m):
    """
    Translate a slope parameter ``m`` to the spectral index ``beta``.

    Parameters
    ----------
    m : float or array-like
        Linear slope used in some fitting convention.

    Returns
    -------
    beta : float or ndarray
        Corresponding spectral index.
    """

    beta = -m/2.5 - 2

    return beta

def beta_function(wave, beta, C):
    """
    Evaluate a power‑law magnitude relation with spectral index ``beta``.

    Parameters
    ----------
    wave : float or array-like
        Wavelength(s) (Å).
    beta : float
        Spectral index.
    C : float
        Normalization constant.

    Returns
    -------
    mag : float or ndarray
        Magnitude computed according to the formula.
    """

    return -2.5 * (beta + 2)*np.log10(wave) + C

def make_param_grid(Muv_grid, z_grid, beta_uv_grid, beta_opt_grid):
    """
    Produce a list of all combinations from the provided 1‑D parameter arrays.

    Parameters
    ----------
    Muv_grid, z_grid, beta_uv_grid, beta_opt_grid : array-like
        Vectors defining the grid for each parameter.

    Returns
    -------
    param_grid : list of tuples
        Cartesian product of the inputs, each tuple is
        ``(Muv, z, beta_uv, beta_opt)``.
    """
    param_grid = list(product(Muv_grid, z_grid, beta_uv_grid, beta_opt_grid))

    return param_grid

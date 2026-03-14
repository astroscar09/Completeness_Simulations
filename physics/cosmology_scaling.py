from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import numpy as np


def build_dl_cache(z_values):
    """Pre-compute luminosity distances for an array of redshifts.

    Parameters
    ----------
    z_values : array-like
        Unique redshift values to evaluate.

    Returns
    -------
    dict
        Mapping ``z -> d_L`` in parsecs.
    """
    z_unique = np.unique(z_values)
    dl_pc = cosmo.luminosity_distance(z_unique).to(u.pc).value
    return dict(zip(z_unique, dl_pc))


def Muv_to_F_lambda_cgs(M1500, z, dl_cache=None):
    """
    Convert observed m_1500 to M_1500 using luminosity distance and redshift.
    
    Parameters:
    - m1500 : float or array
        Apparent magnitude at 1500 Å
    - z : float or array
        Redshift
    - dl_cache : dict, optional
        Pre-computed mapping of z -> d_L (parsecs) from ``build_dl_cache``.
        Falls back to a live Astropy call when not provided.

    Returns:
    - M1500 : float or array
        Absolute magnitude at 1500 Å
    """
    if dl_cache is not None:
        DL = dl_cache[z]
    else:
        DL = cosmo.luminosity_distance(z).to(u.pc).value  # in parsecs
    #m1500 = -2.5*np.log10(f_nu/3631) #f_nu in Jy
    #M1500 = m1500 - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + z)

    f_nu_jy = 3631 * 10**((M1500 + 5 * np.log10(DL / 10) -  2.5 * np.log10(1 + z))/ -2.5 )

    f_nu_cgs = f_nu_jy * 1e-23  # Convert from Jy to cgs

    f_lambda_cgs = f_nu_cgs * (3e18 / (1500**2))  # Convert to F_lambda in cgs

    return f_lambda_cgs
import numpy as np
import pandas as pd
from astropy.table import Table

def perturb_photometry(photom, errors, num_realization):


    perturb = np.random.normal(0, scale=errors, size = (num_realization, len(errors)))

    perturb_photom = photom+perturb

    return perturb_photom

def make_photom_cat(perturb_photom, errors, flux_cols, fluxerr_cols):
    
    cols = ['id'] + list(flux_cols) + list(fluxerr_cols)
    
    DF = pd.DataFrame(perturb_photom, columns = flux_cols)

    for i, val in enumerate(errors):
        DF[fluxerr_cols[i]] = val

    DF['id'] = np.arange(1, DF.shape[0] + 1)

    return Table.from_pandas(DF[cols])
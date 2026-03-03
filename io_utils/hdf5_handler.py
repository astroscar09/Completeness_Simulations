import numpy as np
import h5py 

def save_simulation_results(
    filepath: str,
    photometry: np.ndarray,
    params: np.ndarray
):
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('photometry', data=photometry, compression='gzip')
        f.create_dataset('param_combinations', data=params)


def read_simulation_results(filepath: str,):
    
    with h5py.File(filepath, 'r') as f:
        return f
    

from .yaml_handler import load_config
import numpy as np

def read_filters():
    """
    Load the filter configuration and return file paths, names, and central wavelengths.

    The configuration is read from ``config/filters.yaml`` via ``load_config``.

    Returns
    -------
    paths : list of str
        Full paths to each filter file.
    filter_names : list of str
        Human-readable names for the filters.
    centers : np.ndarray
        Central wavelengths (Å) for each filter.
    """

    config = load_config("config/filters.yaml")

    base = config["filters"]["base_path"]
    filters = config["filters"]["list"]
    centers = config['filters']['filter_centers']

    paths = [f"{base}/{f}" for f in filters]

    filter_names = config["filters"]["filter_names"]

    return paths, filter_names, np.array(centers)

def read_filter_data(paths):
    """
    Read filter data from a file.
    
    Parameters:
    - filter_file : str
        Path to the filter file.
    
    Returns:
    - wav : np.ndarray
        Wavelengths in Angstrom

    """

    transmission_curves = [np.loadtxt(path) for path in paths] 

    return transmission_curves
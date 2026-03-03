from .yaml_handler import load_config
import numpy as np

def read_filters():

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
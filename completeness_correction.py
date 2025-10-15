import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np

df = pd.read_csv('simulation_results.csv')
mask = (df['Pz Flag'] &  df['Best-Fit z Flag'] &  df['Slope Cut'])
df['Passed_All_Cuts'] = mask.values
sorted_sim_results_df = df.sort_values(by = 'id')
sorted_sim_results_df = sorted_sim_results_df.reset_index(drop=True)
sorted_results_useful_cols_df = sorted_sim_results_df[['beta_uv', 'beta_opt', 'c_uv', 'c_opt', 'beta_uv_uerr', 'beta_uv_lerr', 'c_uv_uerr',
                                                        'c_uv_lerr', 'logf_uv', 'logf_uv_uerr', 'logf_uv_lerr', 'beta_opt_uerr',
                                                        'beta_opt_lerr', 'c_opt_uerr', 'c_opt_lerr', 'logf_opt',
                                                        'logf_opt_uerr', 'logf_opt_lerr', 'Pz Flag', 'Best-Fit z Flag', 'Emcee', 'Slope Cut',
                                                        'Passed_All_Cuts' ]]

sim_df = pd.read_csv('Sim_Photometry.txt', sep = ' ', index_col = 0)

merged_sim_df =  sim_df.join(sorted_results_useful_cols_df, lsuffix='Sim', rsuffix = 'Measured')



grouped = merged_sim_df.groupby('Muv')['Passed_All_Cuts']
injected_counts = grouped.count()
recovered_counts = grouped.sum()
completeness = recovered_counts / injected_counts


def compute_completeness(df, column, detect_column = 'Passed_All_Cuts'):
    """
    Compute the completeness for given Muv bins.
    
    Parameters:
    muv_bins (list): List of Muv bin edges.
    completeness (pd.Series): Series containing completeness values indexed by Muv.
    
    Returns:
    pd.DataFrame: DataFrame with Muv bins and corresponding completeness values.
    """
    grouped = df.groupby(column)[detect_column]
    injected_counts = grouped.count()
    recovered_counts = grouped.sum()
    completeness = recovered_counts / injected_counts

    values = np.array(list(grouped.groups.keys()))

    return values, completeness.values


def compute_completeness_2cols(df, column1, column2, detect_column = 'Passed_All_Cuts'):
    """
    Compute the completeness for given Muv bins.
    
    Parameters:
    muv_bins (list): List of Muv bin edges.
    completeness (pd.Series): Series containing completeness values indexed by Muv.
    
    Returns:
    pd.DataFrame: DataFrame with Muv bins and corresponding completeness values.
    """
    grouped = df.groupby(by = [column1, column2])[detect_column]
    injected_counts = grouped.count()
    recovered_counts = grouped.sum()

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        completeness = recovered_counts / injected_counts
        completeness = completeness.fillna(0.0)
    #completeness = recovered_counts / injected_counts

    values = np.array(list(grouped.groups.keys()))

    return values, completeness

beta_uv_values, completeness_beta_uv = compute_completeness(merged_sim_df, 'beta_uvSim')
beta_opt_values, completeness_beta_opt = compute_completeness(merged_sim_df, 'beta_optSim')
redshift_values, completeness_redshift = compute_completeness(merged_sim_df, 'redshift')
muv_values, completeness_muv = compute_completeness(merged_sim_df, 'Muv')


df['i_muv'] = np.digitize(merged_sim_df['Muv'], muv_values) - 1
df['i_z'] = np.digitize(merged_sim_df['redshift'], redshift_values) - 1
df['i_beta_opt'] = np.digitize(merged_sim_df['beta_optSim'], beta_opt_values) - 1
df['i_beta_uv'] = np.digitize(merged_sim_df['beta_uvSim'], beta_uv_values) - 1

# merge_completion['i_muv'] = np.digitize(merge_completion['Muv'].values, muv_values) - 1
# merge_completion['i_z'] = np.digitize(merge_completion['redshift'], redshift_values) - 1
# merge_completion['i_beta_opt'] = np.digitize(merge_completion['beta_opt'], beta_opt_values) - 1
# merge_completion['i_beta_uv'] = np.digitize(merge_completion['beta_uv'], beta_uv_values) - 1


shape = (len(muv_values), len(redshift_values), len(beta_opt_values), len(beta_uv_values))

completeness_array = np.full(shape, np.nan)

grouped = df.groupby(['i_muv', 'i_z', 'i_beta_opt', 'i_beta_uv'])

for idx, group in tqdm(grouped, total=len(grouped)):
    i, j, k, l = idx
    injected = len(group)
    recovered = group['Passed_All_Cuts'].sum()
    if injected > 0:
        completeness_array[i, j, k, l] = recovered / injected


def get_completeness(Muv, z, beta_opt, beta_uv,
                     Muv_bins, z_bins, beta_opt_bins, beta_uv_bins,
                     completeness_array):
    """
    Returns completeness from the 4D array for input parameters.
    If input is out of range, returns np.nan.
    
    Parameters:
        Muv, z, beta_opt, beta_uv : float
            Input parameter values.
        Muv_bins, z_bins, beta_opt_bins, beta_uv_bins : array-like
            Bin edges used to create completeness_array.
        completeness_array : 4D numpy array
            Completeness values indexed by bin indices.
    
    Returns:
        completeness : float or np.nan
    """
    # Find bin indices
    i_muv = np.digitize([Muv], Muv_bins)[0] - 1
    i_z = np.digitize([z], z_bins)[0] - 1
    i_beta_opt = np.digitize([beta_opt], beta_opt_bins)[0] - 1
    i_beta_uv = np.digitize([beta_uv], beta_uv_bins)[0] - 1
    
    # Check if indices are valid (inside array bounds)
    if (0 <= i_muv < completeness_array.shape[0] and
        0 <= i_z < completeness_array.shape[1] and
        0 <= i_beta_opt < completeness_array.shape[2] and
        0 <= i_beta_uv < completeness_array.shape[3]):
        
        return completeness_array[i_muv, i_z, i_beta_opt, i_beta_uv]
    else:
        return np.nan  # or 0 if you prefer


def completeness_vs_param(free_param_name, free_param_bins, fixed_params, bins_dict, completeness_array):
    """
    Returns completeness array as a function of one free parameter, fixing the others.
    
    Parameters:
        free_param_name : str
            One of 'Muv', 'z', 'beta_opt', 'beta_uv' — the parameter to vary.
        free_param_bins : array-like
            Bin edges for the free parameter.
        fixed_params : dict
            Fixed parameter values, keys: 'Muv', 'z', 'beta_opt', 'beta_uv'.
        bins_dict : dict
            Dictionary of bin arrays keyed by parameter names.
        completeness_array : np.ndarray
            4D completeness array ordered as (Muv, z, beta_opt, beta_uv).
    
    Returns:
        completeness_1d : np.ndarray
            Array of completeness values along free parameter bins.
        bin_centers : np.ndarray
            Centers of the free parameter bins.
    """
    
    # Map parameter names to axis indices
    param_order = ['Muv', 'z', 'beta_uv', 'beta_opt',]
    axis_idx = param_order.index(free_param_name)
    
    # Find indices of fixed parameters
    fixed_indices = []
    for p in param_order:
        if p == free_param_name:
            fixed_indices.append(slice(None))  # full slice along free axis
        else:
            # Digitize fixed param value
            val = fixed_params[p]
            bins = bins_dict[p]
            i = np.digitize([val], bins)[0] - 1
            # Check bounds
            if i < 0 or i >= len(bins):
                raise ValueError(f"Fixed parameter {p}={val} out of bin range.")
            fixed_indices.append(i)
    
    # Convert to tuple for indexing
    idx_tuple = tuple(fixed_indices)
    
    # Slice completeness array along free axis
    completeness_1d = completeness_array[idx_tuple]
    
    # Compute bin centers for plotting
    bin_centers = 0.5 * (free_param_bins[:-1] + free_param_bins[1:])
    
    return completeness_1d, bin_centers


bins_dict = {
    'Muv': muv_values,
    'z': redshift_values,
    'beta_opt': beta_opt_values,
    'beta_uv': beta_uv_values
}

fixed_params = {
    'Muv': -20,
    'z': None,          # free param, will be ignored
    'beta_opt': 1,
    'beta_uv': -0.5
}

completeness_vs_z, z_centers = completeness_vs_param('z', redshift_values, 
                                                     fixed_params={'Muv': muv_values[0], 'beta_opt': beta_opt_values[0], 'beta_uv': beta_uv_values[0]},
                                                     bins_dict=bins_dict,
                                                     completeness_array=completeness_array)


def eval_C_at(Muv, z, C, muv_bins, z_bins):
    import numpy as np
    muv_idx = np.digitize(Muv, muv_bins) - 1
    z_idx   = np.digitize(z,   z_bins) - 1
    # bounds check
    mask = (muv_idx>=0)&(muv_idx<C.shape[0])&(z_idx>=0)&(z_idx<C.shape[1])
    out = np.full(Muv.shape, np.nan)
    out[mask] = C.values[muv_idx[mask], z_idx[mask]]
    return out

# Computing volumes for the 
# from astropy.cosmology import Planck18 as cosmo
# import astropy.units as u

# # solid angle of survey (in steradians)
# area_deg2 = 1.0  # example: 1 square degree
# solid_angle = (area_deg2 * u.deg**2).to(u.sr)

# z1, z2 = 2.0, 3.0  # example redshift bin

# # comoving volume between z1 and z2
# vol = cosmo.comoving_volume(z2) - cosmo.comoving_volume(z1)
# vol_bin = vol * (solid_angle / (4*np.pi*u.sr))  # scale by survey area

# print(vol_bin.to(u.Mpc**3))
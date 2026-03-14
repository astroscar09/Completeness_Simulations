import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def plot_setup():
    
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Wavelength', fontsize = 15)

    return fig, ax

def plot_spectrum(wave, flux, ax, show = None):

    ax.loglog(wave, flux)
    if show:
        plt.show()


def param_setup_plot():

    fig, axes = plt.subplots(2, 2, tight_layout = True)
    axes = axes.flatten()
    
    return fig, axes

def plot_histogram(ax, data, bins, color, xlabel, show = None):

    ax.hist(data, bins = bins, color = color)
    ax.set_xlabel(xlabel, fontsize = 15)
    if show:
        plt.show()


def plot_example_spectrum(
    wav_obs,
    flux_fnu_ujy,
    filter_centers,
    phot_ujy,
    perturbed_ujy,
    errors_ujy,
    filter_wave_grid,
    all_filters,
    redshift,
    filter_names=None,
    savepath=None,
):
    """
    Plot the simulated observed-frame spectrum alongside synthetic photometry
    and filter transmission curves.

    Parameters
    ----------
    wav_obs : array-like
        Observed-frame wavelength grid (Å).
    flux_fnu_ujy : array-like
        F_nu spectrum in µJy on wav_obs.
    filter_centers : array-like
        Central wavelengths of each filter (Å).
    phot_ujy : array-like
        Noiseless synthetic photometry in µJy.
    perturbed_ujy : array-like
        One noisy realization of the photometry in µJy.
    errors_ujy : array-like
        1-sigma photometric errors in µJy.
    filter_wave_grid : array-like
        Common wavelength grid for filter transmission curves (Å).
    all_filters : list of array-like
        Normalized transmission arrays for each filter.
    filter_names : list of str, optional
        Names for each filter (used in legend).
    savepath : str, optional
        If given, save the figure to this path instead of displaying.
    """
    cmap = plt.cm.get_cmap('tab10')
    n_filters = len(all_filters)
    colors = [cmap(i % 10) for i in range(n_filters)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    # Plot filter transmission curves on right axis
    #wav_um_grid = filter_wave_grid / 1e4
    for i, (transmission, color) in enumerate(zip(all_filters, colors)):
        peak = np.max(transmission)
        if peak > 0:
            norm_trans = transmission / peak
        else:
            norm_trans = transmission
        label = filter_names[i] if filter_names is not None else f"Filter {i}"
        ax2.plot(filter_wave_grid, norm_trans, color=color, alpha=0.5, linewidth=1, label=label)
    ax2.set_ylim(0, 1.5)
    ax2.set_ylabel("Transmission", fontsize=13)

    # Plot spectrum on main axis (log-log)
    #wav_um = wav_obs / 1e4
    ax.loglog(wav_obs, flux_fnu_ujy, color='black', linewidth=0.8, label=f'z = {redshift:.2f} Spectrum')

    # Noiseless photometry
    centers_um = filter_centers #/ 1e4
    ax.scatter(centers_um, phot_ujy, c=colors, s=60, zorder=5, label='Noiseless photometry')

    # Perturbed photometry with error bars
    for i, (xc, yp, ye, color) in enumerate(zip(centers_um, perturbed_ujy, errors_ujy, colors)):
        ax.errorbar(xc, yp, yerr=ye, fmt='o', color=color,
                    markerfacecolor='none', markeredgewidth=1.5,
                    capsize=3, zorder=6)

    # Dummy handle for perturbed photometry legend entry
    perturbed_handle = plt.Line2D(
        [0], [0], marker='o', color='gray', markerfacecolor='none',
        markeredgewidth=1.5, linestyle='none', label='Perturbed photometry'
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(perturbed_handle)

    ax.set_xlabel("Observed Wavelength (Angstroms)", fontsize=13)
    ax.set_ylabel(r"$F_\nu$ (µJy)", fontsize=13)
    ax.set_xlim(np.amin(centers_um)-1000, np.amax(centers_um)+10000)
    
    positive_values = perturbed_ujy[perturbed_ujy > 0]
    
    ax.set_ylim(np.amin(positive_values)*.9, np.amax(positive_values)*1.2)

    # Filter legend on right axis
    ax2.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.7)
    ax.legend(handles=handles, loc='upper left', fontsize=10)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



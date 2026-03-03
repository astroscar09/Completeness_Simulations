import matplotlib.pyplot as plt
#plt.style.use('style.mplstyle')

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



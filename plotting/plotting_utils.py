import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

def plot_spectrum(wave, flux):

    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    ax.loglog(wave, flux)

    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Flux')

    return fig, ax 

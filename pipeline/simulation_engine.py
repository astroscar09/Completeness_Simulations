from physics.spectra import generate_restframe_spectrum
from physics.photometry_projection import photometry

def run_single_simulation(
    Muv,
    z,
    beta_uv,
    beta_opt,
    filter_trans,
    grid_wav
):
    """
    Execute one forward-model simulation.
    """

    spectrum = generate_restframe_spectrum(
        beta_uv=beta_uv,
        beta_opt=beta_opt,
        wav_break=3500,
        norm_wav=1500,
        wav_range=(912, 40000),
    )

    photometry = photometry(
        spectrum,
        filter_trans,
        grid_wav,
        Muv,
        z
    )

    return photometry
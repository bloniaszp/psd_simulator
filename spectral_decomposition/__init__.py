from spectral_decomposition.simulation import CombinedSimulator, BaseSimulator
from spectral_decomposition.decomposition import ParametricDecomposition, BaseDecomposition
from spectral_decomposition.time_domain import TimeDomainData
from spectral_decomposition.frequency_domain import FrequencyDomainData
from spectral_decomposition.plotting import PSDPlotter, BasePlotter
import numpy as np


def spectrum(
    sampling_rate=1000.0,
    duration=2.0,
    n_samples=None,
    aperiodic_exponent=1.0,
    aperiodic_offset=1.0,
    knee=None,
    peaks=None,
    direct_estimate=False,
    plot=False,
    average_firing_rate=0.0,
    random_state=None
):
    """
    Simulate a time series and its power spectrum given aperiodic (1/f) + knee and
    periodic (Gaussian peak) components.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    duration : float
        Duration of the signal in seconds (if n_samples is not given).
    n_samples : int, optional
        Number of samples; if provided, duration is ignored.
    aperiodic_exponent : float
        Exponent for the aperiodic 1/f component.
    aperiodic_offset : float
        Offset (in log10 space) for the aperiodic component.
    knee : float or None
        Knee parameter (kappa). If None, treated as 0.0 (i.e., no knee).
    peaks : list of dict, optional
        Periodic components. Each dict should have keys: 'freq', 'amplitude', 'sigma'.
    direct_estimate : bool
        If True, perform empirical PSD estimation using spectral_connectivity on the
        time-domain data. (Requires `spectral_connectivity` installed.)
    plot : bool
        If True, generate and show a plot of the PSD decomposition (matplotlib).
    average_firing_rate : float
        Constant offset added to the final simulated time-domain signal.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    SpectralDecompositionResult
        An object containing time domain data, frequency domain data, and parameters.
    """

    # 1) Simulate
    simulator = CombinedSimulator(
        sampling_rate=sampling_rate,
        n_samples=n_samples,
        duration=duration,
        aperiodic_exponent=aperiodic_exponent,
        aperiodic_offset=aperiodic_offset,
        knee=knee,
        peaks=peaks,
        average_firing_rate=average_firing_rate,
        random_state=random_state
    )
    time_data = simulator.simulate()

    # 2) Theoretical PSD on the *same* high-res grid
    decomposer = ParametricDecomposition(
        sampling_rate=sampling_rate,
        n_fft=simulator.n_fft,   # <-- Use simulator's large n_fft
        aperiodic_exponent=aperiodic_exponent,
        aperiodic_offset=aperiodic_offset,
        knee=knee,
        peaks=peaks
    )
    freq_data = decomposer.compute()

    # 3) Optional empirical PSD
    if direct_estimate:
        try:
            from spectral_connectivity import Multitaper, Connectivity
        except ImportError:
            raise ImportError("Install 'spectral_connectivity' for direct_estimate=True.")

        signal = time_data.combined_signal
        m = Multitaper(
            time_series=signal,
            sampling_frequency=sampling_rate,
            n_tapers = 5,
            #time_halfbandwidth_product=5,
            n_fft_samples=len(time_data)  
        )
        c = Connectivity.from_multitaper(m)
        power = c.power().squeeze()  
        freqs_emp = c.frequencies

        # Retain only positive freq from theoretical PSD
        pos_mask = freq_data.frequencies >= 0
        freq_data.frequencies = freq_data.frequencies[pos_mask]
        freq_data.broadband_spectrum = freq_data.broadband_spectrum[pos_mask]
        freq_data.rhythmic_spectrum = freq_data.rhythmic_spectrum[pos_mask]
        freq_data.combined_spectrum = freq_data.combined_spectrum[pos_mask]

        # If DC bin mismatch
        if freqs_emp[0] > 0 and freq_data.frequencies[0] == 0:
            freq_data.frequencies         = freq_data.frequencies[1:]
            freq_data.broadband_spectrum = freq_data.broadband_spectrum[1:]
            freq_data.rhythmic_spectrum  = freq_data.rhythmic_spectrum[1:]
            freq_data.combined_spectrum  = freq_data.combined_spectrum[1:]

        if len(freqs_emp) != len(freq_data.frequencies):
            # Interpolate the theoretical PSD onto the empirical frequency axis.
            original_freqs = freq_data.frequencies
            original_broad = freq_data.broadband_spectrum
            original_rhyth = freq_data.rhythmic_spectrum
            original_comb  = freq_data.combined_spectrum

            # Overwrite freq_data with the empirical axis
            freq_data.frequencies = freqs_emp

            # Interpolate each theoretical PSD array
            freq_data.broadband_spectrum = np.interp(freqs_emp, original_freqs, original_broad)
            freq_data.rhythmic_spectrum  = np.interp(freqs_emp, original_freqs, original_rhyth)
            freq_data.combined_spectrum  = np.interp(freqs_emp, original_freqs, original_comb)

        freq_data.empirical_spectrum = power.T

    # 4) Package
    params = {
        "sampling_rate": sampling_rate,
        "n_samples": len(time_data),
        "duration": float(len(time_data)) / sampling_rate,
        "aperiodic_exponent": aperiodic_exponent,
        "aperiodic_offset": aperiodic_offset,
        "knee": knee if knee is not None else 0.0,
        "peaks": peaks if peaks else [],
        "direct_estimate": direct_estimate,
        "average_firing_rate": average_firing_rate,
        "random_state": random_state
    }

    class SpectralDecompositionResult:
        """Container for the final results."""
        def __init__(self, time_domain, frequency_domain, params):
            self.time_domain = time_domain
            self.frequency_domain = frequency_domain
            self.params = params

        def plot(self):
            fig = PSDPlotter().plot(self.frequency_domain)
            return fig

    result = SpectralDecompositionResult(time_data, freq_data, params)
    if plot:
        fig = result.plot()
        fig.show()

    return result

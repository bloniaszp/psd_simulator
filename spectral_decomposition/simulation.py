import numpy as np
from abc import ABC, abstractmethod
from numpy.fft import ifft, ifftshift


###############################################################################
# 1) simulate_from_psd with two lengths: one for PSD (n_fft) and one for time (n_time)
###############################################################################
def simulate_from_psd(PSD, fs, n_fft, n_time, random_seed=None, lambda_0=0.0):
    """
    Draw a time‐domain realization via random phases + iFFT from 'PSD'.

    * PSD is length = n_fft (the large frequency grid).
    * The final time signal is length = n_time (the actual desired #samples).

    Parameters
    ----------
    PSD : np.ndarray, shape (n_fft,)
        Two-sided PSD array (unshifted, DC at index=0, possibly Nyquist at n_fft//2).
    fs : float
        Sampling rate in Hz.
    n_fft : int
        Size of the FFT grid used to define PSD and do the iFFT (often >> n_time).
    n_time : int
        Number of samples to keep in the final time-domain output.
    random_seed : int or None
        For reproducible random phases.
    lambda_0 : float
        Constant offset added to the final time-domain signal.

    Returns
    -------
    time_signal : np.ndarray, shape (n_time,)
        Real-valued time-domain signal.
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    if len(PSD) != n_fft:
        raise ValueError(f"PSD length ({len(PSD)}) must match n_fft={n_fft}.")

    halfM = n_fft // 2

    # Build random complex amplitudes
    U = np.zeros(n_fft, dtype=np.complex128)

    # DC bin
    U[0] = np.sqrt(PSD[0]) * rng.randn()

    # Positive freqs: 1..halfM
    pos_psd = PSD[1 : halfM + 1] / 2.0
    amp = np.sqrt(pos_psd)
    U[1 : halfM + 1] = amp * rng.randn(len(amp)) + 1j * amp * rng.randn(len(amp))

    # Negative freqs (mirror for real-valued ifft)
    U[halfM + 1 :] = np.flipud(np.conj(U[1 : n_fft - halfM]))

    # Nyquist bin if even
    if n_fft % 2 == 0:
        U[halfM] = np.sqrt(PSD[halfM]) * rng.randn()

    # iFFT
    signal_freq_domain = np.sqrt(fs * n_fft) * ifft(ifftshift(U))
    # Keep only the first n_time samples
    time_signal = np.real_if_close(signal_freq_domain[:n_time])

    # Add offset
    time_signal += lambda_0
    return time_signal


###############################################################################
# 2) A base class for simulators
###############################################################################
class BaseSimulator(ABC):
    """Abstract base class for time series simulation."""
    @abstractmethod
    def simulate(self):
        raise NotImplementedError


###############################################################################
# 3) A Combined Simulator that uses a large n_fft for the PSD
###############################################################################
class CombinedSimulator(BaseSimulator):
    """
    Generates a combined time series with broadband (aperiodic) and rhythmic (periodic) components,
    using a large n_fft for high-resolution PSD, then slicing the first n_samples in time domain.

    Steps:
      1) Define n_fft >> n_samples for the frequency grid (user can set or we pick automatically).
      2) Build symmetrical freq axis [-fs/2..+fs/2) of length n_fft.
      3) Construct broadband PSD = 10^(aperiodic_offset)/(knee^2 + |f|^exponent).
      4) Construct rhythmic PSD via Gaussians at ±f0, sum => combined PSD.
      5) iFFT => time signals of length n_fft, then slice out first n_samples.
      6) Add `average_firing_rate` in time domain.
    """
    def __init__(
        self,
        sampling_rate,
        n_samples=None,
        duration=None,
        aperiodic_exponent=1.0,
        aperiodic_offset=1.0,
        knee=None,
        peaks=None,
        average_firing_rate=0.0,
        n_fft=None,
        random_state=None
    ):
        """
        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz).
        n_samples : int
            Number of time-domain samples to return.
        duration : float
            If n_samples not given, compute it via duration * sampling_rate.
        aperiodic_exponent : float
            1/f exponent.
        aperiodic_offset : float
            log10 offset for the aperiodic PSD.
        knee : float or None
            Knee parameter (k^2 in denominator).
        peaks : list of dict
            Each with {'freq':..., 'amplitude':..., 'sigma':...}.
        average_firing_rate : float
            DC offset added to the final combined time signal.
        n_fft : int or None
            FFT size for building PSD (often >> n_samples). If None, pick a large power-of-2 
            above n_samples + some offset.
        random_state : int or None
            Seed for reproducible phases.
        """
        self.sampling_rate = sampling_rate

        # Determine final time-series length
        if n_samples is None:
            if duration is None:
                raise ValueError("Must specify either n_samples or duration.")
            self.n_samples = int(duration * sampling_rate)
            if self.n_samples < 1:
                raise ValueError("Duration too short for the given sampling rate.")
        else:
            self.n_samples = int(n_samples)
            if duration is not None:
                expected_n = int(duration * sampling_rate)
                if expected_n != self.n_samples:
                    raise ValueError("n_samples and duration are inconsistent.")

        # Decide on n_fft if none provided
        if n_fft is None:
            self.n_fft = 100000 + (2 ** int(np.ceil(np.log2(self.n_samples))))
        else:
            self.n_fft = int(n_fft)

        self.aperiodic_exponent = aperiodic_exponent
        self.aperiodic_offset = aperiodic_offset
        self.knee = knee if knee is not None else 0.0
        self.peaks = peaks if peaks is not None else []
        self.average_firing_rate = average_firing_rate
        self.random_state = random_state

    def simulate(self):
        """
        Returns
        -------
        TimeDomainData
            Contains time array and broadband_signal, rhythmic_signal, combined_signal.
        """
        fs = self.sampling_rate
        n_time = self.n_samples
        n_fft = self.n_fft
        rng_seed = self.random_state

        # Build symmetrical freq axis of length n_fft: [-fs/2 .. +fs/2)
        freqs_shifted = np.fft.fftfreq(n_fft, d=1.0/fs)
        #freqs_shifted = np.fft.fftshift(freqs_shifted)

        # Broadband PSD: 10^(offset) / (k^2 + |f|^exponent)
        denom = (self.knee**2 + np.abs(freqs_shifted)**self.aperiodic_exponent)
        with np.errstate(divide='ignore', invalid='ignore'):
            broadband_psd_shifted = (10.0**self.aperiodic_offset) / denom
        zero_idx = np.argmin(np.abs(freqs_shifted))
        broadband_psd_shifted[zero_idx] = 0.0

        # Rhythmic PSD: sum Gaussians around ±f0
        rhythmic_psd_shifted = np.zeros_like(freqs_shifted)
        for peak in self.peaks:
            f0 = peak.get('freq')
            amp = peak.get('amplitude')
            sigma = peak.get('sigma', None)
            if f0 is None or amp is None or sigma is None:
                continue
            rhythmic_psd_shifted += amp * np.exp(-((freqs_shifted - f0)**2)/(2*sigma**2))
            rhythmic_psd_shifted += amp * np.exp(-((freqs_shifted + f0)**2)/(2*sigma**2))
        rhythmic_psd_shifted[zero_idx] = 0.0

        # Combined PSD (shifted)
        combined_psd_shifted = broadband_psd_shifted + rhythmic_psd_shifted

        # For simulation, we need unshifted PSD
        broadband_psd_unshifted = np.fft.ifftshift(broadband_psd_shifted)
        rhythmic_psd_unshifted  = np.fft.ifftshift(rhythmic_psd_shifted)
        combined_psd_unshifted  = np.fft.ifftshift(combined_psd_shifted)

        # Simulate each partial, slice out n_time
        broadband_signal_big = simulate_from_psd(
            broadband_psd_unshifted, fs, n_fft, n_fft,
            random_seed=rng_seed, lambda_0=0.0
        )
        rhythmic_signal_big = simulate_from_psd(
            rhythmic_psd_unshifted, fs, n_fft, n_fft,
            random_seed=rng_seed, lambda_0=0.0
        )
        combined_signal_big = broadband_signal_big + rhythmic_signal_big + self.average_firing_rate

        broadband_signal = broadband_signal_big[:n_time]
        rhythmic_signal = rhythmic_signal_big[:n_time]
        combined_signal = combined_signal_big[:n_time]

        time = np.arange(n_time) / fs
        from spectral_decomposition.time_domain import TimeDomainData
        return TimeDomainData(
            time=time,
            combined_signal=combined_signal,
            broadband_signal=broadband_signal,
            rhythmic_signal=rhythmic_signal
        )

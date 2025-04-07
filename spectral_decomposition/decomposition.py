"""
Decomposition module for spectral_decomposition.

Defines classes that compute the spectral decomposition (aperiodic + periodic components)
on a frequency axis symmetric around 0.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseDecomposition(ABC):
    """Abstract base class for spectral decomposition."""
    @abstractmethod
    def compute(self):
        pass

class ParametricDecomposition(BaseDecomposition):
    """
    Compute PSD components from aperiodic + periodic parameters (theoretical).
    Uses the same high-resolution grid (n_fft) as the simulator.
    """
    def __init__(
        self,
        sampling_rate,
        n_fft,
        aperiodic_exponent,
        aperiodic_offset,
        knee,
        peaks
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.aperiodic_exponent = aperiodic_exponent
        self.aperiodic_offset = aperiodic_offset
        self.knee = knee if knee is not None else 0.0
        self.peaks = peaks if peaks is not None else []

    def compute(self):
        # Create the same high-res freq axis: [-fs/2..+fs/2), length = n_fft
        freqs = np.fft.fftfreq(self.n_fft, d=1.0/self.sampling_rate)
        freqs = np.fft.fftshift(freqs)

        # Aperiodic
        denom = (self.knee**2 + np.abs(freqs)**self.aperiodic_exponent)
        with np.errstate(divide='ignore', invalid='ignore'):
            broadband_psd = (10.0 ** self.aperiodic_offset) / denom
        zero_idx = np.argmin(np.abs(freqs))
        broadband_psd[zero_idx] = 0.0

        # Rhythmic
        rhythmic_psd = np.zeros_like(freqs)
        for peak in self.peaks:
            f0 = peak.get('freq')
            amp = peak.get('amplitude')
            sigma = peak.get('sigma', None)
            if f0 is None or amp is None or sigma is None:
                continue
            rhythmic_psd += amp * np.exp(-((freqs - f0)**2)/(2*sigma**2))
            rhythmic_psd += amp * np.exp(-((freqs + f0)**2)/(2*sigma**2))
        rhythmic_psd[zero_idx] = 0.0

        total_psd = broadband_psd + rhythmic_psd
        from spectral_decomposition.frequency_domain import FrequencyDomainData
        return FrequencyDomainData(
            frequencies=freqs,
            combined_spectrum=total_psd,
            broadband_spectrum=broadband_psd,
            rhythmic_spectrum=rhythmic_psd,
            empirical_spectrum=None
        )
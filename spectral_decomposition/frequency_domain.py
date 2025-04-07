"""
Frequency domain data structures for spectral_decomposition.
"""
import numpy as np

class FrequencyDomainData:
    """
    Container for frequency domain (power spectrum) data.

    Attributes
    ----------
    frequencies : numpy.ndarray
        Array of frequency values (Hz), symmetric about 0, length = n_samples.
    combined_spectrum : numpy.ndarray
        PSD of combined (aperiodic + periodic).
    broadband_spectrum : numpy.ndarray
        PSD of the broadband (aperiodic) component.
    rhythmic_spectrum : numpy.ndarray
        PSD of the rhythmic (periodic) component(s).
    empirical_spectrum : numpy.ndarray or None
        Empirically estimated PSD, if direct_estimate=True was used; otherwise None.
    """
    def __init__(
        self,
        frequencies: np.ndarray,
        combined_spectrum: np.ndarray,
        broadband_spectrum: np.ndarray,
        rhythmic_spectrum: np.ndarray,
        empirical_spectrum: np.ndarray = None
    ):
        self.frequencies = frequencies
        self.combined_spectrum = combined_spectrum
        self.broadband_spectrum = broadband_spectrum
        self.rhythmic_spectrum = rhythmic_spectrum
        self.empirical_spectrum = empirical_spectrum

    def __len__(self):
        """Return the number of frequency points."""
        return len(self.frequencies)

    def __repr__(self):
        return (
            f"FrequencyDomainData(n_freqs={len(self.frequencies)}, "
            f"freq_range=[{self.frequencies[0]:.2f}-{self.frequencies[-1]:.2f}] Hz)"
        )

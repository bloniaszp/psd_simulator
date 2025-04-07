"""
Time domain data structures for spectral_decomposition.
"""
import numpy as np

class TimeDomainData:
    """
    Container for time domain signals.

    Attributes
    ----------
    time : np.ndarray
        Time array (seconds), length = n_samples.
    combined_signal : np.ndarray
        Combined time series (broadband + rhythmic + average_firing_rate).
    broadband_signal : np.ndarray
        Time series of the broadband (aperiodic) component only.
    rhythmic_signal : np.ndarray
        Time series of the rhythmic (periodic) component only.
    """
    def __init__(
        self,
        time: np.ndarray,
        combined_signal: np.ndarray,
        broadband_signal: np.ndarray,
        rhythmic_signal: np.ndarray
    ):
        self.time = time
        self.combined_signal = combined_signal
        self.broadband_signal = broadband_signal
        self.rhythmic_signal = rhythmic_signal

    def __len__(self):
        return len(self.time)

    def __repr__(self):
        return (
            f"TimeDomainData(n_samples={len(self.time)}, "
            f"combined_signal_mean={np.mean(self.combined_signal):.3f})"
        )

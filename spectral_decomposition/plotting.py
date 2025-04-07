"""
Plotting module for spectral_decomposition.
"""
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BasePlotter(ABC):
    """Abstract base class for plotting spectral decomposition results."""
    @abstractmethod
    def plot(self, *args, **kwargs):
        """Plot data. To be implemented by subclasses."""
        raise NotImplementedError

class PSDPlotter(BasePlotter):
    """Plotter for power spectral density decomposition (log-log)."""
    def plot(self, frequency_data):
        """
        Plot the theoretical and optional empirical PSD components on a log-log axis.

        Parameters
        ----------
        frequency_data : FrequencyDomainData
            Contains .frequencies and the various PSD arrays.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the PSD plot.
        """
        freqs = frequency_data.frequencies
        # We'll focus on positive frequencies for visualization
        positive_mask = freqs > 0
        f_plot = freqs[positive_mask]

        fig, ax = plt.subplots(figsize=(6,4))

        # Plot broadband (aperiodic)
        ax.loglog(
            f_plot,
            frequency_data.broadband_spectrum[positive_mask],
            label="Broadband (1/f + knee)",
            linestyle="--",
            color = 'blue',
        )
        # Plot rhythmic
        ax.loglog(
            f_plot,
            frequency_data.rhythmic_spectrum[positive_mask],
            label="Rhythmic",
            linestyle="--",
            color = 'green',
        )
        # Plot combined
        ax.loglog(
            f_plot,
            frequency_data.combined_spectrum[positive_mask],
            label="Combined (theoretical)",
            linewidth=2.0,
            color="black"
        )
        # If we have an empirical PSD
        if frequency_data.empirical_spectrum is not None:
            ax.loglog(
                f_plot,
                frequency_data.empirical_spectrum[positive_mask],
                label="Combined (empirical)",
                color = 'black',
                alpha = 0.5
            )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (arbitrary units)")
        ax.legend()
        ax.set_title("PSD Decomposition (Theoretical vs Empirical)")
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        # Optionally fix y-lims to something reasonable:
        ymin = 1e-13
        ymax = frequency_data.combined_spectrum[positive_mask].max() + 100
        ax.set_ylim([ymin, ymax])

        fig.tight_layout()
        return fig

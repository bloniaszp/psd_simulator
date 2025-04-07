import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import spectral_decomposition

try:
    import spectral_connectivity
    HAS_SPECTRAL_CONN = True
except ImportError:
    HAS_SPECTRAL_CONN = False


def test_time_domain_components():
    """Test that time domain signals add up correctly, and no average_firing_rate => near-zero mean."""
    result = spectral_decomposition.spectrum(
        sampling_rate=500,
        duration=2.0,
        aperiodic_exponent=1.0,
        aperiodic_offset=1.0,
        knee=None,
        peaks=[{'freq': 10, 'amplitude': 0.5, 'sigma': 2.0}],
        direct_estimate=False,
        plot=False,
        average_firing_rate=0.0,
        random_state=42
    )
    combined = result.time_domain.combined_signal
    broadband = result.time_domain.broadband_signal
    rhythmic = result.time_domain.rhythmic_signal

    # Combined == broadband + rhythmic
    assert np.allclose(combined, broadband + rhythmic, atol=1e-10)

    # Basic param checks
    for key in (
        "sampling_rate",
        "n_samples",
        "duration",
        "aperiodic_exponent",
        "aperiodic_offset",
        "peaks",
        "direct_estimate"
    ):
        assert key in result.params


def test_frequency_domain_components():
    """Test that broadband + rhythmic = combined, and check 1/f slope in the broadband PSD."""
    exponent = 2.0
    result = spectral_decomposition.spectrum(
        sampling_rate=500,
        duration=2.0,
        aperiodic_exponent=exponent,
        aperiodic_offset=1.0,
        knee=None,
        peaks=[{'freq': 10, 'amplitude': 0.5, 'sigma': 1.0}],
        direct_estimate=False,
        plot=False,
        random_state=1
    )
    fdata = result.frequency_domain
    f = fdata.frequencies
    # Check lengths match
    assert len(fdata) == len(fdata.combined_spectrum)
    assert len(fdata) == len(fdata.broadband_spectrum)
    assert len(fdata) == len(fdata.rhythmic_spectrum)
    # Combined = broadband + rhythmic
    assert np.allclose(
        fdata.combined_spectrum,
        fdata.broadband_spectrum + fdata.rhythmic_spectrum
    )

    # Check power-law ratio
    idx_f1 = np.argmin(np.abs(f - 5.0))
    idx_f2 = np.argmin(np.abs(f - 10.0))
    if f[idx_f1] != 0 and f[idx_f2] != 0:
        measured_ratio = fdata.broadband_spectrum[idx_f1] / fdata.broadband_spectrum[idx_f2]
        expected_ratio = (np.abs(f[idx_f2]) / np.abs(f[idx_f1])) ** exponent
        assert pytest.approx(measured_ratio, rel=1e-2) == expected_ratio

def test_empirical_psd_estimation():
    """Test that empirical PSD estimation is included if spectral_connectivity is present."""
    if not HAS_SPECTRAL_CONN:
        with pytest.raises(ImportError):
            spectral_decomposition.spectrum(
                sampling_rate=200,
                duration=1.0,
                aperiodic_exponent=1.0,
                aperiodic_offset=1.0,
                knee=None,
                peaks=[],
                direct_estimate=True,
                plot=False
            )
    else:
        result = spectral_decomposition.spectrum(
            sampling_rate=200,
            duration=1.0,
            aperiodic_exponent=1.0,
            aperiodic_offset=1.0,
            knee=None,
            peaks=[{'freq': 20, 'amplitude': 0.3, 'sigma': 2.0}],
            direct_estimate=True,
            plot=False,
            random_state=0
        )
        fdata = result.frequency_domain
        assert fdata.empirical_spectrum is not None
        assert len(fdata.empirical_spectrum) == len(fdata.frequencies)

def test_plotting_output():
    """Test that plotting returns a matplotlib Figure."""
    from matplotlib.figure import Figure
    result = spectral_decomposition.spectrum(
        sampling_rate=300,
        duration=1.0,
        aperiodic_exponent=1.0,
        aperiodic_offset=0.5,
        knee=None,
        peaks=[{'freq': 50, 'amplitude': 0.5, 'sigma': 5.0}],
        direct_estimate=False,
        plot=False,
        random_state=5
    )
    fig = result.plot()
    assert isinstance(fig, Figure)

def test_class_inheritance():
    """Check that classes inherit from their respective ABCs."""
    from spectral_decomposition import (
        BaseSimulator, CombinedSimulator,
        BaseDecomposition, ParametricDecomposition,
        BasePlotter, PSDPlotter
    )
    assert issubclass(CombinedSimulator, BaseSimulator)
    assert issubclass(ParametricDecomposition, BaseDecomposition)
    assert issubclass(PSDPlotter, BasePlotter)

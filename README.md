# Spectral Decomposition Package

A Python package to simulate Gaussian time-domain signals from theoretical power spectra (1/f + peaks), and optionally estimate PSDs empirically with [spectral_connectivity](https://github.com/Eden-Kramer-Lab/spectral_connectivity).

## Installation

```bash
git clone https://github.com/<your_username>/spectral_decomposition.git
cd spectral_decomposition
pip install -e .
```

## Basic Usage
#### 1. Simple Simulation

```python
from spectral_decomposition import spectrum

# Simulate 2 seconds at 500 Hz, with a 1/f exponent=2.0, offset=1.0
# plus a peak at 10 Hz (amplitude=50, sigma=2).
res = spectrum(
    sampling_rate=500,
    duration=2.0,
    aperiodic_exponent=2.0,
    aperiodic_offset=1.0,
    knee=None,
    peaks=[{'freq':10, 'amplitude':50.0, 'sigma':2.0}],
    average_firing_rate=0.0,
    random_state=42,
    direct_estimate=False,  # skip empirical PSD
    plot=True
)

# Access the time-domain data
time_data = res.time_domain
print("Time-domain samples:", len(time_data))
print("Mean amplitude:", time_data.combined_signal.mean())

# Access the frequency-domain data
freq_data = res.frequency_domain
print("Number of freq bins:", len(freq_data))
```

#### 2. Empirical PSD Estimation

```python
res_emp = spectrum(
    sampling_rate=500,
    duration=10.0,
    aperiodic_exponent=1.5,
    aperiodic_offset=1.0,
    knee=10.0,
    peaks=[{'freq':12, 'amplitude':10.0, 'sigma':3.0}],
    average_firing_rate=0.0,
    random_state=0,
    direct_estimate=True,   # requires spectral_connectivity
    plot=True
)
# This will show both the theoretical and the empirically estimated PSD.
```
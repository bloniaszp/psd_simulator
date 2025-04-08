# Gaussian Process Simulator for Testing Spectral Decomposition 

A Python package to simulate Gaussian time-domain signals from theoretical power spectra (1/f + peaks), and optionally estimate PSDs empirically with [spectral_connectivity](https://github.com/Eden-Kramer-Lab/spectral_connectivity).

## Installation

```bash
# 1) Ensure you have conda installed (e.g., Miniconda or Anaconda).
#    Then install mamba into the 'base' environment:
conda install mamba -n base -c conda-forge

# 2) Clone your repository:
git clone https://github.com/bloniaszp/spectral-decomposition.git
cd spectral-decomposition

# 3) Create and activate your environment with mamba:
#    (Assumes environment.yml is in this repo)
mamba env create -f environment.yml
mamba activate spectral-decomposition

# 4) (Optional) Install your local package in editable mode
#    If environment.yml doesn't already do so:
python -m pip install -e .

# or if you want to install from GitHub directly:
# python -m pip install git+https://github.com/bloniaszp/spectral-decomposition.git

```

## Basic Usage
#### 1. Simple Simulation

![Theoretical PSD](spectral-decomposition/static/example1_psd.png)

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
```

#### 2. Empirical PSD Estimation

![Theoretical vs Empirical PSD](spectral-decomposition/static/example2_psd.png)

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

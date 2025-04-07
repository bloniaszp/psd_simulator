#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
from spectral_decomposition import spectrum

def example1():
    """
    Example 1: Single Oscillatory Peak (No Empirical PSD)
    Simulate 2 seconds at 500 Hz with:
      - Aperiodic PSD: exponent=2.0, offset=1.0
      - A single Gaussian peak: 10 Hz (amplitude=50, sigma=2)
      - No additional DC offset (average_firing_rate=0.0)
    """
    res = spectrum(
        sampling_rate=500,
        duration=2.0,
        aperiodic_exponent=2.0,
        aperiodic_offset=1.0,
        knee=None,
        peaks=[{'freq': 10, 'amplitude': 50.0, 'sigma': 2.0}],
        average_firing_rate=0.0,
        random_state=42,
        direct_estimate=False,  # skip empirical PSD estimation
        plot=True             # we will handle plotting manually
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Example 1: Theoretical PSD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/example1_psd.png")
    plt.close()

def example2():
    """
    Example 2: Single Peak with Empirical PSD
    Simulate 10 seconds at 500 Hz with:
      - Aperiodic PSD: exponent=1.5, offset=1.0, knee=10.0
      - A single Gaussian peak: 12 Hz (amplitude=10, sigma=3)
      - No additional DC offset (average_firing_rate=0.0)
      - Empirical PSD estimation enabled (requires spectral_connectivity)
    """
    res_emp = spectrum(
        sampling_rate=500,
        duration=10.0,
        aperiodic_exponent=1.5,
        aperiodic_offset=1.0,
        knee=10.0,
        peaks=[{'freq': 12, 'amplitude': 10.0, 'sigma': 3.0}],
        average_firing_rate=0.0,
        random_state=0,
        direct_estimate=True,  # enable empirical PSD estimation
        plot=True             # we handle plotting below
    )
    
    print("Example 2:")
 
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Example 2: Theoretical vs Empirical PSD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/example2_psd.png")
    plt.close()

if __name__ == '__main__':
    # Create a directory for static images if it doesn't exist.
    os.makedirs("static", exist_ok=True)
    
    print("Running Example 1...")
    example1()
    
    print("Running Example 2...")
    example2()
    
    print("Figures saved in the 'static/' folder.")

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero, mp
from scipy.signal import find_peaks
import os

# Set precision for RZF calculations
mp.dps = 10  # Precision to match observed values (5-6 decimals)

# Parameters
initial_harmonics = 100  # Initial number of harmonics
n_points = 2**20         # FFT resolution
sigma = 0.5              # Gaussian width parameter
alpha = 1.0              # Scaling for Gaussian decay
iterations = 50           # Number of refinement iterations
tolerance = 0.1          # Tolerance for matching peaks to true zeros
top_peaks_limit = 20     # Limit on number of top peaks to refine

# Step 1: Compute True Zeros
def compute_true_zeros(n_zeros):
    """Compute the first n_zeros of the Riemann zeta function on the critical line."""
    return [float(zetazero(n).imag) for n in range(1, n_zeros + 1)]

# Step 2: Generate Gaussian Harmonic Series
def gaussian_series(x, sigma, alpha):
    """Generate a Gaussian-like series."""
    return cp.exp(-alpha * cp.pi * x**2 / sigma**2)

def cosine_component(x, frequency):
    """Cosine modulation for a given frequency."""
    return cp.cos(2 * cp.pi * frequency * x)

# Step 3: Compute FFT and Peaks
def compute_fft_and_peaks(x, harmonics):
    """Compute FFT of Gaussian harmonic series and detect peaks."""
    combined = cp.zeros_like(x)
    for freq in harmonics:
        combined += gaussian_series(x, sigma, alpha) * cosine_component(x, freq)
    
    # Compute FFT
    fft_result = cp.fft.fftshift(cp.fft.fft(combined))
    fft_magnitude = cp.abs(fft_result)
    frequencies = cp.fft.fftshift(cp.fft.fftfreq(n_points, d=(x[1] - x[0])))
    
    # Convert to numpy for peak detection
    fft_magnitude_cpu = cp.asnumpy(fft_magnitude)
    frequencies_cpu = cp.asnumpy(frequencies)
    
    # Detect peaks using adaptive threshold
    height_threshold = np.percentile(fft_magnitude_cpu, 90)  # Top 10% peaks
    peaks, _ = find_peaks(fft_magnitude_cpu, height=height_threshold)
    
    return frequencies_cpu, fft_magnitude_cpu, peaks

# Step 4: Match Peaks to True Zeros
def match_true_zeros(peak_frequencies, true_zeros, tolerance):
    """Match detected FFT peaks to true zeros within a tolerance."""
    matches = []
    for pf in peak_frequencies:
        for tz in true_zeros:
            if abs(pf - tz) < tolerance:
                matches.append((pf, tz))
    return matches

# Step 5: Iterative Refinement
x = cp.linspace(-50, 50, n_points)  # Imaginary axis values
harmonics = np.linspace(0.1, 10, initial_harmonics)
true_zeros = compute_true_zeros(20)  # Compute the first 20 true zeros
errors = []  # Track total error for visualization

for iteration in range(iterations):
    print(f"\nIteration {iteration + 1}: Harmonics = {len(harmonics)}")
    frequencies, fft_magnitude, peaks = compute_fft_and_peaks(x, harmonics)
    peak_frequencies = frequencies[peaks]
    
    # Match detected peaks to true zeros and compute error
    matched_zeros = match_true_zeros(peak_frequencies, true_zeros, tolerance)
    total_error = sum(abs(pf - tz) for pf, tz in matched_zeros)
    errors.append(total_error)
    print(f"Iteration {iteration + 1}: Total Error = {total_error:.5f}")
    
    # Refine harmonics near matched peaks
    refined_peaks = [pf for pf, _ in matched_zeros]
    new_harmonics = np.array(refined_peaks) + np.random.uniform(-0.05, 0.05, size=len(refined_peaks))
    harmonics = np.unique(np.concatenate([harmonics, new_harmonics]))[:initial_harmonics + top_peaks_limit]

# Step 6: Visualization - FFT Peaks and True Zeros
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_magnitude, label="FFT Magnitude", color="blue")
plt.scatter(peak_frequencies, fft_magnitude[peaks], color='red', label="Harmonic Peaks")
for tz in true_zeros:
    if np.min(frequencies) <= tz <= np.max(frequencies):
        idx = np.abs(frequencies - tz).argmin()
        plt.scatter(frequencies[idx], fft_magnitude[idx], color='green', s=50, zorder=5)
        plt.text(frequencies[idx], fft_magnitude[idx] + 1000, f"{tz:.5f}", 
                 rotation=45, fontsize=8, color='green')
plt.title("FFT Peaks and True Zeros on the Critical Line")
plt.xlabel("Frequency (Imaginary Part)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()
plt.show()

# Step 7: Visualization - Error Reduction Over Iterations
plt.figure(figsize=(10, 5))
plt.plot(range(1, iterations + 1), errors, marker='o', color="purple", label="Error per Iteration")
plt.title("Error Reduction Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Total Error")
plt.legend()
plt.grid()
plt.show()

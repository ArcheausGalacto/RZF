import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero, mp
from scipy.signal import find_peaks

# Set precision for RZF calculations
mp.dps = 10  # Precision to match observed values (5-6 decimals)

# Parameters
initial_harmonics = 100  # Initial number of harmonics
n_points = 2**22        # FFT resolution
sigma = 0.5             # Gaussian width parameter
alpha = 1.0             # Scaling for Gaussian decay
iterations = 3          # Number of iterations to refine zeros
tolerance = 0.1         # Tolerance for matching peaks to true zeros

# Step 1: Compute True Zeros of the Riemann Zeta Function
def compute_true_zeros(n_zeros):
    """Compute the first n_zeros of the Riemann zeta function on the critical line."""
    zeros = []
    for n in range(1, n_zeros + 1):
        t = zetazero(n)  # mpmath provides zeros on the critical line
        zeros.append(float(t.imag))  # Extract the imaginary part
    return zeros

# Step 2: Generate Harmonic Gaussian Series
def gaussian_series(x, sigma, alpha):
    """Generate a Gaussian-like series."""
    return cp.exp(-alpha * cp.pi * x**2 / sigma**2)

def cosine_component(x, frequency):
    """Cosine modulation for a given frequency."""
    return cp.cos(2 * cp.pi * frequency * x)

# Step 3: FFT and Peak Detection
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
    
    # Adaptive threshold for peak detection
    height_threshold = np.percentile(fft_magnitude_cpu, 90)  # Top 10% peaks
    peaks, _ = find_peaks(fft_magnitude_cpu, height=height_threshold, distance=50)
    
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

for iteration in range(iterations):
    print(f"Iteration {iteration + 1}: Harmonics = {len(harmonics)}")
    frequencies, fft_magnitude, peaks = compute_fft_and_peaks(x, harmonics)
    peak_frequencies = frequencies[peaks]
    
    # Match detected peaks to true zeros
    matched_zeros = match_true_zeros(peak_frequencies, true_zeros, tolerance)
    
    # Output results
    for pf, tz in matched_zeros:
        print(f"Detected Peak: {pf:.5f}, True Zero: {tz:.5f}")
    
    # Add new harmonics near detected peaks
    new_harmonics = peak_frequencies + np.random.uniform(-0.05, 0.05, size=len(peak_frequencies))
    harmonics = np.unique(np.concatenate([harmonics, new_harmonics]))

# Step 6: Visualization
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_magnitude, label="FFT Magnitude", color="blue")

# Display harmonic peaks as red dots
plt.scatter(peak_frequencies, fft_magnitude[peaks], color='red', label="Harmonic Peaks")

# Overlay all true zeros as green dots
for true_zero in true_zeros:
    if np.min(frequencies) <= true_zero <= np.max(frequencies):  # Ensure zero is in range
        idx = np.abs(frequencies - true_zero).argmin()  # Find closest FFT frequency
        plt.scatter(frequencies[idx], fft_magnitude[idx], color='green', s=50, zorder=5)
        plt.text(frequencies[idx], fft_magnitude[idx] + 500, f"{true_zero:.5f}", 
                 rotation=45, fontsize=8, color='green')

plt.title("Harmonic Peaks and True Zeros on the Critical Line")
plt.xlabel("Frequency (Imaginary Part)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()
plt.show()

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta, mp
from scipy.signal import find_peaks

# Set precision for RZF calculations
mp.dps = 20  # High precision for the zeta function

# Parameters
initial_harmonics = 100  # Initial number of harmonics
n_points = 2**22        # FFT resolution
sigma = 0.5             # Gaussian width parameter
alpha = 1.0             # Scaling for Gaussian decay
iterations = 3          # Number of iterations to refine zeros

# List of true Riemann zeta function zeros on the critical line
true_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935061,
              37.586178, 40.918719, 43.327073, 48.005150, 49.773832]

# Step 1: Generate Harmonic Gaussian Series
def gaussian_series(x, sigma, alpha):
    """Generate a Gaussian-like series."""
    return cp.exp(-alpha * cp.pi * x**2 / sigma**2)

def cosine_component(x, frequency):
    """Cosine modulation for a given frequency."""
    return cp.cos(2 * cp.pi * frequency * x)

# Step 2: FFT and Peak Detection
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



# Step 3: Iterative Refinement
x = cp.linspace(-50, 50, n_points)  # Imaginary axis values
harmonics = np.linspace(0.1, 10, initial_harmonics)

for iteration in range(iterations):
    print(f"Iteration {iteration + 1}: Harmonics = {len(harmonics)}")
    frequencies, fft_magnitude, peaks = compute_fft_and_peaks(x, harmonics)
    peak_frequencies = frequencies[peaks]
    
    # Match detected peaks to true zeros
    matched_zeros = []
    for pf in peak_frequencies:
        nearest_zero = min(true_zeros, key=lambda tz: abs(tz - pf))
        matched_zeros.append((pf, nearest_zero))
    
    # Output results
    for pf, tz in matched_zeros:
        print(f"Detected Peak: {pf:.5f}, Nearest True Zero: {tz:.5f}")
    
    # Add new harmonics near detected peaks
    new_harmonics = peak_frequencies + np.random.uniform(-0.05, 0.05, size=len(peak_frequencies))
    harmonics = np.unique(np.concatenate([harmonics, new_harmonics]))

# Step 4: Visualization
plt.figure(figsize=(12, 6))
plt.plot(cp.asnumpy(frequencies), cp.asnumpy(fft_magnitude), label="FFT Magnitude")
plt.scatter(peak_frequencies, cp.asnumpy(fft_magnitude)[peaks], color='red', label="Detected Peaks")
for pf, tz in matched_zeros:
    plt.text(pf, cp.asnumpy(fft_magnitude)[peaks[np.argmin(abs(peak_frequencies - pf))]] + 100, 
             f"True: {tz:.5f}", rotation=45, fontsize=8, color='red')
plt.title("Iterative FFT Peaks Compared to True Zeros on Critical Line")
plt.xlabel("Frequency (Imaginary Part)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()
plt.show()

# Step 5: Match Detected Peaks with True Zeros on Critical Line
tolerance = 0.1  # Tolerance for matching peaks to true zeros
filtered_peaks = []
filtered_labels = []

for pf in peak_frequencies:
    for tz in true_zeros:
        if abs(pf - tz) < tolerance:  # Check if the peak is close to a true zero
            filtered_peaks.append(pf)
            filtered_labels.append(tz)

# Visualization: Only True Peaks with Red Dots
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_magnitude, label="FFT Magnitude", color="blue")
plt.scatter(filtered_peaks, [fft_magnitude[np.abs(frequencies - pf).argmin()] for pf in filtered_peaks],
            color='red', label="True Zeros (Detected Peaks)")

# Add Labels to the Filtered Peaks
for i, (pf, tz) in enumerate(zip(filtered_peaks, filtered_labels)):
    plt.text(pf, fft_magnitude[np.abs(frequencies - pf).argmin()] + 500, 
             f"True: {tz:.5f}", rotation=45, fontsize=8, color='red')

plt.title("Filtered FFT Peaks Matching True Zeros on Critical Line")
plt.xlabel("Frequency (Imaginary Part)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()
plt.show()

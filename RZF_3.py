import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta, mp

# Set precision for Riemann zeta calculations
mp.dps = 20  # High precision for zeta function

# Parameters
n_points = 2**22  # High resolution for FFT
sigma = 0.5       # Gaussian width parameter
alpha = 1.0       # Scaling for Gaussian decay
n_gpus = 4        # Number of GPUs to use
harmonic_count = 20  # Number of harmonic frequencies

# List of true Riemann zeta function zeros on the critical line (first few zeros)
true_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935061,
              37.586178, 40.918719, 43.327073, 48.005150, 49.773832]

# Split computation across GPUs
gpus = [cp.cuda.Device(i) for i in range(n_gpus)]

# Step 1: Generate Harmonic Gaussian Series
def gaussian_series(x, sigma, alpha):
    """Generate a Gaussian-like series."""
    return cp.exp(-alpha * cp.pi * x**2 / sigma**2)

def cosine_component(x, frequency):
    """Cosine modulation for a given frequency."""
    return cp.cos(2 * cp.pi * frequency * x)

# Step 2: Divide Data and Assign to GPUs
x = np.linspace(-50, 50, n_points)  # Imaginary axis values
chunks = np.array_split(x, n_gpus)

results = []

# Step 3: Compute Harmonic Series and FFT on Each GPU
for i, gpu in enumerate(gpus):
    with gpu:  # Set active GPU
        x_chunk = cp.asarray(chunks[i])
        gaussian = gaussian_series(x_chunk, sigma, alpha)
        combined = cp.zeros_like(gaussian)
        
        # Generate harmonic series
        frequencies = cp.linspace(0.1, 10, harmonic_count)
        for freq in frequencies:
            combined += gaussian * cosine_component(x_chunk, freq)
        
        # FFT computation
        fft_result = cp.fft.fftshift(cp.fft.fft(combined))
        fft_magnitude = cp.abs(fft_result)
        results.append(cp.asnumpy(fft_magnitude))

# Step 4: Aggregate FFT Results on CPU
fft_final = np.concatenate(results)
frequencies = np.fft.fftshift(np.fft.fftfreq(n_points, d=(x[1] - x[0])))

# Step 5: Identify Peaks in FFT Magnitude
from scipy.signal import find_peaks

peaks, _ = find_peaks(fft_final, height=np.max(fft_final) * 0.01)
peak_frequencies = frequencies[peaks]

# Step 6: Match Peaks to True Zeros on the Critical Line
def find_nearest(array, value):
    """Find the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

matched_zeros = [find_nearest(true_zeros, freq) for freq in peak_frequencies]

# Step 7: Output Results
print("Detected FFT Peaks and Nearest True Zeros:")
for pf, tz in zip(peak_frequencies, matched_zeros):
    print(f"Detected Peak: {pf:.5f}, Nearest True Zero: {tz:.5f}")

# Step 8: Visualization
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_final, label="FFT Magnitude")
plt.scatter(peak_frequencies, fft_final[peaks], color='red', label="Detected Peaks")
for i, txt in enumerate(matched_zeros):
    plt.text(peak_frequencies[i], fft_final[peaks[i]] + 100, f"True: {txt:.5f}", 
             rotation=45, fontsize=8, color='red')

plt.title("FFT Peaks Compared to True Zeros on Critical Line")
plt.xlabel("Frequency (Imaginary Part)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()
plt.show()

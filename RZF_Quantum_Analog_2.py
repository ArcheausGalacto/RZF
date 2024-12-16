import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_points = 2**22  # Increased resolution for better precision
sigma = 0.5       # Gaussian width parameter
alpha = 1.0       # Scaling for Gaussian decay
n_gpus = 4        # Number of GPUs to use

# Split computation across GPUs
gpus = [cp.cuda.Device(i) for i in range(n_gpus)]

# Step 1: Generate Gaussian Series
def gaussian_series(x, sigma, alpha):
    """Generate a Gaussian-like series."""
    return cp.exp(-alpha * cp.pi * x**2 / sigma**2)

def cosine_component(x, frequency):
    """Cosine modulation for a given frequency."""
    return cp.cos(2 * cp.pi * frequency * x)

# Step 2: Divide Data and Assign to GPUs
x = np.linspace(-50, 50, n_points)  # Imaginary axis values for the critical line
chunks = np.array_split(x, n_gpus)  # Split the data into equal parts

results = []

# Step 3: Compute Gaussian Series and Fourier Transforms on Each GPU
for i, gpu in enumerate(gpus):
    with gpu:  # Explicitly set the active GPU
        x_chunk = cp.asarray(chunks[i])  # Move data chunk to the current GPU
        
        # Compute the Gaussian series
        gaussian = gaussian_series(x_chunk, sigma, alpha)
        
        # Apply cosine modulations to simulate harmonics
        frequencies = cp.linspace(0.1, 10, 10)  # Example harmonic frequencies
        combined = cp.zeros_like(gaussian)
        
        for freq in frequencies:
            combined += gaussian * cosine_component(x_chunk, freq)
        
        # Compute FFT to identify dominant harmonics
        fft_result = cp.fft.fftshift(cp.fft.fft(combined))
        fft_magnitude = cp.abs(fft_result)
        
        # Move result back to CPU
        results.append(cp.asnumpy(fft_magnitude))

# Step 4: Aggregate Results on CPU
fft_final = np.concatenate(results)
frequencies = np.fft.fftshift(np.fft.fftfreq(n_points, d=(x[1] - x[0])))

# Step 5: Refine Zero Detection
threshold = np.percentile(fft_final, 0.5)  # Dynamic threshold for near-zero detection
potential_zeros = np.where(fft_final < threshold)[0]

# Step 6: Output Results with Labels
print("Refined Non-Trivial Zeros (Imaginary Components):")
refined_zeros = []
for idx in potential_zeros:
    refined_zeros.append(frequencies[idx])
    print(f"Zero at frequency: {frequencies[idx]:.5f}")

# Step 7: Visualization with Labels
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_final, label="FFT Magnitude")
plt.scatter(frequencies[potential_zeros], fft_final[potential_zeros], color='red', label="Potential Zeros")

# Add labels to detected zeros
for idx in potential_zeros:
    plt.text(frequencies[idx], fft_final[idx] + 100, f"{frequencies[idx]:.2f}", 
             rotation=45, fontsize=8, color='red')

plt.title("Refined Non-Trivial Zeros along Critical Line")
plt.xlabel("Frequency (Imaginary Part of Zeros)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()

# Step 8: Zoom into Regions Near Zeros for Analysis
plt.figure(figsize=(12, 6))
zoom_region = np.where((frequencies > -8000) & (frequencies < 8000))
plt.plot(frequencies[zoom_region], fft_final[zoom_region], label="Zoomed FFT Magnitude")
plt.scatter(frequencies[potential_zeros], fft_final[potential_zeros], color='red')
plt.title("Zoomed-In View of Zeros")
plt.xlabel("Frequency (Imaginary Part of Zeros)")
plt.ylabel("Magnitude of FFT")
plt.legend()
plt.grid()
plt.show()

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero, mp
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor
import os

# Set precision for RZF calculations
mp.dps = 10  # Precision for Newton-Raphson refinement

# Parameters
initial_harmonics = 50    # Initial number of harmonics
n_points = 2**18          # Start with lower FFT resolution
sigma = 0.5               # Gaussian width parameter
iterations = 5            # Number of refinement iterations
tolerance = 0.1           # Tolerance for matching peaks to true zeros
batch_size = 10           # Number of harmonics processed per batch
increase_factor = 2       # Factor to increase resolution after convergence

# Step 1: Compute True Zeros
def compute_true_zeros(n_zeros):
    """Compute the first n_zeros of the Riemann zeta function on the critical line."""
    return [float(zetazero(n).imag) for n in range(1, n_zeros + 1)]

# Step 2: Generate Gaussian Harmonic Series
def gaussian_series(x, sigma, alpha, beta):
    """Generate a Gaussian series with harmonic cosine modulation."""
    series = cp.zeros_like(x)
    for a, b in zip(alpha, beta):
        series += cp.exp(-a * cp.pi * x**2) * cp.cos(2 * cp.pi * b * x)
    return series

# Step 3: Batched FFT Calculation
def compute_fft_batch(x, alpha, beta_batch):
    """Compute FFT for a batch of harmonics."""
    combined = gaussian_series(x, sigma, alpha, beta_batch)
    fft_result = cp.fft.fftshift(cp.fft.fft(combined))
    fft_magnitude = cp.abs(fft_result)
    return cp.asnumpy(fft_magnitude)

def compute_parallel_fft(x, alpha, beta):
    """Compute FFT across batches of harmonics."""
    fft_result = np.zeros(len(x), dtype=np.float32)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_fft_batch, x, alpha, beta[i:i + batch_size])
                   for i in range(0, len(beta), batch_size)]
        for future in futures:
            fft_result += future.result()
    return fft_result

# Step 4: Error Function
def compute_error(params, x, true_zeros):
    """Compute error between detected peaks and true zeros."""
    n = len(params) // 2
    alpha = params[:n]
    beta = params[n:]
    fft_magnitude = compute_parallel_fft(x, alpha, beta)

    # Convert x to NumPy for fftfreq
    x_cpu = cp.asnumpy(x)
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(x_cpu), d=(x_cpu[1] - x_cpu[0])))

    peaks, _ = find_peaks(fft_magnitude, height=np.percentile(fft_magnitude, 90))
    peak_frequencies = frequencies[peaks]
    error = sum(min(abs(pf - tz) for tz in true_zeros) for pf in peak_frequencies)
    return error


# Step 5: Main Execution
if __name__ == "__main__":
    x = cp.linspace(-50, 50, n_points)
    true_zeros = compute_true_zeros(20)
    
    # Initial parameters for the Gaussian series
    alpha = np.ones(initial_harmonics) * 0.5  # Initial widths
    beta = np.linspace(10, 80, initial_harmonics)  # Initial frequencies
    params = np.concatenate([alpha, beta])  # Combine parameters

    errors = []  # Track total error for visualization

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}: FFT Resolution = {n_points}, Harmonics = {len(beta)}")
        # Optimize parameters
        from scipy.optimize import minimize
        result = minimize(compute_error, params, args=(x, true_zeros), method='L-BFGS-B')
        params = result.x
        alpha = params[:initial_harmonics]
        beta = params[initial_harmonics:]
        errors.append(result.fun)

        print(f"Iteration {iteration + 1}: Total Error = {result.fun:.5f}")

        # Increase resolution if convergence slows
        if iteration > 0 and abs(errors[-1] - errors[-2]) < 1e-2:
            n_points *= increase_factor
            x = cp.linspace(-50, 50, n_points)

    # Final FFT and Peaks
    fft_magnitude = compute_parallel_fft(x, alpha, beta)
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_points, d=(x[1] - x[0])))
    peaks, _ = find_peaks(fft_magnitude, height=np.percentile(fft_magnitude, 90))
    peak_frequencies = frequencies[peaks]

    # Visualization - FFT Peaks and True Zeros
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, fft_magnitude, label="FFT Magnitude", color="blue")
    plt.scatter(peak_frequencies, fft_magnitude[peaks], color='red', label="Harmonic Peaks")
    for tz in true_zeros:
        if np.min(frequencies) <= tz <= np.max(frequencies):
            plt.axvline(x=tz, color='green', linestyle='--', label=f"True Zero: {tz:.5f}")
    plt.title("Refined Gaussian Series Peaks and True Zeros")
    plt.xlabel("Frequency (Imaginary Part)")
    plt.ylabel("Magnitude of FFT")
    plt.legend()
    plt.grid()
    plt.show()

    # Visualization - Error Reduction Over Iterations
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, iterations + 1), errors, marker='o', color="purple", label="Error per Iteration")
    plt.title("Error Reduction Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.legend()
    plt.grid()
    plt.show()

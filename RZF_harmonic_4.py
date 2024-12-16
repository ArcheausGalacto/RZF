import numpy as np
import mpmath as mp
from mpmath import zetazero
import matplotlib.pyplot as plt

# Configuration variables
M = 10       # Number of known zeros to use
N = 500     # Number of basis functions
gamma_min = 10.0
gamma_max = 40.0
num_points = 5000  # Number of points to plot in the range

# Set precision if needed
mp.mp.prec = 100

# Extract the imaginary parts of the first M nontrivial zeros
gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])
f_values = np.ones(M)  # We want peaks (value=1) at each known zero

# Construct the Phi matrix for a cosine basis:
# Phi[i,j] = cos(2*pi*j*gamma_values[i]), j=1,...,N
j_indices = np.arange(1, N+1)
Phi = np.zeros((M, N), dtype=float)
for i, g in enumerate(gamma_values):
    Phi[i,:] = np.cos(2*np.pi*j_indices*g)

# Solve the least-squares problem: Phi * c = f_values
c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

def F(g):
    # Evaluate the constructed function at gamma = g
    return np.sum(c * np.cos(2*np.pi*j_indices*g))

# Create a range of gamma values
g_range = np.linspace(gamma_min, gamma_max, num_points)
F_values = [F(g) for g in g_range]

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(g_range, F_values, label='Constructed Function F(gamma)')

# Mark the known zeros on the plot
for gv in gamma_values:
    plt.axvline(x=gv, color='r', linestyle='--', alpha=0.7)
    plt.text(gv, 1.05, f'γ={gv:.2f}', rotation=90, verticalalignment='bottom', color='red')

plt.title('Harmonic Approximation with Peaks at Known Zeros of Zeta')
plt.xlabel('gamma')
plt.ylabel('F(gamma)')
plt.ylim(0.0, 1.2)  # Adjust y-limits so we can clearly see peaks and labels
plt.legend()
plt.grid(True)
plt.show()

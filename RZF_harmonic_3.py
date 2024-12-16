import numpy as np
import mpmath as mp
from mpmath import zetazero

# Set arbitrary precision if desired
mp.mp.prec = 100

# Number of zeros (data points) to use
M = 5
gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])

# We want the function to form peaks at these zeros
f_values = np.ones(M)

# Number of basis functions
N = 500
j_indices = np.arange(1, N+1)  # frequencies j=1,...,N

# Construct the Phi matrix:
# Phi[i,j] = cos(2*pi*j*gamma_values[i])
# i runs over the known zeros, j runs over the basis functions
Phi = np.zeros((M, N), dtype=float)
for i, g in enumerate(gamma_values):
    Phi[i,:] = np.cos(2*np.pi*j_indices*g)

# Solve the least-squares problem: Phi * c = f_values
c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

def F(g):
    # Evaluate the constructed function at gamma = g
    # F(g) = sum_j c_j * cos(2*pi*j*g)
    return np.sum(c * np.cos(2*np.pi*j_indices*g))

# Print the known zero locations
print("# Known zeros (imag parts):")
for gv in gamma_values:
    print(f"# gamma_zero = {gv}")

print()
print("# Gamma      F(gamma)")
print("#--------------------")

# Evaluate F(gamma) over a chosen range
# Let's pick a range that includes all known zeros and extends beyond them
gamma_min = 10.0
gamma_max = 40.0
step = 0.1

g_range = np.arange(gamma_min, gamma_max+step, step)
for g in g_range:
    val = F(g)
    print(f"{g:15.8f} {val:15.8f}")

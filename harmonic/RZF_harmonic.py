import numpy as np
import mpmath as mp
from mpmath import zetazero

mp.mp.prec = 100

# Number of zeros to use
M = 5
# Extract only the imaginary parts of the zeros
gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])

# Set the target values
f_values = np.ones_like(gamma_values)

N = 500
alpha = 1.0
betas = np.linspace(0.1, 2.0, N)

Phi = np.zeros((M, N), dtype=float)
for i, g in enumerate(gamma_values):
    for j, b in enumerate(betas):
        Phi[i,j] = np.cos(2*np.pi*j*gamma_values[i])

c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

def F(g):
    val = 0.0
    for j, b in enumerate(betas):
        val += c[j]*np.exp(-alpha*np.pi*g**2)*np.cos(2*np.pi*b*g)
    return val

print("Checking F at the known zeros:")
for gv in gamma_values:
    print(f"F({gv}) = {F(gv)}")

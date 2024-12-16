import numpy as np
import mpmath as mp
from mpmath import zetazero
import time

# Set precision for mpmath if desired
mp.mp.prec = 100

# Number of zeros (data points) to use
M = 5
gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])

# Desired target values at these points, say all ones
f_values = np.ones(M)

# We will try multiple values of N to see how the solution behaves
N_values = [50, 100, 500, 1000, 5000]

print("Gamma values:", gamma_values)
print("Target values:", f_values)
print()

for N in N_values:
    start_time = time.time()

    # Construct Phi with a simple cosine basis:
    # Phi[i,j] = cos(2*pi*j*gamma_values[i]), where j runs from 1 to N
    # We'll use j starting from 1 to N to avoid the trivial j=0 case (cos(0)=1)
    j_indices = np.arange(1, N+1)  # frequencies from 1 to N
    # Broadcasting: gamma_values shape: (M,1), j_indices shape: (1,N)
    # This creates a 2D array of gamma_values * j_indices.
    Gamma, J = np.meshgrid(gamma_values, j_indices, indexing='ij')
    Phi = np.cos(2*np.pi*Gamma*J)

    # Solve the least squares problem: Phi * c = f_values
    c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

    # Define the constructed function
    def F(g):
        # F(g) = sum_j c_j * cos(2*pi*j*g)
        # We can use a vectorized approach for speed, but it's simple either way:
        return np.sum(c * np.cos(2*np.pi*j_indices*g))

    # Check how well we matched the target points:
    errors = [abs(F(gv) - fv) for gv, fv in zip(gamma_values, f_values)]
    max_error = max(errors)

    elapsed_time = time.time() - start_time
    print(f"For N={N}: max error at known zeros = {max_error}, time = {elapsed_time:.6f} s")

    # If desired, print the results at each zero:
    for gv in gamma_values:
        print(f"F({gv}) = {F(gv)}")
    print()

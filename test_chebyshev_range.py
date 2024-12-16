import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import chebvander

# Parameters
mp.mp.prec = 200    # High precision for mpmath
T = 50
num_points = 20000
lambda_reg = 1e-6

def xi(s):
    return mp.zeta(s)*0.5*s*(s-1)*mp.pi**(-s/2)*mp.gamma(s/2)

# Sample xi on the line s = 1/2 + it
t_values = [i*(2*T/num_points)-T for i in range(num_points)]
xi_values = [xi(0.5+1j*t) for t in t_values]
xi_real = np.array([float(x.real) for x in xi_values])

def construct_cheb_matrix(t_vals, N, T):
    u = np.array(t_vals)/T
    # Chebvander(u, N-1) returns a matrix of shape (len(u), N)
    # with columns [T0(u), T1(u), ..., T_{N-1}(u)]
    Phi = chebvander(u, N-1)
    return Phi

def regularized_lstsq(Phi, y, lambda_reg):
    A = Phi.T @ Phi + lambda_reg*np.eye(Phi.shape[1])
    b = Phi.T @ y
    c = np.linalg.solve(A, b)
    return c

results = []
best_residual = None
best_N = None

print("N | Residual_norm | Max_error | Mean_error")
print("-------------------------------------------")

for N in range(1, 301):
    Phi = construct_cheb_matrix(t_values, N, T)
    c = regularized_lstsq(Phi, xi_real, lambda_reg)
    approx = Phi @ c
    residual_vector = xi_real - approx
    residual_norm = np.linalg.norm(residual_vector)
    max_error = np.max(np.abs(residual_vector))
    mean_error = np.mean(np.abs(residual_vector))

    results.append((N, residual_norm, max_error, mean_error))

    # Print progress
    print(f"{N:3d} | {residual_norm:.6e} | {max_error:.6e} | {mean_error:.6e}")

    # Track best solution
    if best_residual is None or residual_norm < best_residual:
        best_residual = residual_norm
        best_N = N

# After the loop, print best results
print("\nBest approximation found at N =", best_N)
for r in results:
    if r[0] == best_N:
        print(f"N={best_N}, Residual_norm={r[1]:.6e}, Max_error={r[2]:.6e}, Mean_error={r[3]:.6e}")
        break

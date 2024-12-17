import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

# Optional plotting
# import matplotlib.pyplot as plt

#####################
# Parameters
#####################
mp.mp.prec = 200    # High precision
T = 50              # Interval for t in [-T,T]
num_points = 20000  # Increase for large tests
N = 1000             # Polynomial degree (N terms means degree N-1)
lambda_reg = 1e-9   # Regularization factor (can try 0, 1e-6, 1e-9, etc.)

#####################
# Define xi(s)
#####################
def xi(s):
    return mp.zeta(s)*0.5*s*(s-1)*mp.pi**(-s/2)*mp.gamma(s/2)

#####################
# Sample xi on the line s = 1/2 + it
#####################
t_values = [i*(2*T/num_points)-T for i in range(num_points)]
xi_values = [xi(0.5+1j*t) for t in t_values]
xi_real = np.array([float(x.real) for x in xi_values])

#####################
# Chebyshev basis construction
# Chebyshev polynomials T_k(x) are defined on [-1,1].
# Map t in [-T,T] to u = t/T in [-1,1].
# T_k(u) = Chebyshev polynomial of degree k at u
#####################
def cheb_basis_matrix(t_vals, N):
    # We'll construct a matrix Phi where Phi[i,j] = T_j(u_i), u_i = t_i/T
    # Instead of using a loop, use the Chebyshev polynomials directly.
    # One way: construct each polynomial and evaluate.
    # For large N, this might be slow; we can optimize if needed.
    Phi = np.zeros((len(t_vals), N), dtype=np.float64)
    # Create Chebyshev polynomials once if needed:
    # Alternatively, use np.polynomial.chebyshev.chebvander(u, N-1)
    u_vals = np.array(t_vals)/T
    # chebvander(u, deg) gives a matrix of shape (len(u), deg+1)
    # with T0(u), T1(u), ..., Tdeg(u).
    Phi = np.polynomial.chebyshev.chebvander(u_vals, N-1)
    return Phi

Phi = cheb_basis_matrix(t_values, N)

#####################
# Regularized least squares solve
#####################
def regularized_lstsq(Phi, y, lambda_reg):
    A = Phi.T @ Phi + lambda_reg*np.eye(Phi.shape[1])
    b = Phi.T @ y
    c = np.linalg.solve(A, b)
    return c

c = regularized_lstsq(Phi, xi_real, lambda_reg)

# Compute residuals
approx = Phi @ c
residual_vector = xi_real - approx
residual_norm = np.linalg.norm(residual_vector)
max_error = np.max(np.abs(residual_vector))
mean_error = np.mean(np.abs(residual_vector))

#####################
# Print results
#####################
print("### Chebyshev Approximation ###")
print(f"N = {N}, num_points = {num_points}, T = {T}, lambda_reg = {lambda_reg}")
print(f"First 10 coefficients: {c[:10]}")
print(f"Residual norm (L2): {residual_norm}")
print(f"Max absolute error: {max_error}")
print(f"Mean absolute error: {mean_error}")

#####################
# Optional plotting:
# Uncomment if you want to visualize
#####################
# plt.figure(figsize=(10,6))
# plt.plot(t_values, xi_real, label='xi_real')
# plt.plot(t_values, approx, label='Chebyshev approx')
# plt.title('Chebyshev Approximation of xi(s)')
# plt.xlabel('t')
# plt.ylabel('xi(0.5+it) (real part)')
# plt.legend()
# plt.grid(True)
# plt.show()

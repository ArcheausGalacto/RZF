import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

# Optional: enable plotting if desired
#import matplotlib.pyplot as plt

#####################
# Parameters
#####################
mp.mp.prec = 200   # High precision
T = 50
num_points = 20000  # Increase for more stable results
N = 100             # Number of basis functions
alpha = 0.01        # Gaussian width
lambda_reg = 1e-3   # Regularization factor

#####################
# Define xi(s)
#####################
def xi(s):
    # Using the definition of xi function
    # xi(s) = (1/2)*s*(s-1)*pi^{-s/2}*Gamma(s/2)*zeta(s)
    return mp.zeta(s)*0.5*s*(s-1)*mp.pi**(-s/2)*mp.gamma(s/2)

#####################
# Sample xi on the line s = 1/2 + it
#####################
t_values = [i*(2*T/num_points)-T for i in range(num_points)]
xi_values = [xi(0.5+1j*t) for t in t_values]
xi_real = np.array([float(x.real) for x in xi_values])

#####################
# Basis 1: Gaussian-modulated cosines
#####################
def G_cos(t, n, alpha=0.01):
    return np.exp(-alpha*t*t)*np.cos(2*np.pi*n*t)

#####################
# Basis 2: Hermite-based
# Hermite polynomial H_n(x) from mpmath.hermite
# Hermite function: e^{-alpha*t^2} * H_n(sqrt(alpha)*t)
#####################
def G_hermite(t, n, alpha=0.01):
    # hermite poly from mpmath
    return float(mp.hermite(n, mp.mpf(np.sqrt(alpha)*t)))*np.exp(-alpha*t*t)

#####################
# Basis 3: Chebyshev polynomials on [-T,T]
# Map t to u in [-1,1] by u = t/T
#####################
def G_cheb(t, n):
    u = t/T
    p = Chebyshev([0]*(n) + [1])  # Coeff for x^n
    return p(u)

#####################
# Helper function: Regularized least squares solve
#####################
def regularized_lstsq(Phi, y, lambda_reg):
    # Solve (Phi^T Phi + lambda_reg I)c = Phi^T y
    A = Phi.T @ Phi + lambda_reg*np.eye(Phi.shape[1])
    b = Phi.T @ y
    c = np.linalg.solve(A, b)
    residual = np.linalg.norm(Phi@c - y)
    return c, residual

#####################
# Run approximations and print results
#####################

def run_approximation(name, basis_func):
    # Construct Phi for given basis
    Phi = np.zeros((num_points, N), dtype=np.float64)
    for i, t in enumerate(t_values):
        for n_ in range(N):
            Phi[i, n_] = basis_func(t, n_)

    # Solve regularized system
    c, residual = regularized_lstsq(Phi, xi_real, lambda_reg)

    print(f"\n### {name} Basis ###")
    print(f"First 10 coefficients: {c[:10]}")
    print(f"Residual norm: {residual}")

    # Uncomment to visualize approximation:
    # approx = Phi @ c
    # plt.figure()
    # plt.plot(t_values, xi_real, label="xi_real")
    # plt.plot(t_values, approx, label=f"{name} approx")
    # plt.title(f"{name} Approximation")
    # plt.legend()
    # plt.show()

#####################
# Run All Three Approaches
#####################

# 1. Gaussian-modulated cosines
run_approximation("Gaussian-cosine", lambda t, n: G_cos(t, n+1, alpha)) # n+1 since originally defined from 1..N

# 2. Hermite-based
run_approximation("Hermite", lambda t, n: G_hermite(t, n, alpha))

# 3. Chebyshev polynomials
run_approximation("Chebyshev", G_cheb)

print("\nAll done. Review the printed residuals and coefficients.")

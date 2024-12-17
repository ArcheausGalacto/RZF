import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from sympy import Rational, nsimplify, pi, E, sympify, log, zeta, gamma

# Set precision and parameters
mp.mp.prec = 200
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
    Phi = chebvander(u, N-1)
    return Phi

def regularized_lstsq(Phi, y, lambda_reg):
    A = Phi.T @ Phi + lambda_reg*np.eye(Phi.shape[1])
    b = Phi.T @ y
    c = np.linalg.solve(A, b)
    return c

# Choose N=69 (as previously found good)
N = 69
Phi = construct_cheb_matrix(t_values, N, T)
c = regularized_lstsq(Phi, xi_real, lambda_reg)

symbolic_coeffs = []
known_constants = [pi, E, log(2), zeta(3), gamma(1)]  # Attempt simplification using pi, E

for coeff in c:
    val = sympify(coeff)
    # Attempt to simplify with nsimplify
    simplified_val = nsimplify(val, known_constants)
    # If nsimplify didn't return a non-Float expression, try rational approximation
    if simplified_val.is_Float:
        rational_approx = Rational(coeff).limit_denominator(10**20)
        simplified_val = rational_approx
    symbolic_coeffs.append(simplified_val)

# Print and check for pi, E
for i, sc in enumerate(symbolic_coeffs, start=1):
    sc_str = str(sc)
    print(f"Coefficient c[{i}] =", sc_str)
    if 'pi' in sc_str:
        print(f"  -> pi appears in c[{i}]")
    # Check carefully for Euler's number E
    if 'E' in sc_str and not sc_str.startswith('E'):
        # We ensure this isn't scientific notation by checking free_symbols
        if E in sc.free_symbols:
            print(f"  -> E (Euler's number) appears in c[{i}]")

import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from sympy import Rational, nsimplify, pi, E, gamma, sympify

mp.mp.prec = 200
T = 50
num_points = 20000
lambda_reg = 1e-9

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

# Let's pick the best N found previously (for demonstration, assume N=69)
N = 82
Phi = construct_cheb_matrix(t_values, N, T)
c = regularized_lstsq(Phi, xi_real, lambda_reg)

# c now holds decimal approximations. Let's try to convert them to symbolic form.
# We'll attempt to rationalize them and then use nsimplify with some known constants.
# Note: Many coefficients won't simplify nicely. We'll just demonstrate the approach.

from sympy import nsimplify, Rational, pi, E
# You can add other constants to help nsimplify attempt to find closed-forms:
from sympy import I # imaginary unit if needed

symbolic_coeffs = []
for coeff in c:
    # Start by creating a sympy Float from the coefficient
    val = sympify(coeff)

    # Attempt to nsimplify with known constants pi, E:
    # nsimplify tries to find a closed form using these constants if possible.
    simplified_val = nsimplify(val, [pi, E])
    
    # If that doesn't yield something nice, you could also try rational approximation:
    # Attempt to find a close rational approximation with a large denominator:
    # rational_approx = Rational(str(coeff))  # convert float to rational approx
    # You can limit the denominator size if you like:
    # rational_approx = Rational(coeff).limit_denominator(10**10)

    # Decide which form you prefer: simplified_val or rational_approx.
    # If simplified_val is too complicated or just returns the same decimal,
    # consider rational_approx as a fallback.
    
    # For demonstration, let's store simplified_val:
    symbolic_coeffs.append(simplified_val)

# Print out symbolic coefficients:
for i, sc in enumerate(symbolic_coeffs, start=1):
    print(f"Coefficient c[{i}] =", sc)

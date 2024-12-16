import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import chebvander

# Parameters (Use the same T, num_points, lambda_reg as before)
T = 50
lambda_reg = 1e-6
N = 69  # from the best result found

mp.mp.prec = 200

def xi(s):
    return mp.zeta(s)*0.5*s*(s-1)*mp.pi**(-s/2)*mp.gamma(s/2)

# We'll sample xi again (or we can reuse previously sampled data if available)
num_points = 20000
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

# Construct matrix and solve for N=69
Phi = construct_cheb_matrix(t_values, N, T)
c = regularized_lstsq(Phi, xi_real, lambda_reg)

# The approximation: xi_approx(t) = sum_j c[j]*T_j(t/T)
# Let's create a function xi_approx(t)
def xi_approx(t):
    # Evaluate polynomial at t: we must transform t to u = t/T
    u = t/T
    # Evaluate Chebyshev polynomial with coefficients from c
    # c are the coefficients in the Chebyshev basis (T0(u), T1(u), ...)
    val = 0.0
    for j in range(N):
        # T_j(u) can be computed via np.polynomial.chebyshev.chebval(u, [0]*(j)+[1])
        # but that would be slow. Let's use chebvander once for a single point:
        # Actually, for a single value, let's just use chebval:
        # But let's build a single polynomial object once:
        pass

    # More efficient: create a Chebyshev object from coefficients c
    # Note: The Chebyshev class expects coefficients in order of increasing degree.
    from numpy.polynomial.chebyshev import Chebyshev
    # c is already in that form: c[0]*T0 + c[1]*T1 + ...
    p = Chebyshev(c)
    return p(u)

# Create xi_approx as a closure to avoid re-creating polynomial every time
from numpy.polynomial.chebyshev import Chebyshev
p = Chebyshev(c)
def xi_approx(t):
    u = t/T
    return p(u)

# Now we attempt to find zeros of xi_approx(t) in [-T, T]
# We know xi(s) is entire and has an infinite number of zeros, but let's just see if we can find some here.

# Let's scan for sign changes in xi_approx(t):
sign_changes = []
prev_val = xi_approx(t_values[0])
prev_t = t_values[0]
for t in t_values[1:]:
    val = xi_approx(t)
    if (prev_val > 0 and val < 0) or (prev_val < 0 and val > 0):
        # sign change detected between prev_t and t
        sign_changes.append((prev_t, t))
    prev_val = val
    prev_t = t

print("Sign changes found between points:")
for interval in sign_changes:
    print(interval)

# Use mp.findroot to refine zeros:
# Choose midpoint of each sign change interval or just the interval endpoints
zeros = []
for (t1, t2) in sign_changes:
    # Try findroot
    # findroot requires a continuous function and a good initial guess.
    guess = 0.5*(t1 + t2)
    # We'll define a lambda that uses xi_approx:
    zero_t = mp.findroot(lambda x: xi_approx(x), [t1, t2])
    zeros.append(zero_t)

print("Approximate zeros found:")
for z in zeros:
    print(z, xi_approx(z))

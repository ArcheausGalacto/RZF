import numpy as np
def G_n(t, n, alpha=0.01):
    return np.exp(-alpha*t*t)*np.cos(2*np.pi*n*t)

N = 50  # start with something small
alpha = 0.01
Phi = np.zeros((num_points, N), dtype=np.float64)
for i, t in enumerate(t_values):
    for n_ in range(1, N+1):
        Phi[i, n_-1] = G_n(t, n_, alpha)

xi_real = [float(x.real) for x in xi_values]

c, residuals, rank, svals = np.linalg.lstsq(Phi, xi_real, rcond=None)

print("Estimated coefficients:", c)
print("Residuals:", residuals)
import numpy as np
import mpmath as mp
from mpmath import zetazero

# Configuration: number of zeros, gamma range, and data points
M = 50
gamma_min = 10.0
gamma_max = 40.0
num_points = 2000  # more points for better slope resolution
mp.mp.prec = 100

gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])
f_values = np.ones(M)

def construct_function(alpha, beta_min, beta_max, N, scale):
    """Constructs the function F(g) given the parameters."""
    betas = np.linspace(beta_min, beta_max, N)

    # Build Phi matrix
    M_points = len(gamma_values)
    Phi = np.zeros((M_points, N), dtype=float)
    for i, g in enumerate(gamma_values):
        g_scaled = g * scale
        exp_part = np.exp(-alpha * np.pi * g_scaled**2)
        for j, b in enumerate(betas):
            Phi[i,j] = exp_part * np.cos(2*np.pi*b*g_scaled)

    # Solve least squares
    c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

    def F(g):
        g_scaled = g * scale
        exp_part = np.exp(-alpha * np.pi * g_scaled**2)
        return np.sum(c * exp_part * np.cos(2*np.pi*betas*g_scaled))

    return F

def find_max_slope_points(F):
    """Find local maxima of |F'(g)| within the range [gamma_min, gamma_max]."""
    # Sample F(g) densely
    g_range = np.linspace(gamma_min, gamma_max, num_points)
    # Numerical derivative F'(g)
    F_values = np.array([F(g) for g in g_range])
    # Central difference for derivative
    dg = (gamma_max - gamma_min) / (num_points - 1)
    F_prime = (F_values[2:] - F_values[:-2])/(2*dg)
    g_mid = g_range[1:-1]

    # Find local maxima of |F'(g)|
    # A local maximum of |F'(g)| occurs where F'(g) changes from increasing to decreasing.
    # For simplicity, we can check sign changes in F''(g) or just find peaks in |F'(g)|.
    # Let's find peaks in |F'(g)|:
    abs_fp = np.abs(F_prime)
    max_slope_points = []
    for i in range(1, len(abs_fp)-1):
        if abs_fp[i] > abs_fp[i-1] and abs_fp[i] > abs_fp[i+1]:
            max_slope_points.append(g_mid[i])

    return max_slope_points

def error_metric(max_slope_points, gamma_values):
    """Compute how well max_slope_points align with known zeros.
       Error = sum of min distance from each gamma_value to any max_slope_point."""
    error = 0.0
    for gv in gamma_values:
        dists = [abs(gv - msp) for msp in max_slope_points]
        if dists:
            error += min(dists)
        else:
            # If no max_slope_points found, large penalty
            error += 1000
    return error

# Parameter ranges for a rough grid search based on the insights:
# We'll search a small grid to illustrate the concept.
alpha_values = [0.01, 0.025, 0.05, 0.075, 0.1]
beta_min = 0.83  # keep beta_min constant as suggested
beta_max_values = [5.0, 10.0, 15.0]  # low beta_max as suggested
N_values = [1000, 1500, 2000]  # from suggestions
scale_values = [0.1, 0.15, 0.2, 0.25]

best_error = float('inf')
best_params = None
best_points = None

for alpha in alpha_values:
    for beta_max in beta_max_values:
        for N in N_values:
            for scale in scale_values:
                F = construct_function(alpha, beta_min, beta_max, N, scale)
                max_slope_pts = find_max_slope_points(F)
                err = error_metric(max_slope_pts, gamma_values)
                if err < best_error:
                    best_error = err
                    best_params = (alpha, beta_min, beta_max, N, scale)
                    best_points = max_slope_pts

# Print the best found parameters and error
print("Best Parameters Found:")
print(f"alpha = {best_params[0]}")
print(f"beta_min = {best_params[1]} (fixed)")
print(f"beta_max = {best_params[2]}")
print(f"N = {best_params[3]}")
print(f"scale_factor = {best_params[4]}")
print("Error:", best_error)
print("Max Slope Points Found:", best_points)

# Optional: Construct final F with best parameters and print a short summary
F_best = construct_function(*best_params)
print("\nCheck alignment with known zeros:")
for gv in gamma_values:
    closest = min(best_points, key=lambda x: abs(x - gv)) if best_points else None
    print(f"Zero: {gv:.4f}, Closest Max Slope: {closest:.4f}, Distance: {abs(gv - closest) if closest else 'N/A'}")

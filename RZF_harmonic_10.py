import numpy as np
import mpmath as mp
from mpmath import zetazero

# Configuration: number of zeros, gamma range, and data points
M = 5
gamma_min = 10.0
gamma_max = 40.0
num_points = 2000  # more points for better slope resolution
mp.mp.prec = 100

gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])
f_values = np.ones(M)

def construct_function(alpha, beta_min, beta_max, N, scale):
    """Constructs the function F(g) given the parameters."""
    if beta_min <= 0 or beta_max <= beta_min:
        # Invalid parameter combination, return a dummy function that yields large error
        def F(g):
            return np.inf
        return F

    betas = np.linspace(beta_min, beta_max, N)

    M_points = len(gamma_values)
    Phi = np.zeros((M_points, N), dtype=float)
    for i, g in enumerate(gamma_values):
        g_scaled = g * scale
        exp_part = np.exp(-alpha * np.pi * g_scaled**2)
        for j, b in enumerate(betas):
            Phi[i,j] = exp_part * np.cos(2*np.pi*b*g_scaled)

    c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

    def F(g):
        g_scaled = g * scale
        exp_part = np.exp(-alpha * np.pi * g_scaled**2)
        return np.sum(c * exp_part * np.cos(2*np.pi*betas*g_scaled))

    return F

def find_max_slope_points(F):
    """Find local maxima of |F'(g)| within the range [gamma_min, gamma_max]."""
    g_range = np.linspace(gamma_min, gamma_max, num_points)
    F_values = np.array([F(g) for g in g_range])
    dg = (gamma_max - gamma_min) / (num_points - 1)
    F_prime = (F_values[2:] - F_values[:-2])/(2*dg)
    g_mid = g_range[1:-1]

    abs_fp = np.abs(F_prime)
    max_slope_points = []
    for i in range(1, len(abs_fp)-1):
        if abs_fp[i] > abs_fp[i-1] and abs_fp[i] > abs_fp[i+1]:
            max_slope_points.append(g_mid[i])

    return max_slope_points

def error_metric(max_slope_points, gamma_values):
    """Compute how well max_slope_points align with known zeros."""
    error = 0.0
    for gv in gamma_values:
        dists = [abs(gv - msp) for msp in max_slope_points]
        if dists:
            error += min(dists)
        else:
            error += 1000
    return error

# Initial search centers and widths
alpha_center, alpha_width = 0.05, 0.05
beta_min_center, beta_min_width = 0.83, 0.5
beta_max_center, beta_max_width = 10.0, 100.0
N_center, N_width = 1500, 500
scale_center, scale_width = 0.2, 0.1

# Initial number of steps
alpha_steps = 5
beta_min_steps = 5
beta_max_steps = 10
N_steps = 3
scale_steps = 3

while True:
    # Create value grids
    alpha_values = np.linspace(alpha_center - alpha_width, alpha_center + alpha_width, alpha_steps)
    beta_min_values = np.linspace(beta_min_center - beta_min_width, beta_min_center + beta_min_width, beta_min_steps)
    beta_max_values = np.linspace(beta_max_center - beta_max_width, beta_max_center + beta_max_width, beta_max_steps)
    N_values = np.linspace(N_center - N_width, N_center + N_width, N_steps, dtype=int)
    scale_values = np.linspace(scale_center - scale_width, scale_center + scale_width, scale_steps)

    best_error = float('inf')
    best_params = None
    best_points = None

    # Grid search over the current ranges
    for alpha in alpha_values:
        for beta_min in beta_min_values:
            for beta_max in beta_max_values:
                for N in N_values:
                    for scale in scale_values:
                        # Ensure parameters are sensible
                        if alpha <= 0 or beta_min <= 0 or beta_max <= beta_min or N < 50 or scale <= 0:
                            continue

                        F = construct_function(alpha, beta_min, beta_max, N, scale)
                        if F(gamma_min) == np.inf:  # Invalid function due to parameters
                            continue
                        max_slope_pts = find_max_slope_points(F)
                        err = error_metric(max_slope_pts, gamma_values)
                        if err < best_error:
                            best_error = err
                            best_params = (alpha, beta_min, beta_max, N, scale)
                            best_points = max_slope_pts

    # Print the best found parameters and error
    if best_params is None:
        print("No valid parameters found in this iteration. Try different ranges.")
        break

    print("Best Parameters Found This Iteration:")
    print(f"alpha = {best_params[0]}")
    print(f"beta_min = {best_params[1]}")
    print(f"beta_max = {best_params[2]}")
    print(f"N = {best_params[3]}")
    print(f"scale_factor = {best_params[4]}")
    print("Error:", best_error)
    print("Max Slope Points Found:", best_points)

    # Construct final F with best parameters and summarize alignment
    F_best = construct_function(*best_params)
    print("\nCheck alignment with known zeros:")
    for gv in gamma_values:
        closest = min(best_points, key=lambda x: abs(x - gv)) if best_points else None
        dist = abs(gv - closest) if closest else 'N/A'
        print(f"Zero: {gv:.4f}, Closest Max Slope: {closest}, Distance: {dist}")
    print()

    # Ask user if they want to refine further
    user_input = input("Press Enter to refine further, 'q' to quit: ")
    if user_input.strip().lower() == 'q':
        break

    # Refine ranges around best_params
    alpha_center, beta_min_center, beta_max_center, N_center, scale_center = best_params

    # Reduce search range by half for finer granularity
    alpha_width /= 2.0
    beta_min_width = max(0.01, beta_min_width / 2.0)
    beta_max_width /= 2.0
    N_width = max(10, N_width / 2.0)
    scale_width /= 2.0

    # Increase steps for finer resolution if desired
    alpha_steps = min(alpha_steps + 2, 11)
    beta_min_steps = min(beta_min_steps + 2, 11)
    beta_max_steps = min(beta_max_steps + 1, 10)
    N_steps = min(N_steps + 1, 7)
    scale_steps = min(scale_steps + 1, 7)

    print("Refining search around best parameters...\n")

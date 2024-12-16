import numpy as np

def F(x, c, alpha, beta):
    """
    F(x) = sum_n c_n * exp(-alpha_n * pi * x^2) * cos(2*pi*beta_n*x)
    """
    x = np.array(x, dtype=float)
    total = np.zeros_like(x)
    for cn, an, bn in zip(c, alpha, beta):
        total += cn * np.exp(-an * np.pi * x**2) * np.cos(2 * np.pi * bn * x)
    return total

def F_prime(x, c, alpha, beta):
    """
    Derivative of F(x):
    F'(x) = sum_n c_n * d/dx[exp(-alpha_n*pi*x^2)*cos(2*pi*beta_n*x)]
          = sum_n c_n * exp(-alpha_n*pi*x^2) 
            [-2*pi*beta_n*sin(2*pi*beta_n*x) - 2*alpha_n*pi*x*cos(2*pi*beta_n*x)]
    """
    x = np.array(x, dtype=float)
    total = np.zeros_like(x)
    for cn, an, bn in zip(c, alpha, beta):
        exp_part = np.exp(-an * np.pi * x**2)
        sin_part = np.sin(2 * np.pi * bn * x)
        cos_part = np.cos(2 * np.pi * bn * x)
        term = cn * exp_part * (-2*np.pi*bn*sin_part - 2*an*np.pi*x*cos_part)
        total += term
    return total

def newton_method(x0, F, F_prime, c, alpha, beta, tol=1e-10, max_iter=100):
    """
    Newton's method to refine an approximate zero.

    Parameters:
        x0       : initial guess
        F        : function F(x, c, alpha, beta)
        F_prime  : derivative F'(x, c, alpha, beta)
        c, alpha, beta : parameters for F and F'
        tol      : tolerance for convergence
        max_iter : maximum iterations
    
    Returns:
        (root, iterations_done)
    """
    x = x0
    for i in range(max_iter):
        fx = F(x, c, alpha, beta)
        fpx = F_prime(x, c, alpha, beta)

        # Print iteration details
        print(f"Iteration {i}: x = {x}, F(x) = {fx}, F'(x) = {fpx}")

        if np.abs(fpx) < 1e-15:
            print("Derivative is too close to zero. Stopping.")
            break
        x_new = x - fx/fpx
        if np.abs(x_new - x) < tol and np.abs(F(x_new, c, alpha, beta)) < tol:
            print("Converged!")
            return x_new, i+1
        x = x_new
    print("Reached maximum iterations without full convergence.")
    return x, max_iter

if __name__ == "__main__":
    # Define a more complex function with three terms
    # Chosen somewhat arbitrarily to create a more complex zero structure.
    c = [1.0, 0.5, 0.8]      # Amplitudes
    alpha = [1.0, 2.0, 0.5]   # Gaussian scalings
    beta = [1.0, 0.75, 1.25]  # Frequencies

    # Initial guess, not trivial: let's pick x0 = 0.3
    x0 = 0.3

    # Run Newton's method to find a zero
    root, iterations = newton_method(x0, F, F_prime, c, alpha, beta)
    print(f"Refined zero approximation: {root} after {iterations} iterations")
    print(f"F(root): {F(root, c, alpha, beta)}")

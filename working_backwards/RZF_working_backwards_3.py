import mpmath as mp

mp.mp.prec = 100  # Set precision as needed

def riemann_xi(s):
    """
    Compute the Riemann ξ-function:
    xi(s) = 1/2 * s(s-1)*pi^(-s/2)*Gamma(s/2)*zeta(s)
    """
    return 0.5 * s*(s-1) * (mp.pi**(-s/2)) * mp.gamma(s/2) * mp.zeta(s)

def xi_of_gamma(gamma):
    # Evaluate ξ(1/2 + i*gamma)
    s = 0.5 + gamma*1j
    return riemann_xi(s)

def dxi_of_gamma(gamma):
    # Numerical derivative of ξ w.r.t gamma
    # mp.diff will use a small step to approximate the derivative numerically
    return mp.diff(lambda g: xi_of_gamma(g), gamma)

def newton_method_gamma(gamma0, tol=1e-15, max_iter=100):
    gamma = gamma0
    for i in range(max_iter):
        f = xi_of_gamma(gamma)
        df = dxi_of_gamma(gamma)
        print(f"Iteration {i}: gamma = {gamma}, F(gamma) = {f}, F'(gamma) = {df}")

        if abs(df) < 1e-50:
            print("Derivative too small, stopping.")
            break

        gamma_new = gamma - f/df
        # Check convergence: if the step is small and function value is small
        if abs(gamma_new - gamma) < tol and abs(xi_of_gamma(gamma_new)) < tol:
            print("Converged!")
            return gamma_new
        gamma = gamma_new
    print("Reached max iterations without full convergence.")
    return gamma

if __name__ == "__main__":
    # The first nontrivial zero of ζ(s) is about s = 1/2 + 14.134725141... i
    # We'll start near that value:
    initial_guess = 14.1347

    refined_gamma = newton_method_gamma(initial_guess)
    print("Refined zero approximation (gamma):", refined_gamma)
    final_val = xi_of_gamma(refined_gamma)
    print("ξ(0.5 + i*gamma) at refined zero:", final_val)

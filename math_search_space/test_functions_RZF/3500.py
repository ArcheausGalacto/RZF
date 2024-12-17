import sympy
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, log, pi, E, GoldenRatio, EulerGamma, Catalan, sqrt

# Define the variable
x = symbols('x', real=True)

# Define the function: (sin((EulerGamma / x)) / (sin(3) - (x * x)))
expr_str = "(sin((EulerGamma / x)) / (sin(3) - (x * x)))"
expr = sympy.sympify(expr_str, {
    'x': x,
    'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio,
    'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sin': sin, 'cos': cos, 'log': log, 'sqrt': sqrt
})

f = sympy.lambdify(x, expr, 'numpy')

# Non-trivial RZF zeros imaginary parts
t_values = [
    14.1347251417,
    21.0220396388,
    25.0108575801,
    30.4248761259,
    32.9350615877,
    37.5861781588,
    40.9187190121,
    43.3270732809,
    48.0051508812,
    49.7738324777
]

# Generate data
x_min = 0.1  # start from a small positive number to avoid division by zero at x=0
x_max = 55
num_points = 2000
X = np.linspace(x_min, x_max, num_points)
Y = f(X)

# Filter out non-finite values if they occur
finite_mask = np.isfinite(Y)
X_finite = X[finite_mask]
Y_finite = Y[finite_mask]

if len(Y_finite) == 0:
    print("No valid finite values for the function in the given range.")
    exit()

y_min, y_max = np.min(Y_finite), np.max(Y_finite)
margin = 0.1 * (y_max - y_min if y_max != y_min else 1)
y_min -= margin
y_max += margin

plt.figure(figsize=(12, 6))
plt.plot(X_finite, Y_finite, label=f'y = {expr_str}', color='blue')

# Plot vertical lines at RZF zeros and points
for t in t_values:
    # Calculate f(t)
    val = f(t)
    # Check if finite
    if np.isfinite(val):
        # Vertical line
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)
        # Point at (t, f(t))
        plt.plot(t, val, 'ro')
        plt.text(t, val, f"{val:.6f}", fontsize=8, ha='center', va='bottom', rotation=90)

plt.title('Function vs RZF Zero Approximations')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

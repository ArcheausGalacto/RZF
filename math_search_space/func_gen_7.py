import random
import sympy
from sympy import symbols, sin, cos, log, pi, E, GoldenRatio, EulerGamma, Catalan, sqrt
import matplotlib.pyplot as plt
import numpy as np

def random_operand():
    # Fundamental constants and selected prime integers
    operands = [
        'x', 'x', 'x',
        '2', '3', '5', '7', '11',   # prime integers
        'pi', 'E', 'GoldenRatio', 'EulerGamma', 'Catalan', 'sqrt(2)'
    ]
    return random.choice(operands)

def random_operation():
    operations = [
        ('+', 2),
        ('-', 2),
        ('*', 2),
        ('/', 2),
        ('sin', 1),
        ('cos', 1),
        ('log', 1)
    ]
    return random.choice(operations)

def is_pure_constant(expr_str):
    x = symbols('x', real=True)
    try:
        val = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio, 
                                       'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sqrt': sqrt})
        return val.free_symbols == set()
    except:
        return False

def is_pure_integer(expr_str):
    x = symbols('x', real=True)
    try:
        val = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio, 
                                       'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sqrt': sqrt})
        return val.is_Integer
    except:
        return False

def build_random_expression(max_ops=3):
    if max_ops == 0:
        return random_operand()
    
    while True:
        op, arity = random_operation()
        if arity == 2:
            left = build_random_expression(max_ops - 1)
            right = build_random_expression(max_ops - 1)

            # Avoid trivial addition/subtraction of constants
            if op in ['+', '-'] and is_pure_constant(left) and is_pure_constant(right):
                continue

            # Avoid trivial multiplication of integers
            if op == '*' and is_pure_integer(left) and is_pure_integer(right):
                continue

            return f'({left} {op} {right})'
        else:
            sub = build_random_expression(max_ops - 1)
            return f'{op}({sub})'

def is_constant_function(expr):
    x = symbols('x', real=True)
    deriv = sympy.diff(expr, x)
    return sympy.simplify(deriv) == 0

def is_linear_function(expr):
    x = symbols('x', real=True)
    first_deriv = sympy.simplify(sympy.diff(expr, x))
    second_deriv = sympy.simplify(sympy.diff(expr, (x, 2)))
    
    # If second derivative is zero and first derivative is nonzero constant => linear
    if second_deriv == 0:
        if first_deriv.free_symbols == set() and first_deriv != 0:
            return True
    return False

def is_just_unary_function(expr):
    # Check if top-level expr is just sin(...), cos(...), or log(...).
    if expr.func in [sin, cos, log]:
        return True
    return False

def matches_rzf_zeros(expr, tolerance=0.1):
    # Hardcode the first 10 non-trivial zeros (imag parts)
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

    # We must check if for these t-values, f(t) ~ 0
    x = symbols('x', real=True)
    f = sympy.lambdify(x, expr, 'numpy')
    for t in t_values:
        val = f(t)
        if not np.isfinite(val) or abs(val) > tolerance:
            return False
    return True

def generate_non_constant_and_non_linear_expression(max_ops=3):
    x = symbols('x', real=True)
    attempts = 0
    while True:
        attempts += 1
        expr_str = build_random_expression(max_ops)
        expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio, 
                                        'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sin': sin, 'cos': cos, 'log': log, 'sqrt': sqrt})
        
        # Filter out constant and linear functions, and top-level unary only
        if is_constant_function(expr):
            continue
        if is_linear_function(expr):
            continue
        if is_just_unary_function(expr):
            continue
        # Check RZF zeros intersection
        if matches_rzf_zeros(expr):
            print(f"Found a function matching RZF zeros after {attempts} attempts.")
            return expr_str, expr

def plot_expression(expr_str):
    x = symbols('x', real=True)
    expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio,
                                    'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sin': sin, 'cos': cos, 'log': log, 'sqrt': sqrt})
    f = sympy.lambdify(x, expr, 'numpy')
    
    X = np.linspace(-10, 10, 400)
    Y = f(X)

    # Filter out invalid values
    finite_mask = np.isfinite(Y)
    X_finite = X[finite_mask]
    Y_finite = Y[finite_mask]

    if len(Y_finite) == 0:
        print("No valid finite values for the function in the given range.")
        return

    # Adjust plot limits to fit the data
    y_min, y_max = np.min(Y_finite), np.max(Y_finite)
    margin = 0.1 * (y_max - y_min if y_max != y_min else 1)
    y_min -= margin
    y_max += margin

    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, label=f'y = {expr_str}')
    plt.title('Random Expression Fitting RZF Zeros (Conceptual)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.show()

if __name__ == "__main__":
    expr_str, expr = generate_non_constant_and_non_linear_expression(max_ops=3)
    print("Generated non-constant, non-linear equation matching RZF zeros:", expr_str)
    plot_expression(expr_str)

import random
import sympy
from sympy import symbols, sin, cos, log, pi, E, GoldenRatio, EulerGamma, Catalan, sqrt, zoo, oo
import matplotlib.pyplot as plt
import numpy as np

def contains_problematic_values(expr):
    # Check if the expression contains infinity (oo) or complex infinity (zoo)
    return expr.has(oo) or expr.has(zoo)

def random_operand():
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
    val = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio,
                                   'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sqrt': sqrt})
    return val.free_symbols == set()

def is_pure_integer(expr_str):
    x = symbols('x', real=True)
    val = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio,
                                   'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sqrt': sqrt})
    return val.is_Integer

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
    if second_deriv == 0 and first_deriv.free_symbols == set() and first_deriv != 0:
        return True
    return False

def is_just_unary_function(expr):
    if expr.func in [sin, cos, log]:
        return True
    return False

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def score_function(expr, tolerance=0.1, epsilon=0.001):
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

    x = symbols('x', real=True)
    f = sympy.lambdify(x, expr, 'numpy')

    total_abs = 0.0
    all_sign_changes = True

    for t in t_values:
        val_center = f(t)
        val_left = f(t - epsilon)
        val_right = f(t + epsilon)

        if not (np.isfinite(val_center) and np.isfinite(val_left) and np.isfinite(val_right)):
            return None, False

        total_abs += abs(val_center)

        if abs(val_center) < tolerance:
            if sign(val_left) * sign(val_right) >= 0:
                all_sign_changes = False
        else:
            all_sign_changes = False

    return total_abs, all_sign_changes

def generate_non_constant_and_non_linear_expression(max_ops=3):
    x = symbols('x', real=True)
    attempts = 0
    best_score = float('inf')
    best_expr_str = None

    while True:
        attempts += 1
        expr_str = build_random_expression(max_ops)
        expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio,
                                        'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sin': sin, 'cos': cos, 'log': log, 'sqrt': sqrt})
        
        # Check for problematic values
        if contains_problematic_values(expr):
            continue

        if is_constant_function(expr):
            continue
        if is_linear_function(expr):
            continue
        if is_just_unary_function(expr):
            continue

        current_score, all_sign_changes = score_function(expr)
        if current_score is not None:
            if current_score < best_score:
                best_score = current_score
                best_expr_str = expr_str

            if all_sign_changes:
                print(f"Found a perfect zero-crossing function after {attempts} attempts: {expr_str}")
                return expr_str, expr

        if attempts % 500 == 0:
            if best_expr_str is not None:
                print(f"After {attempts} attempts, best function so far: {best_expr_str} with score {best_score}")
            else:
                print(f"After {attempts} attempts, no viable function found yet.")

def plot_expression(expr_str):
    x = symbols('x', real=True)
    expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'GoldenRatio': GoldenRatio,
                                    'EulerGamma': EulerGamma, 'Catalan': Catalan, 'sin': sin, 'cos': cos, 'log': log, 'sqrt': sqrt})
    f = sympy.lambdify(x, expr, 'numpy')
    
    X = np.linspace(-10, 10, 400)
    Y = f(X)

    finite_mask = np.isfinite(Y)
    X_finite = X[finite_mask]
    Y_finite = Y[finite_mask]

    if len(Y_finite) == 0:
        print("No valid finite values for the function in the given range.")
        return

    y_min, y_max = np.min(Y_finite), np.max(Y_finite)
    margin = 0.1 * (y_max - y_min if y_max != y_min else 1)
    y_min -= margin
    y_max += margin

    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, label=f'y = {expr_str}')
    plt.title('Random Expression Attempting RZF Zero-Crossing')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.show()

if __name__ == "__main__":
    expr_str, expr = generate_non_constant_and_non_linear_expression(max_ops=3)
    print("Generated function:", expr_str)
    plot_expression(expr_str)

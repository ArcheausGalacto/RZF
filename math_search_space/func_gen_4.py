import random
import sympy
from sympy import symbols, sin, cos, log, pi, E, Integer
import matplotlib.pyplot as plt
import numpy as np
import math

def random_operand():
    # Increase the likelihood of 'x' to avoid constants only
    operands = [
        'x', 'x', 'x',
        '2', '3', '5', '7', '11', '17', '23',
        'pi', 'E'
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
        val = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E})
        # Pure constant if there are no free symbols:
        return val.free_symbols == set()
    except:
        return False

def is_pure_integer(expr_str):
    x = symbols('x', real=True)
    try:
        val = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E})
        # Check if val is an integer (e.g., type Integer or can be simplified to integer)
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
    
    # If second derivative is zero and first derivative is a nonzero constant => linear
    if second_deriv == 0:
        if first_deriv.free_symbols == set() and first_deriv != 0:
            return True
    return False

def generate_non_constant_and_non_linear_expression(max_ops=3):
    x = symbols('x', real=True)
    while True:
        expr_str = build_random_expression(max_ops)
        expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'sin': sin, 'cos': cos, 'log': log})
        
        if not is_constant_function(expr) and not is_linear_function(expr):
            return expr_str, expr

def plot_expression(expr_str):
    x = symbols('x', real=True)
    expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'sin': sin, 'cos': cos, 'log': log})
    f = sympy.lambdify(x, expr, 'numpy')
    
    # We'll try a base range of -10 to 10
    X = np.linspace(-10, 10, 400)
    Y = f(X)

    # Filter out invalid (NaN/Inf) values
    finite_mask = np.isfinite(Y)
    X_finite = X[finite_mask]
    Y_finite = Y[finite_mask]

    if len(Y_finite) == 0:
        # If no finite values, just skip plotting (or handle differently)
        print("No valid finite values for the function in the given range.")
        return

    # Adjust plot limits to fit the data
    y_min, y_max = np.min(Y_finite), np.max(Y_finite)
    # Add a small margin
    margin = 0.1 * (y_max - y_min if y_max != y_min else 1)
    y_min -= margin
    y_max += margin

    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, label=f'y = {expr_str}')
    plt.title('Randomly Generated Non-Constant, Non-Linear Expression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.ylim(y_min, y_max)  # Fit to data
    plt.show()

if __name__ == "__main__":
    expr_str, expr = generate_non_constant_and_non_linear_expression(max_ops=3)
    print("Generated non-constant, non-linear equation:", expr_str)
    plot_expression(expr_str)

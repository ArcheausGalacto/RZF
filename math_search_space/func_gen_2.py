import random
import sympy
from sympy import symbols, sin, cos, log, pi, E
import matplotlib.pyplot as plt
import numpy as np

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
    # This function checks if the expr_str is a single token that's an integer or a known constant
    # and not 'x'.
    # We'll consider pi, E, and integers as constants.
    # Return True if expr_str is one of these or can be sympified to a number directly.
    try:
        val = sympy.sympify(expr_str, {'x': symbols('x', real=True), 'pi': pi, 'E': E})
        # Check if val is free of x and is a number
        return val.free_symbols == set()  # no x present => pure constant
    except:
        return False

def build_random_expression(max_ops=3):
    if max_ops == 0:
        return random_operand()
    
    # We try to ensure we don't create trivial + or - on pure constants
    while True:
        op, arity = random_operation()
        
        if arity == 2:
            left = build_random_expression(max_ops - 1)
            right = build_random_expression(max_ops - 1)
            
            # Check for trivial addition or subtraction of constants
            if op in ['+', '-']:
                if is_pure_constant(left) and is_pure_constant(right):
                    # This creates a trivial step, retry generating a different structure
                    continue
            
            return f'({left} {op} {right})'
        else:
            sub = build_random_expression(max_ops - 1)
            return f'{op}({sub})'

def is_constant_function(expr):
    x = symbols('x', real=True)
    deriv = sympy.diff(expr, x)
    return sympy.simplify(deriv) == 0

def generate_non_constant_expression(max_ops=3):
    x = symbols('x', real=True)
    while True:
        expr_str = build_random_expression(max_ops)
        expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'sin': sin, 'cos': cos, 'log': log})
        if not is_constant_function(expr):
            return expr_str, expr

def plot_expression(expr_str):
    x = symbols('x', real=True)
    expr = sympy.sympify(expr_str, {'x': x, 'pi': pi, 'E': E, 'sin': sin, 'cos': cos, 'log': log})
    f = sympy.lambdify(x, expr, 'numpy')
    
    X = np.linspace(-10, 10, 400)
    Y = f(X)
    if np.isscalar(Y):
        Y = np.full_like(X, Y)
    
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, label=f'y = {expr_str}')
    plt.title('Randomly Generated Non-Trivial, Non-Constant Expression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    expr_str, expr = generate_non_constant_expression(max_ops=3)
    print("Generated non-constant, non-trivial equation:", expr_str)
    plot_expression(expr_str)

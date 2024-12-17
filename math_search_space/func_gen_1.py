import random
import sympy
from sympy import symbols, sin, cos, log, pi, E
import matplotlib.pyplot as plt
import numpy as np

def random_operand():
    # Increase the likelihood of 'x' to reduce chances of constant functions
    operands = [
        'x', 'x', 'x', 
        '2', '3', '5', '7', '11', '17', '23',
        'pi', 'E'
    ]
    return random.choice(operands)

def random_operation():
    # Include basic binary and unary operations
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

def build_random_expression(max_ops=3):
    if max_ops == 0:
        return random_operand()
    
    op, arity = random_operation()
    
    if arity == 2:
        left = build_random_expression(max_ops - 1)
        right = build_random_expression(max_ops - 1)
        return f'({left} {op} {right})'
    else:
        sub = build_random_expression(max_ops - 1)
        return f'{op}({sub})'

def is_constant_function(expr):
    x = symbols('x', real=True)
    # Differentiate the expression wrt x
    deriv = sympy.diff(expr, x)
    # If the simplified derivative is 0, expression is constant wrt x
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
    # If Y is scalar, make it a matching array
    if np.isscalar(Y):
        Y = np.full_like(X, Y)
    
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, label=f'y = {expr_str}')
    plt.title('Randomly Generated Non-Constant Expression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    expr_str, expr = generate_non_constant_expression(max_ops=3)
    print("Generated non-constant equation:", expr_str)
    plot_expression(expr_str)

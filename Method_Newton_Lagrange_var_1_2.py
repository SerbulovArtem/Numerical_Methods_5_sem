import math

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f_x(n):
    ans = []
    for x in range(n):
        ans.append(x)
    return ans

def f1_nodes(x, n) -> list:
    ans = []
    for i in range(n):
        ans.append(1 / (1 + 25 * (x[i]**2)))
    return ans


def f1(x) -> float:
    return 1 / (1 + 25 * (x ** 2))


def f2_nodes(x, n) -> list:
    ans = []
    for i in range(n):
        ans.append(math.log(x[i] + 2))
    return ans


def f2(x) -> float:
    return math.log(x + 2)


def f3_nodes(x, n) -> list:
    ans = []
    for i in range(n):
        ans.append(x[i]**5 + 3 * x[i] ** 4 - 2 * x[i] ** 2 + x[i] + 3)
    return ans


def f3(x) -> float:
    return x**5 + 3 * x ** 4 - 2 * x ** 2 + x + 3


def interpolation_nodes(x, n, inter_func):
    ans = []
    for i in range(n):
        ans.append(inter_func(x[i]))
    return ans


def equal_nodes(n, a: float = -1, b: float = 1) -> list:
    ans = []
    for i in range(n):
        x = a + ((b - a) / n) * i
        ans.append(x)
    return ans


def chebyshev_nodes(n, a: float = -1,  b: float = 1) -> list:
    ans = []
    for i in range(n):
        x = ((a + b) / 2) + (((b - a) / 2) * math.cos(((2 * i + 1) / (2*(n + 1))) * math.pi))
        ans.append(x)
    return ans


def lagrange_polynomial(xi, yi) -> tuple:
    x = sp.symbols('x')

    lagrange_poly = 0
    for i in range(len(xi)):
        term = yi[i]
        for j in range(len(xi)):
            if i != j:
                term *= (x - xi[j]) / (xi[i] - xi[j])
        lagrange_poly += term

    simplified_poly = sp.simplify(lagrange_poly)
    return simplified_poly, x


def divided_differences(x, y):
    n = len(x)
    f = [[0] * n for _ in range(n)]

    for i in range(n):
        f[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            f[i][j] = (f[i + 1][j - 1] - f[i][j - 1]) / (x[i + j] - x[i])

    return f


def newton_interpolation(x, y):
    n = len(x)
    f = divided_differences(x, y)
    x_var = sp.symbols('x')
    p = f[0][0]

    for i in range(1, n):
        term = f[0][i]
        for j in range(i):
            term *= (x_var - x[j])
        p += term

    return sp.simplify(p), x_var


def plot(func, interpolation_func_lag, interpolation_func_new, inter_plot3, inter_plot4, n, f, a, b) -> plt:
    plt.switch_backend('TkAgg')
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    figure, axis = plt.subplots(2, figsize=(1920 * px, 1080 * px))
    x = np.linspace(a, b, num=n)

    axis[0].plot(x, func, color='red', label=f'Function f{f}', linewidth=3.0)
    axis[0].plot(x, interpolation_func_lag, color='purple', label=f'Lagrange interpolation function f{f}', linewidth=3.0)
    axis[0].plot(x, interpolation_func_new, color='black', label=f'Newton interpolation function f{f}', linewidth=1.5)

    axis[0].set_title("Lagrange And Newton Method Equal Nodes", fontsize=10)
    axis[0].set_xlabel('Value', fontsize=10)
    axis[0].set_ylabel('Function Value', fontsize=10)
    axis[0].legend()
    axis[0].grid(True)

    axis[1].plot(x, func, color='red', label=f'Function f{f}', linewidth=3.0)
    axis[1].plot(x, inter_plot3, color='purple', label=f'Lagrange interpolation function f{f}',
                 linewidth=3.0)
    axis[1].plot(x, inter_plot4, color='black', label=f'Newton interpolation function f{f}', linewidth=1.5)

    axis[1].set_title("Lagrange And Newton Method Chebyshev Nodes", fontsize=10)
    axis[1].set_xlabel('Value', fontsize=10)
    axis[1].set_ylabel('Function Value', fontsize=10)
    axis[1].legend()
    axis[1].grid(True)

    plt.show()
    pass


def func_plot(func, n, a, b):
    ans = []
    for x in np.linspace(a, b, num=n):
        ans.append(func(x))
    return ans


def Var1_2() -> None:
    function_nodes = [f1_nodes, f2_nodes, f3_nodes]
    functions_plot = [f1, f2, f3]
    functions = [1, 2, 3]
    for func_node, funct_plot, functions in zip(function_nodes, functions_plot, functions):
        n = int(input("Enter n: "))
        a, b = tuple(map(float, input("Enter a and b: ").split(' ')))
        xi = equal_nodes(n, a, b)
        xi_c = chebyshev_nodes(n, a, b)
        yi = func_node(xi, n)
        yi_c = func_node(xi_c, n)

        x_ = float(input(f"Enter x in [{a}:{b}] for error: "))

        simplified_poly1, x1 = lagrange_polynomial(xi, yi)
        simplified_poly2, x2 = newton_interpolation(xi, yi)
        simplified_poly3, x3 = lagrange_polynomial(xi_c, yi_c)
        simplified_poly4, x4 = newton_interpolation(xi_c, yi_c)
        print(f"Lagrange polynomial equal nodes: {simplified_poly1}")
        print(f"Newton polynomial equal nodes: {simplified_poly2}")
        print(f"Lagrange polynomial chebyshev nodes: {simplified_poly3}")
        print(f"Newton polynomial chebyshev nodes: {simplified_poly4}")
        interpolation_function1 = sp.lambdify(x1, simplified_poly1, 'numpy')
        interpolation_function2 = sp.lambdify(x2, simplified_poly2, 'numpy')
        interpolation_function3 = sp.lambdify(x3, simplified_poly3, 'numpy')
        interpolation_function4 = sp.lambdify(x4, simplified_poly4, 'numpy')
        print(f"Error Lagrange: f(x) - p(x) equal nodes = {abs(funct_plot(x_) - interpolation_function1(x_))}")
        print(f"Error Newton: f(x) - p(x) equal nodes = {abs(funct_plot(x_) - interpolation_function2(x_))}")
        print(f"Error Lagrange: f(x) - p(x) chebyshev nodes = {abs(funct_plot(x_) - interpolation_function3(x_))}")
        print(f"Error Newton: f(x) - p(x) chebyshev nodes = {abs(funct_plot(x_) - interpolation_function4(x_))}")

        n_ = 500
        f1_plot = func_plot(funct_plot, n_, a, b)
        inter_plot1 = func_plot(interpolation_function1, n_, a, b)
        inter_plot2 = func_plot(interpolation_function2, n_, a, b)
        inter_plot3 = func_plot(interpolation_function3, n_, a, b)
        inter_plot4 = func_plot(interpolation_function4, n_, a, b)
        plot(f1_plot, inter_plot1, inter_plot2, inter_plot3, inter_plot4, n_, functions, a, b)


Var1_2()

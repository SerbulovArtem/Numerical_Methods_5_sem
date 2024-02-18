import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f1_nodes(x, n) -> list:
    ans = []
    for i in range(2 * n + 1):
        ans.append(math.exp(math.cos(x[i]) + math.sin(x[i])))
    return ans


def f1(x) -> float:
    return math.exp(math.cos(x) + math.sin(x))


def f2_nodes(x, n) -> list:
    ans = []
    for i in range(2 * n + 1):
        ans.append(3 * math.cos(15 * x[i]))
    return ans


def f2(x) -> float:
    return 3 * math.cos(15 * x)


def equal_nodes_odd(n, a: float = 0, b: float = 2 * math.pi) -> list:   # +++
    ans = []
    h = (b - a) / (2 * n + 1)
    for i in range(2 * n + 1):
        x = a + h * i
        ans.append(x)
    return ans


def trigonometric_polynomial(a, b, xi, yi) -> tuple:
    n = len(xi) // 2
    a_o = [0] * (n + 1)
    b_o = [0] * (n + 1)
    for i in range(n + 1):
        for j in range(2 * n + 1):
            a_o[i] += 2 / (2 * n + 1) * yi[j] * math.cos((b - a) * j * i / (2 * n + 1))
    for i in range(1, n + 1):
        for j in range(2 * n + 1):
            b_o[i] += 2 / (2 * n + 1) * yi[j] * math.sin((b - a) * j * i / (2 * n + 1))

    x = sp.symbols('x')

    term = a_o[0] / 2
    for i in range(1, n + 1):
        term += a_o[i] * sp.cos(i * x) + b_o[i] * sp.sin(i * x)
    trigonometric_poly = term

    simplified_poly = sp.simplify(trigonometric_poly)

    return simplified_poly, x


def plot(func, interpolation_func_trig, xi, yi, n, f, a, b) -> plt:
    plt.switch_backend('TkAgg')
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    figure = plt.figure(figsize=(1920 * px, 1080 * px))
    x = np.linspace(a, b, num=n)

    plt.plot(x, func, color='red', label=f'Function f{f}', linewidth=3.0)
    plt.plot(x, interpolation_func_trig, color='purple', label=f'Trigonometric interpolation function f{f}'
             , linewidth=3.0)
    plt.scatter(xi, yi, label='Equidistant Nodes', s=100, c=xi, cmap='viridis')

    plt.title("Trigonometric Method Equidistant Odd Nodes", fontsize=10)
    plt.xlabel('Value x*pi', fontsize=10)
    plt.ylabel('Function Value', fontsize=10)
    plt.colorbar()
    plt.legend()
    plt.grid(True)

    plt.show()
    pass


def func_plot(func, n, a, b):
    ans = []
    for x in np.linspace(a, b, num=n):
        ans.append(func(x))
    return ans


def var1() -> None:
    function_nodes = [f1_nodes, f2_nodes]
    functions_plot = [f1, f2]
    functions = [1, 2]
    for func_node, funct_plot, functions in zip(function_nodes, functions_plot, functions):
        n = int(input("Enter Power n: "))
        a, b = tuple(map(float, input("Enter a and b in Pi radians: ").split(' ')))
        a *= math.pi
        b *= math.pi
        xi = equal_nodes_odd(n, a, b)
        yi = func_node(xi, n)

        x_ = float(input(f"Enter x*pi in [{a}:{b}] for error: ")) * math.pi

        simplified_poly1, x1 = trigonometric_polynomial(a, b, xi, yi)
        print(f"Trigonometric polynomial equidistant odd nodes: {simplified_poly1}")
        interpolation_function1 = sp.lambdify(x1, simplified_poly1, 'numpy')
        print(f"Error Trigonometric: f(x) - p(x) equidistant odd nodes = {abs(funct_plot(x_) - interpolation_function1(x_))}")

        n_ = 500
        f1_plot = func_plot(funct_plot, n_, a, b)
        inter_plot1 = func_plot(interpolation_function1, n_, a, b)
        plot(f1_plot, inter_plot1, xi, yi,  n_, functions, a, b)


if __name__ == "__main__":
    var1()

import math
import numpy as np
import matplotlib.pyplot as plt


def f_linear(x) -> float:
    if abs(x) <= 1:
        return 1 - abs(x)
    else:
        return 0


def f_quadratic(x) -> float:
    if abs(x) <= 0.5:
        return (1.0/2) * ((2 - ((abs(x) - 0.5) ** 2)) - ((abs(x) + 0.5) ** 2))
    elif abs(x) <= 1.5:
        return (1.0/2) * ((abs(x) - 1.5) ** 2)
    else:
        return 0


def f_cubic(x) -> float:
    if abs(x) <= 1:
        return (1.0/6) * (((2 - abs(x)) ** 3) - (4 * ((1 - abs(x)) ** 3)))
    elif abs(x) <= 2:
        return (1.0/6) * ((2 - (abs(x))) ** 3)
    else:
        return 0


def equal_nodes(n, a: float = 0, b: float = 2) -> list:
    ans = []
    h = (b - a) / n
    for i in range(n):
        x = a + h * i
        ans.append(x)
    return ans


def func_plot_var_1(func, n, a, b, k):
    ans = []
    x = np.linspace(a, b, num=n)
    for i in range(n):
        ans.append(func(x[i] - k))
    return ans


def var1() -> None:
    functions_plot = [f_linear, f_quadratic, f_cubic]
    for funct_plot in functions_plot:
        n = int(input("Enter n:"))
        a, b = tuple(map(float, input("Enter a and b: ").split(' ')))
        xk = equal_nodes(1000, a, b)

        plt.switch_backend('TkAgg')
        px = 1 / plt.rcParams['figure.dpi']
        figure = plt.figure(figsize=(1920 * px, 1080 * px))

        # f1_plot = func_plot(funct_plot, n, a, b, 0)
        # plt.plot(xk, f1_plot)
        # plt.show()

        k_array = np.linspace(a, b, n)
        for k in k_array:
            f1_plot = func_plot_var_1(funct_plot, 1000, a, b, k)
            plt.plot(xk, f1_plot)
        plt.show()


def func_plot_var_2_1(func, f_i, n, a, b, k) -> list:
    ans = []
    x = np.linspace(a, b, num=n)
    for i in range(n):
        ans.append((func(x[i] - k)) * f_i(x[i]))
    return ans


def f_arbitrary_1(x) -> float:
    return x ** 2 + x + 3


def f_f_arbitrary_nodes_1(x, n) -> list:
    ans = []
    for i in range(n):
        ans.append(x[i] ** 2 + x[i] + 3)
    return ans


def f_arbitrary_2(x) -> float:
    return math.cos(x)


def f_f_arbitrary_nodes_2(x, n) -> list:
    ans = []
    for i in range(n):
        ans.append(math.cos(x[i]))
    return ans


def var2_1() -> None:
    functions_plot = [(f_linear, f_arbitrary_1, f_f_arbitrary_nodes_1), (f_linear, f_arbitrary_2, f_f_arbitrary_nodes_2),
                      (f_linear, f_arbitrary_3, f_f_arbitrary_nodes_3)]
    for funct_plot, f_i, f_nodes in functions_plot:
        n = int(input("Enter n:"))
        _n = 1000
        a, b = tuple(map(float, input("Enter a and b: ").split(' ')))

        plt.switch_backend('TkAgg')
        px = 1 / plt.rcParams['figure.dpi']
        figure = plt.figure(figsize=(1920 * px, 1080 * px))

        x_our_func = equal_nodes(500, a, b)
        y_our_func = f_nodes(x_our_func, 500)
        plt.plot(x_our_func, y_our_func, color='red', label='Function f', linewidth=5.0)

        ### Liniar beta-spline interpolation

        ab = np.linspace(a, b, num=n + 1)
        ab_y = [f_i(ab[i]) for i in range(n + 1)]

        for k in range(n):
            f1_plot_linear = []
            x_f1_plot_linear = np.linspace(ab[k], ab[k + 1], num=n + 1)
            x = np.linspace(ab[k], ab[k + 1], num=n + 1)
            for j in range(n + 1):
                sum = 0
                for i in range(n + 1):
                    sum += (funct_plot(j - i)) * f_i(x[i])
                f1_plot_linear.append(sum)
            plt.plot(x_f1_plot_linear, f1_plot_linear, color='blue', linewidth=3.0)

        plt.scatter(ab, ab_y, marker='o', c='purple', s=50, zorder=2)
        plt.legend()
        plt.show()


def derivative(f, a, h=1e-6):
    return (f(a) - f(a - h))/h


def var_2_2() -> None:
    functions_plot = [(f_cubic, f_arbitrary_1, f_f_arbitrary_nodes_1)]
    for funct_plot, f_i, f_nodes in functions_plot:
        n = int(input("Enter n:"))
        _n = 1000
        a, b = tuple(map(float, input("Enter a and b: ").split(' ')))

        plt.switch_backend('TkAgg')
        px = 1 / plt.rcParams['figure.dpi']
        figure = plt.figure(figsize=(1920 * px, 1080 * px))

        x_our_func = equal_nodes(500, a, b)
        y_our_func = f_nodes(x_our_func, 500)
        plt.plot(x_our_func, y_our_func, color='red', label='Function f', linewidth=5.0)



        # Cubic beta-spline interpolation

        _n = 100 * n
        ab = np.linspace(a, b, num=n + 1)
        ab_y = [f_i(ab[i]) for i in range(n + 1)]
        #print(ab)

        ### Tridiagonal Matrix

        a = np.array([-1 / 2] + [0] + [1 / 2] + [0 for x in range(_n)], dtype=float) ##

        void_matrix = np.empty((0, _n + 3), dtype=float) ##

        b_matrix = []
        for i in range(_n + 1): ##
            b = np.array([0 for x in range(i)] + [1 / 6] + [2 / 3] + [1 / 6] + [0 for x in range(_n - i)], ##
                         dtype=float)
            b_matrix.append(b)

        c = np.array([0 for x in range(_n)] + [-1 / 2] + [0] + [1 / 2], dtype=float) ##

        void_matrix = np.append(void_matrix, [a], axis=0)
        for b in b_matrix:
            void_matrix = np.append(void_matrix, [b], axis=0)
        void_matrix = np.append(void_matrix, [c], axis=0)

        ### End of Matrix code

        for k in range(n):
            f1_plot_cubic = []
            h = (ab[k + 1] - ab[k]) / _n ##

            a1 = derivative(f_i, ab[k])
            b1 = derivative(f_i, ab[k + 1])

            x = np.linspace(ab[k], ab[k + 1], num=_n + 1) ##

            d = np.array([h * a1] + [f_i(x[i]) for i in range(_n + 1)] + [h * b1], dtype=float) ##

            A = void_matrix

            roots = np.linalg.solve(A, d)

            for j in range(0, _n): ##
                sum = 0
                for i in range(-1, _n + 2): ##
                    sum += (funct_plot(j - i)) * roots[i + 1]
                f1_plot_cubic.append(sum)

            x_plot = np.linspace(ab[k], ab[k + 1], num=_n) ##
            plt.plot(x_plot, f1_plot_cubic, color='blue', linewidth=3.0)

        plt.scatter(ab, ab_y, marker='o', c='purple', s=50, zorder=2)
        plt.legend()
        plt.show()


def f_arbitrary_3(x) -> float:
    return math.cos(math.exp(x))


def f_f_arbitrary_nodes_3(x, n) -> list:
    ans = []
    for i in range(n):
        ans.append(math.cos(math.exp(x[i])))
    return ans


if __name__ == "__main__":
    #var1()
    var2_1()
    #var_2_2()
    pass

import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt


def func_plot(func, a, b):
    ans = []
    for x in np.linspace(a, b, num=500):
        ans.append(func(x))
    return ans


def f1(x):
    return (1 / np.sqrt(2 * np.pi)) * np.e ** (-((x ** 2)/2.0))


def evaluation_Q(a, b, error):
    n = 2
    Q1, Q2 = 1e1, 1e10
    while abs(Q2 - Q1) > error:
        Q1 = Q2
        sum = 0
        sum += f1(a) / 2
        for i in range(1, n - 1):
            sum += f1(a + i * ((b - a) / n))
        sum += f1(b) / 2
        sum *= ((b - a) / n)
        Q2 = sum
        n *= 2
    return Q2


def var_2() -> None:
    # Begin Input

    error = float(input("Enter power of e, error: "))
    error = pow(10, -error)

    a, b = 0, 4

    # End input

    # Begin Calculations

    n = 2
    Q1, Q2 = 1e1, 1e10
    while abs(Q2 - Q1) > error:
        Q1 = Q2
        sum = 0
        sum += f1(a) / 2
        for i in range(1, n - 1):
            sum += f1(a + i * ((b - a) / n))
        sum += f1(b) / 2
        sum *= ((b - a) / n)
        Q2 = sum
        n *= 2

    value, min_error = quad(f1, a, b)
    print(f"Scipy value: {value}, error: {min_error}")

    print(f"From composite trapezoidal rule value: {Q2}, error: {error}, n: {n}")

    # End Calculations

    # Begin Calculations number 2

    m = int(input("Enter m: "))
    error = float(input("Enter power of e, error: "))
    error = pow(10, -error)

    x = np.linspace(a, b, num=m)
    print(x)

    array = []
    for i in range(m - 1):
        temp = evaluation_Q(x[i], x[i + 1], error)
        print(temp)
        array.append(temp)
    array.append(evaluation_Q(x[-1], x[-1] + x[1], error))

    # End Calculations number 2

    # Begin plot package

    plt.switch_backend('TkAgg')
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    figure, axis = plt.subplots(2, figsize=(1920 * px, 1080 * px))

    func_plot1 = func_plot(f1, a, b)

    axis[0].plot(np.linspace(a, b, num=500), func_plot1, label= f'Function', linewidth=3.0)
    axis[0].fill_between(np.linspace(a, b, num=500), func_plot1, color='r',
                         label=f'Definite Integral of the function - Area')

    axis[0].set_title("Integral approximation", fontsize=10)
    axis[0].set_xlabel('Value', fontsize=10)
    axis[0].set_ylabel('Function Value', fontsize=10)
    axis[0].legend(loc=9)
    axis[0].grid(True)

    axis[1].plot(x, array, label= f'Function table', linewidth=3.0)
    for i in range(m - 1):
        axis[1].fill_between(x[i:i+2], array[i:i+2])

    axis[1].set_title(f"Integral approximation, nodes:{m}", fontsize=10)
    axis[1].set_xlabel('Value', fontsize=10)
    axis[1].set_ylabel('Function Value', fontsize=10)
    axis[1].legend(loc=9)
    axis[1].grid(True)

    plt.show()

    # End plot package

if __name__ == "__main__":
    var_2() # +
    pass
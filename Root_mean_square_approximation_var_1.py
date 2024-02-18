import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def get_s_array(n, matrix_x_y):
    s_array = np.array([0] * (2 * n + 1), dtype=float)
    for power in iter(range(2 * n + 1)):
        s = 0
        for i in iter(range(matrix_x_y.shape[1])):
            s += matrix_x_y[0][i] ** power
        s_array[power] = s
    return s_array


def get_t_array(n, matrix_x_y):
    t_array = np.array([0] * (n + 1), dtype=float)
    for power in iter(range(n + 1)):
        t = 0
        for i in iter(range(matrix_x_y.shape[1])):
            t += matrix_x_y[0][i] ** power * matrix_x_y[1][i]
        t_array[power] = t
    return t_array


def func_plot(func, a, b):
    ans = []
    for x in np.linspace(a, b, num=500):
        ans.append(func(x))
    return ans


def var_1_1() -> None:
    test_array = np.array([[2, 3, 4, 5]
                              ,[7, 5, 8, 7]], dtype=float)

    array1 = np.array([[1.03, 1.08, 1.16, 1.23, 1.26, 1.33, 1.39]
                            ,[2.8011, 2.9447, 3.1899, 3.4212, 3.5254, 3.7810, 4.0149]], dtype=float)

    array2 = np.array([[-3, -1, 0, 1, 3]
                            ,[-4, -0.8, 1.6, 2.3, 1.5]], dtype=float)

    check_array = array1

    n = int(input("Enter n: "))

    a, b = check_array[0][0], check_array[0][-1]

    # Begin SLAE

    s_array = get_s_array(n, check_array)

    t_array = get_t_array(n, check_array)

    A = np.empty((0, n + 1), dtype=float)

    for i in iter(range(n + 1)):
        A = np.append(A, [s_array[i:n + i + 1]], axis=0)

    print(s_array)
    print(t_array)
    print(A)

    roots = np.linalg.solve(A, t_array)

    print(roots)

    # End SLAE

    # Symbolic package

    x = sp.symbols('x')

    polynomial = 0
    for i in range(n + 1):
        polynomial += roots[i] * x ** i

    print(polynomial)

    interpolation_function = sp.lambdify(x, polynomial, 'numpy')

    func_plot1 = func_plot(interpolation_function, a, b)

    # End Symbolic package

    # Plot package

    plt.switch_backend('TkAgg')
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    figure = plt.figure(figsize=(1920 * px, 1080 * px))

    plt.plot(np.linspace(a, b, num=500), func_plot1, label= f'Interpolation function of power {n}')
    plt.plot(check_array[0], check_array[1],label=f'Table')

    plt.title('Approximation function')
    plt.xlabel('Values')
    plt.ylabel('Function Values')
    plt.legend(loc=9)
    plt.show()

    # End plot package


def var_1_2() -> None:
    n, m = tuple(map(int, input("Enter n and m: ").split(' ')))

    matrix = np.array([[0 for x in range(m)] for x in range(n)], dtype=float)
    array = np.array([0 for x in range(m)])
    print('Enter matrix A:')
    for i in range(n):
        single_row = np.array(list(map(int, input().split())))

        matrix[i] = single_row

    print('Enter array b:')
    array = np.array(list(map(int, input().split())))

    print('Matrix A:\n', matrix)
    print('Array b:', array)

    matrix_transpose = np.transpose(matrix)
    print('Transposed matrix A:\n', matrix_transpose)

    constant_A = np.dot(matrix_transpose, matrix)
    print('Dot product A * A transposed:\n', constant_A)

    constant_b = np.dot(matrix_transpose, array)
    print('Dot product b * A transposed: ', constant_b)

    roots = np.linalg.solve(constant_A, constant_b)
    print('Roots x:', roots)

    error = np.dot(matrix, roots) - array
    print("Error matrix: ", error)
    print('Error e:', sum(error))


if __name__ == "__main__":
    # var_1_1() # +
    var_1_2() # +
    pass
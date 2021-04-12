from sympy import S, zeros, Matrix, nsimplify, factorial, maximum, lambdify, Interval, expand, symbols, pi, cos, eye
from ..genericas import matriz_inversa
from tqdm import tqdm
import numpy as np


def polinomio_lagrange(x_vals, y_vals, var=symbols('x')):
    """Genera el polinomio de interpolación de Lagrange.

    Args:
        x_vals (list): Valores x de interpolación
        y_vals (list): Valores f(x) de interpolación
        var (variable, optional): Variable sobre la que construir el polinomio. Defaults to symbols('x').

    Returns:
        p: polinomio de Lagrange.
        lista_L: Lista de coeficientes l_i de los polinomios.
    """
    lista_L, p, n = [], S(0), len(x_vals)
    for i in range(n):
        li = S(1)
        for k in range(n):
            if k != i:
                li *= (var - x_vals[k]) / (x_vals[i] - x_vals[k])

        lista_L.append(li)
        p += y_vals[i] * li

    return p, lista_L


def polinomio_newton(x_vals, y_vals, var=symbols('x'), evalf=None):
    """Genera el polinomio de interpolación de Newton.

    Args:
        x_vals (list): Valores x de interpolación
        y_vals (list): Valores f(x) de interpolación
        var (variable, optional): Variable sobre la que construir el polinomio. Defaults to symbols('x').
        evalf (int): Redondea la matriz de coeficiente y, por tanto, los coeficientes del polinomio a un número de cifras. 

    Returns:
        p: polinomio de Newton.
        matriz_coeffs: Matriz de coeficientes con el método iterativo.
    """

    matriz_coeffs = zeros(len(y_vals), len(y_vals))
    for i in range(len(y_vals)):  # Asignamos la primera columna como f(x)
        matriz_coeffs[i, 0] = y_vals[i]

    for col in range(1, len(y_vals)):
        for row in range(len(y_vals) - col):
            num = matriz_coeffs[row + 1, col - 1] - matriz_coeffs[row, col - 1]
            den = x_vals[row + col] - x_vals[row]

            matriz_coeffs[row, col] = num / den

    matriz_coeffs = nsimplify(matriz_coeffs, tolerance=1e-10, rational=True)  # Esto es importante para quitar valores de redondeo
    if evalf is not None:  # si evalf es un numero redondea a ese numero de decimales
        matriz_coeffs = matriz_coeffs.evalf(evalf)

    # Aqui hacemos el polinomio
    p = matriz_coeffs[0, 0]
    for col in range(1, len(y_vals)):
        p_col = S(1)
        for i in range(0, col):
            p_col *= (var - x_vals[i])

        p += p_col * matriz_coeffs[0, col]

    return p, matriz_coeffs


def error_lagrange(f, x_vals, I=[0, 1], var=symbols('x')):
    """Calcula el error del polinomio de Lagrange, definido por PROD[(x-xi)]/(n+1)! max f^(n+1)(t)

    Args:
        f (función): Función interpolada (si se conoce).
        x_vals (list): Valores x de interpolación.
        I (list): Intervalo de interpolación.
        var (variable, optional): Variable de interpolación. Defaults to symbols('x').

    Returns:
        E_x (float): Error máximo de interpolación
    """
    E_x_fact = S(1)
    for x_val in x_vals:
        E_x_fact *= (var - x_val)

    E_x_fact /= factorial(len(x_vals))

    # Ahora derivamos la función n veces
    diff_f = f

    for _ in range(len(x_vals)):
        diff_f = diff_f.diff(var)

    max_diff_f = maximum(diff_f, var, Interval(I[0], I[1]))

    E_x = E_x_fact * abs(max_diff_f)

    return E_x


def error_maximo_estocastico(f, var=symbols('x'), grado=3, N=1000, I=[0, 1], list_x_vals=None):
    """Emplea un método estocástico para simular el error máximo de la interpolación. Para ello genera n (grado) nodos dentro del intervalo, calcula el error máximo para ese caso específico, y 
    actualiza el error máximo para cada conjunto de nodos.

    Args:
        f (funcion): Función a interpolar.
        var (variable, optional): Variable de la función. Defaults to symbols('x').
        grado (int, optional): Grado de interpolación. Defaults to 3.
        N (int, optional): Número de generaciones aleatorias de nodos. Defaults to 1000.
        I (list, optional): Intervalo de interpolación. Defaults to [0, 1].
        list_x_vals (list, optional): Listado de nodos, definido por el usuario. Defaults to None.

    Returns:
        max_error (float): Error estocástico máximo.
    """
    max_error = 0

    # Hacemos la funcion por separado porque el cálculo de derivada es costoso y cada iteración de error_lagrange tarda mucho
    diff_f = f
    for _ in range(grado):
        diff_f = diff_f.diff(var)
    max_diff_f = maximum(diff_f, var, Interval(I[0], I[1]))

    for _ in tqdm(range(N)):
        if list_x_vals is None:
            x_list = np.sort(np.random.rand(grado) * (I[1] - I[0]) + I[0])
        else:
            x_list = list_x_vals

        E_x_fact = S(1) / S(factorial(grado))
        for x_val in x_list:
            E_x_fact *= (var - x_val)

        E_x = lambdify(var, expand(E_x_fact * max_diff_f))  # lambdify es para que la evaluación numérica sea mucho más ágil

        x_range = np.linspace(I[0], I[1], int(N / 10))
        y_f = [E_x(i) for i in x_range]
        max_y_f = np.max(abs(np.array(y_f)))

        if max_y_f >= max_error:
            max_error = max_y_f

        if list_x_vals is not None:
            break

    return max_error


def roots_chebyshev(n, I=[-1, 1]):
    """Calcula las raíces del polinomio de Chebyshev de grado n.

    Args:
        n (int): Grado del polinomio
        I (list, optional): Intervalo de generación del polinomio. Defaults to [-1, 1].

    Returns:
        roots (list): Lista con las raices, en forma trigonométrica.
    """
    roots = [0.5 * (I[1] + I[0]) + 0.5 * (I[1] - I[0]) * cos(S(2 * (i + 1) - 1) / S(2 * n) * pi) for i in range(n)][::-1]
    return roots


def aitken_neville(x_vals, y_vals, x0, modo='aitken'):
    """Aplica los algoritmos de Aitken y Nevile para calcular el valor de una función interpolada en un punto nuevo, dados los valores en nodos anteriores.

    Args:
        x_vals (list): Valores x de interpolación
        y_vals (list): Valores f(x) de interpolación
        x0 (float): Valor x del nodo a calcular su f(x)
        modo (str, optional): ['aitken', 'neville']. Defaults to 'aitken'.

    Returns:
        matriz_coeffs: Matriz de coeficientes de la interpolación. El elemento [0, -1] es el valor interpolado.
    """
    matriz_coeffs = zeros(len(y_vals), len(y_vals))
    for i in range(len(y_vals)):  # Asignamos la primera columna como f(x)
        matriz_coeffs[i, 0] = y_vals[i]

    for col in range(1, len(y_vals)):
        for row in range(len(y_vals) - col):
            xk = None
            if modo == 'aitken':
                xk = x_vals[0]
            elif modo == 'neville':
                xk = x_vals[row]

            num = (x0 - xk) * matriz_coeffs[row + 1, col - 1] - (x0 - x_vals[row + col]) * matriz_coeffs[row, col - 1]
            den = x_vals[row + col] - xk

            matriz_coeffs[row, col] = num / den

    matriz_coeffs = nsimplify(matriz_coeffs, tolerance=1e-10, rational=True)  # Esto es importante para quitar valores de redondeo

    return matriz_coeffs


def interpolacion_hermite(x_vals, y_vals, diff_vals, var=symbols('x'), evalf=None):
    """Aplica la interpolación de Hermite, que permite la introducción de valores de derivadas. En esta implementación solo
    se permiten valores de la primera derivada.

    Args:
        x_vals (list): Valores x de interpolación.
        y_vals (list): Valores f(x) de interpolación.
        diff_vals (list): Valores f'(x) de interpolación.
        var (variable, optional): Variable sobre la que construir el polinomio. Defaults to symbols('x').
        evalf (int): Redondea la matriz de coeficiente y, por tanto, los coeficientes del polinomio a un número de cifras. 

    Returns:
        p: polinomio de Hermite.
        matriz_coeffs: Matriz de coeficientes con el método iterativo.
    """
    base_w = [S(1)]
    double_x_vals = []
    for val in x_vals:
        double_x_vals += [val, val]

    for x_val in x_vals:
        base_w += [base_w[-1] * (var - x_val), base_w[-1] * (var - x_val) ** 2]

    base_w = base_w[:-1]  # Eliminamos la componente w_2n+1, que no se emplea para el cálculo del polinomio
    matriz_coeffs = zeros(len(base_w), len(base_w))

    # Asignamos la primera columna como f(x)
    for i in range(2 * len(y_vals)):
        matriz_coeffs[i, 0] = y_vals[i // 2]

    # Asignamos la segunda columna para f'(x) o f[xa, xa+1]
    for i in range(0, 2 * len(y_vals), 2):
        matriz_coeffs[i, 1] = diff_vals[i // 2]

    for i in range(1, 2 * len(y_vals) - 1, 2):
        num = matriz_coeffs[i + 1, 0] - matriz_coeffs[i, 0]
        den = double_x_vals[i + 1] - double_x_vals[i]
        matriz_coeffs[i, 1] = num / den

    # Asignamos para el resto de columnas
    for col in range(2, 2 * len(y_vals)):
        for row in range(2 * len(y_vals) - col):
            num = matriz_coeffs[row + 1, col - 1] - matriz_coeffs[row, col - 1]
            den = double_x_vals[row + col] - double_x_vals[row]

            matriz_coeffs[row, col] = num / den

    matriz_coeffs = nsimplify(matriz_coeffs, tolerance=1e-10, rational=True)  # Esto es importante para quitar valores de redondeo
    if evalf is not None:  # si evalf es un numero redondea a ese numero de decimales
        matriz_coeffs = matriz_coeffs.evalf(evalf)

    # Aqui hacemos el polinomio
    p = S(0)
    for i in range(len(base_w)):
        p += base_w[i] * matriz_coeffs[0, i]

    return p, matriz_coeffs


def polinomio_generico(lista_condiciones, var=symbols('x'), evalf=None):
    """Calcula un polinomio que satisface las condiciones de interpolación.
       La lista de condiciones viene data por una lista de listas:
       [ [(a, b), (c, d)],  [],  [(e, f)]]
       Esto significaria: f(a) = b;  f(c) = d;  f''(e) = f
    Args:
        lista_condiciones (list): Lista de condiciones
        var (variable, optional): Variable sobre la que construir el polinomio. Defaults to symbols('x').
        evalf (int): Redondea la matriz de coeficiente y, por tanto, los coeficientes del polinomio a un número de cifras. 

    Returns:
        p (polinomio): polinomio de interpolación. 
        D (matriz): Matriz de coeficientes. Cada fila tiene tantas columnas como coeficientes a_i, y satisface la condición determinada: f''(3) -> 0  0 2 3·2·(3) 4·3·(3)^2 5·4·(3)^3 ...
        rhs: Matriz de rhs de la ecuación: f''(3) = 34 -> 34
        a_vals: Matrices de valores del polinomio que cumplen la ecuación D·a = rhs
    """
    # La lista de condiciones viene data por una lista de listas:
    # [ [(a, b), (c, d)],  [],  [(e, f)]]
    # Esto significaria: f(a) = b;  f(c) = d;  f''(e) = f

    # Primero contamos el número de elementos para generar el polinomio de grado n-1
    n = sum([len(i) for i in lista_condiciones])
    p_list = []
    for i in range(n):
        p_list.append(var ** i)

    # Ahora implementamos las condiciones. Cada elemento de la lista de condiciones será p, o sus derivadas, sustituyendo el elemento determinado
    D, rhs = zeros(n, n), []

    row = 0
    for diff_range in range(len(lista_condiciones)):
        # Primero aplicamos la n-derivada a ese polinomio
        for pair in lista_condiciones[diff_range]:
            for col in range(diff_range, n):
                if col == 0:
                    multiplicado_derivada = 1
                else:
                    multiplicado_derivada = factorial(col) / factorial(col - diff_range)
                D[row, col] = multiplicado_derivada * pair[0] ** (col - diff_range)  # Añadimos a la matriz su elemento, que es la derivada por x^n

            rhs.append(pair[1])
            row += 1

    try:
        a = matriz_inversa(D) * Matrix(rhs)
        a_vals = {symbols(f'a{i}'): a[i, 0] for i in range(D.shape[0])}

        p = S(0)
        for i in range(n):
            p += var**i * a_vals[symbols(f'a{i}')]
    except:
        print('La matriz de valores no tiene inversa. No existe un polinomio de interpolación que verifique las condiciones.')
        a_vals = None
        p = None

    return p, D, rhs, a_vals


def esplines(x_vals, y_vals, var=symbols('x')):
    """Genera un esplín natural cúbico.

    Args:
        x_vals (list): Valores x de interpolación.
        y_vals (list): Valores f(x) de interpolación.
        var (variable, optional): Variable sobre la que construir el esplín. Defaults to symbols('x').

    Returns:
        S_dict (dict): Diccionario con los polinomios S_i para cada par de nodos x_i x_i+1
        valores (dict): Para cada i, retorna los coeficientes a_i, b_i, c_i, d_i del polinomio S_i = a_i(x-x_i)^3 + b_i(x-x_i)^2 + c_i(x-x_i) + d_i
        D (matriz): Matriz de construcción del esplín.
        z (list): Lista de valores z_i para derivar a_i, b_i, c_i, d_i
        rhs (matriz): Matriz con los valores 6 * f[x_i, x_i+1, x_i+2]
    """
    # Primero creamos los polinomios de los esplines
    S_dict = {}
    valores = {}

    # Ahora hallamos h y r para resolver el sistema
    h_list = [x_vals[i + 1] - x_vals[i] for i in range(len(x_vals) - 1)]
    r_list = [h_list[i] / (h_list[i] + h_list[i + 1]) for i in range(len(x_vals) - 2)]

    D = 2 * eye(len(x_vals) - 2)
    for i in range(len(x_vals) - 3):
        D[i, i + 1] = 1 - r_list[i]
        D[i + 1, i] = r_list[i + 1]

    # Ahora hallamos el rhs, que está compuesto por f[a, b, c] = (f[b, c] - f[a, b])/(c-a) = ({(f(c) - f(b))/(c-b)} - {(f(b) - f(a))/(b-a)})/(c-a)
    rhs = 6 * Matrix([(((y_vals[i + 2] - y_vals[i + 1]) / (x_vals[i + 2] - x_vals[i + 1])) -
                       ((y_vals[i + 1] - y_vals[i]) / (x_vals[i + 1] - x_vals[i]))) /
                      (x_vals[i + 2] - x_vals[i])
                      for i in range(len(x_vals) - 2)])

    # Resolvemos el sistema D*z = rhs
    z = matriz_inversa(D) * rhs
    z_list = [0] + list(z) + [0]

    # Asignamos los valores de a, b, c, d, y con ello creamos los polinomios
    for i in range(len(x_vals) - 1):
        a = (z_list[i + 1] - z_list[i]) / (6 * h_list[0])
        b = z_list[i] / 2
        c = (y_vals[i + 1] - y_vals[i]) / (h_list[i]) - (2 * z_list[i] + z_list[i + 1]) / (6) * h_list[i]

        valores[i] = [a, b, c, y_vals[i]]
        S_dict[f'S_{i}'] = expand(y_vals[i] + c * (var - x_vals[i]) + b * (var - x_vals[i]) ** 2 + a * (var - x_vals[i]) ** 3)

    return S_dict, valores, D, z, rhs

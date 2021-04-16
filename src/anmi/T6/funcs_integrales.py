from sympy import S, zeros, Matrix, nsimplify, factorial, maximum, Interval, symbols, Function, ones, Integral, solve, legendre, chebyshevt
from ..genericas import matriz_inversa

import numpy as np


def polinomio_lagrange_integracion(f, x_vals=None, grado=3, var=symbols('x'), I=[0, 1]):
    """Aplica el método de lagrange para integrar una función.

    Args:
        f (funcion): Función a integrar.
        x_vals (list, optional): Lista de nodos donde se evaluará el polinomio. Si None, selecciona tantos nodos como grado, de manera equidistante. Defaults to None.
        grado (int, optional): Número de nodos. Defaults to 3.
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        I (list, optional): Intervalo de integración. Defaults to [0, 1].

    Returns:
        p (polinomio): Polinomio de integración
        val (float): Valor aproximado de la integral, tras la evaluación de p en los nodos.
    """

    ff = Function('f')

    if x_vals is None:
        x_vals = np.linspace(I[0], I[1], grado)

    G = ones(len(x_vals), len(x_vals))
    for row in range(1, len(x_vals)):
        for col in range(len(x_vals)):
            G[row, col] = x_vals[col] ** row

    x_mat = nsimplify(Matrix([(I[1] ** (i + 1) - I[0] ** (i + 1)) / (i + 1) for i in range(len(x_vals))]), rational=True)

    # Con esto creamos el vector de valores de A
    a = nsimplify(matriz_inversa(G) * x_mat, rational=True)

    p, val = S(0), S(0)
    for i in range(len(x_vals)):
        p += a[i] * ff(x_vals[i])
        val += a[i] * f.subs(var, x_vals[i])

    return p, val


def polinomio_newton_integracion(f, x_vals=None, grado=3, var=symbols('x'), I=[0, 1]):
    """Aplica el método de lagrange para integrar una función usando las funciones w de Newton.

    Args:
        f (funcion): Función a integrar.
        x_vals (list, optional): Lista de nodos donde se evaluará el polinomio. Si None, selecciona tantos nodos como grado, de manera equidistante. Defaults to None.
        grado (int, optional): Número de nodos. Defaults to 3.
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        I (list, optional): Intervalo de integración. Defaults to [0, 1].

    Returns:
        p (polinomio): Polinomio de integración
        val (float): Valor aproximado de la integral, tras la evaluación de p en los nodos.
    """

    ff = Function('f')

    if x_vals is None:
        x_vals = np.linspace(I[0], I[1], grado)

    # Creamos la lista de omegas, que tienen la expresión [w_1, w_2, w_3, etc]
    lista_w = [1, (var - x_vals[0])]
    for x_val in x_vals[1:]:
        lista_w.append(lista_w[-1] * (var - x_val))

    G = zeros(len(x_vals), len(x_vals))
    x_mat = zeros(len(x_vals), 1)
    for row in range(len(x_vals)):
        for col in range(row, len(x_vals)):
            if row == 0:
                G[row, col] = 1
            else:
                G[row, col] = lista_w[row].subs(var, x_vals[col])

        x_mat[row, 0] = Integral(lista_w[row], (var, I[0], I[1])).doit()

    # Con esto creamos el vector de valores de A
    a = matriz_inversa(G) * x_mat

    p, val = S(0), S(0)
    for i in range(len(x_vals)):
        p += a[i] * ff(x_vals[i])
        val += a[i] * f.subs(var, x_vals[i])

    return p, val


def integral_positiva_newton(omega, I, var, x_vals):
    # Esta función la creo para que calcule la integral del valor absoluto. Como las integrales son de la forma (x-x0)(x-x1)(x-x2)(x-x3)...
    # con xi diferentes y en orden creciente, la función va a tomar un valor negativo si el número de términos es impar, y positivo si par
    # para a < x_0, y de ahí va a ir alternando signos. Como pueden surgir casos diferentes (que I[0] sea x_vals[0]) y la casuistica cambia
    # para cada x_i, x_i+1 tomaremos la integral y evaluaremos el signo, para ponerlo a la integral de nuevo.
    integral = S(0)

    x_vals_I = [I[0]] + x_vals + [I[1]]
    for i in range(len(x_vals_I) - 1):
        a, b = x_vals_I[i], x_vals_I[i + 1]
        integral_i = Integral(omega, (var, a, b)).doit()

        if integral_i > 0:
            signo = 1
        elif integral_i < 0:
            signo = -1
        else:  # a = b y la integral es cero
            continue
        integral += signo * integral_i

    return integral


def error_cuadratura_integral(f, var=symbols('x'), x_vals=None, I=[0, 1], grado=2):
    """Calcula el error máximo de cuadratura de integral en base al polinomio de aproximación

    Args:
        f (funcion): Función a integrar.        
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        x_vals (list, optional): Lista de nodos donde se evaluará el polinomio. Si None, selecciona tantos nodos como grado, de manera equidistante. Defaults to None.
        I (list, optional): Intervalo de integración. Defaults to [0, 1].
        grado (int, optional): Número de nodos. Defaults to 3.

    Returns:
        e (float): Error de aproximación
        max_diff_f: Valor de máximo de diferencia (entre el factorial)
        int_omega: Valor de la integral de Newton
    """
    if x_vals is None:
        x_vals = np.linspace(I[0], I[1], grado)

    # Hallamos el primer término, el del máximo de la derivada
    diff_f = f
    for _ in range(len(x_vals)):
        diff_f = diff_f.diff(var)

    max_diff_f = nsimplify(maximum(diff_f, var, Interval(I[0], I[1])) / factorial(len(x_vals)), rational=True)

    # Ahora hallamos el de la integral
    omega = S(1)
    for x_val in x_vals:
        omega *= (var - x_val)

    int_omega = nsimplify(integral_positiva_newton(omega, I, var, x_vals), rational=True)

    e = max_diff_f * int_omega

    return e, max_diff_f, int_omega


def formulas_newton_cotes(f, orden=1, x_vals=None, y_vals=None, I=[0, 1], var=symbols('x')):
    """Calcula las fórmulas de Newton-Cotes para la resolución de la integral. 

    Args:
        f (funcion): Función a integrar.  
        orden (int): Orden de las fórmulas de Newton-Cotes.      
        x_vals (list, optional): Lista de nodos donde se evaluará el polinomio. Si None, selecciona tantos nodos como grado, de manera equidistante. Defaults to None.
        y_vals (list, optional): Lista de valores de la función. Emplear si no hay una función dada.
        I (list, optional): Intervalo de integración. Defaults to [0, 1].
        var (variable, optional): Variable de integración. Defaults to symbols('x').

    Returns:
        Q (polinomio): Fórmula de Newton-Cotes simbólica.
        val (float): Evaluación de la fórmula de Newton-Cotes en los nodos de evaluación.
    """

    ff = Function('f')

    if x_vals is None:
        x_vals = np.linspace(I[0], I[1], orden + 1)
    if y_vals is None:
        y_vals = [f.subs(var, i) for i in x_vals]

    G = ones(len(x_vals), len(x_vals))
    for row in range(1, len(x_vals)):
        for col in range(len(x_vals)):
            G[row, col] = x_vals[col] ** row

    x_mat = nsimplify(Matrix([(I[1] ** (i + 1) - I[0] ** (i + 1)) / (i + 1) for i in range(len(x_vals))]), rational=True)

    # Con esto creamos el vector de valores de A
    a = nsimplify(matriz_inversa(G) * x_mat, rational=True)

    Q, val = S(0), S(0)
    for i in range(len(x_vals)):
        Q += a[i] * ff(nsimplify(x_vals[i], rational=True))
        val += nsimplify(a[i] * f.subs(var, x_vals[i]), rational=True)

    return Q, val


def error_trapecio(f, var=symbols('x'), I=[0, 1]):
    """Calcula el error máximo de cuadratura de integral usando las fórmulas de Newton-Cotes para el trapecio
    Args:
        f (funcion): Función a integrar.        
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        I (list, optional): Intervalo de integración. Defaults to [0, 1].

    Returns:
        e (float): Error de aproximación
        max_diff_f: Valor de máximo de diferencia (entre el factorial)
        int_omega: Valor de la integral de Newton
    """

    diff_f = f
    for _ in range(2):
        diff_f = diff_f.diff(var)

    max_diff_f = nsimplify(maximum(diff_f, var, Interval(I[0], I[1])), rational=True)

    factor_division = (I[1] - I[0]) ** 3 / 12

    e = max_diff_f * factor_division

    return e, max_diff_f, factor_division


def error_simpson(f, var=symbols('x'), I=[0, 1]):
    """Calcula el error máximo de cuadratura de integral usando las fórmulas de Newton-Cotes para Simpson
    Args:
        f (funcion): Función a integrar.        
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        I (list, optional): Intervalo de integración. Defaults to [0, 1].

    Returns:
        e (float): Error de aproximación
        max_diff_f: Valor de máximo de diferencia (entre el factorial)
        int_omega: Valor de la integral de Newton
    """

    diff_f = f
    for _ in range(4):
        diff_f = diff_f.diff(var)

    max_diff_f = nsimplify(maximum(diff_f, var, Interval(I[0], I[1])), rational=True)

    factor_division = ((I[1] - I[0]) / 2) ** 5 / 90

    e = max_diff_f * factor_division

    return e, max_diff_f, factor_division


def formulas_newton_cotes_m(f, orden=1, m=10, I=[0, 1], var=symbols('x')):
    """Calcula las fórmulas de Newton-Cotes para la resolución de la integral usando m particiones del intervalo. 

    Args:
        f (funcion): Función a integrar.  
        orden (int): Orden de las fórmulas de Newton-Cotes.      
        m (int): Número de particiones del intervalo. Si m=1 la funcion es idéntica a formulas_newton_cotes
        I (list, optional): Intervalo de integración. Defaults to [0, 1].
        var (variable, optional): Variable de integración. Defaults to symbols('x').

    Returns:
        Q (polinomio): Fórmula de Newton-Cotes simbólica.
        val (float): Evaluación de la fórmula de Newton-Cotes en los nodos de evaluación.
    """

    lista_vals, lista_Q = [], []
    lista_m = np.linspace(I[0], I[1], m + 1)

    for m_i in range(m):
        m0, mf = lista_m[m_i], lista_m[m_i + 1]
        Q, val = formulas_newton_cotes(f, orden=orden, I=[m0, mf], var=var)
        lista_Q.append(Q)
        lista_vals.append(val)

    Q = sum(lista_Q)
    val = sum(lista_vals)

    return Q, val


def error_trapecio_m(f, var=symbols('x'), I=[0, 1], m=2):
    """Calcula el error máximo de cuadratura de integral usando las fórmulas de Newton-Cotes para el trapecio
    Args:
        f (funcion): Función a integrar.        
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        I (list, optional): Intervalo de integración. Defaults to [0, 1].
    m (int): Número de particiones del intervalo. Si m=1 la funcion es idéntica a error_trapecio
    Returns:
        e (float): Error de aproximación
        max_diff_f: Valor de máximo de diferencia (entre el factorial)
        int_omega: Valor de la integral de Newton
    """
    diff_f = f
    for _ in range(2):
        diff_f = diff_f.diff(var)

    a, b = I  # Me da fallo random si no hago esto!
    max_diff_f = nsimplify(maximum(diff_f, var, Interval(a, b)), rational=True)

    factor_division = (I[1] - I[0]) ** 3 / (12 * (m ** 2))

    e = max_diff_f * factor_division

    return e, max_diff_f, factor_division


def error_simpson_m(f, var=symbols('x'), I=[0, 1], m=2):
    """Calcula el error máximo de cuadratura de integral usando las fórmulas de Newton-Cotes para Simpson
    Args:
        f (funcion): Función a integrar.        
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        I (list, optional): Intervalo de integración. Defaults to [0, 1].
    m (int): Número de particiones del intervalo. Si m=1 la funcion es idéntica a error_trapecio
    Returns:
        e (float): Error de aproximación
        max_diff_f: Valor de máximo de diferencia (entre el factorial)
        int_omega: Valor de la integral de Newton
    """
    diff_f = f
    for _ in range(4):
        diff_f = diff_f.diff(var)

    a, b = I  # Me da fallo random si no hago esto!
    max_diff_f = nsimplify(maximum(diff_f, var, Interval(a, b)), rational=True)

    factor_division = (I[1] - I[0]) ** 5 / (90 * (m ** 4) * (2 ** 5))

    e = max_diff_f * factor_division

    return e, max_diff_f, factor_division


def formulas_gauss(f, I=[-1, 1], var=symbols('x'), grado=3, modo='legendre'):
    """Calcula las fórmulas de Newton-Cotes para la resolución de la integral nodos de Chebyshev o Legendre. 

    Args:
        f (funcion): Función a integrar.  
        I (list, optional): Intervalo de integración. Defaults to [0, 1].
        var (variable, optional): Variable de integración. Defaults to symbols('x').
        grado (int): Grado del polinomio de Legendre o Chebyshev

    Returns:
        Q (polinomio): Fórmula de Gauss simbólica.
        val (float): Evaluación de la fórmula de Gauss en los nodos de evaluación.
    """

    ff = Function('f')

    x_vals = []
    if modo == 'legendre':
        x_vals = sorted(solve(legendre(grado, var)))
    elif modo == 'chebyshev':
        x_vals = sorted(solve(chebyshevt(grado, var)))

    G = ones(len(x_vals), len(x_vals))
    for row in range(1, len(x_vals)):
        for col in range(len(x_vals)):
            G[row, col] = x_vals[col] ** row

    x_mat = nsimplify(Matrix([((1) ** (i + 1) - (-1) ** (i + 1)) / (i + 1) for i in range(len(x_vals))]), rational=True)

    # Con esto creamos el vector de valores de A
    a = nsimplify((I[1] - I[0]) / 2 * (matriz_inversa(G) * x_mat), rational=True)

    # Aquí efectuamos el cambio de variable:
    cambio_var = (I[1] - I[0]) / 2 * var + (I[1] + I[0]) / 2
    Q, val = S(0), S(0)
    for i in range(len(x_vals)):
        Q += a[i] * ff(nsimplify(cambio_var.subs(var, x_vals[i]), rational=True))
        val += nsimplify(a[i] * f.subs(var, cambio_var.subs(var, x_vals[i])), rational=True)

    return Q, val

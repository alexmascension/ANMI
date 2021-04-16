from sympy import S, zeros, Matrix, nsimplify, symbols, Function, ones
from ..T5 import polinomio_newton
from ..genericas import matriz_inversa

import numpy as np


def derivacion_polinomio_newton(f, x0, var=symbols('x'), grado=1, h=0.01):
    """Aplica el método de Newton para calcular la derivada de una función en un punto, creando una matriz con x, x+h, x+2h, etc.

    Args:
        f (funcion): Función a derivar
        x0 (float): Punto de evaluación
        var (variable, optional): Variable de f. Defaults to symbols('x').
        grado (int, optional): Grado del polinomio de aproximación. Defaults to 1.
        h (float, optional): Valor h de diferencia. Defaults to 0.01.

    Returns:
        p_diff (float): Valor de la derivada.
        p (polinomio): Polinomio de Newton.
        m (matriz): Matriz de diferencias del algoritmo.
    """
    x_vals = [x0 + i * h for i in range(grado + 1)]
    y_vals = [f.subs(var, (x0 + i * h)) for i in range(grado + 1)]

    # Con esto creamos el polinomio
    p, m = polinomio_newton(x_vals, y_vals, var)

    # Ahora lo diferenciamos
    p_diff = p
    for i in range(grado):
        p_diff = p_diff.diff(var)

    return p_diff, p, m


def polinomio_lagrange_derivada(f, x_vals=None, var=symbols('x'), grado=1, I=[0, 1]):
    """Genera el polinomio de Lagrange para la derivada de f.

    Args:
        f (funcion): Función a derivar
        x_vals (list): Lista de nodos dentro del intervalo donde se evaluará el polinomio. Si None, se distribuyen equidistantemente por el intervalo.
        var (variable, optional): Variable de f. Defaults to symbols('x').
        grado (int, optional): Grado del polinomio de aproximación. Defaults to 1. Se cumple que el grado es igual al número de valores - 1
        I (list): Intervalo de aproximación

    Returns:
        p (polinomio): función con los coeficientes de aproximación
        val (float): evaluación del polinomio en los nodos correspondientes
    """

    ff = Function('f')

    if x_vals is None:
        x_vals = np.linspace(I[0], I[1], grado + 1)
    assert len(x_vals) == grado + 1

    G = ones(len(x_vals), len(x_vals))
    for row in range(1, len(x_vals)):
        for col in range(len(x_vals)):
            G[row, col] = x_vals[col] ** row

    x_mat = Matrix([var ** i for i in range(grado + 1)])
    for _ in range(grado):
        x_mat = x_mat.diff(var)

    # Con esto creamos el vector de valores de A
    a = matriz_inversa(G) * x_mat

    p, val = S(0), S(0)
    for i in range(grado + 1):
        p += a[i] * ff(x_vals[i])
        val += a[i] * f.subs(var, x_vals[i])

    return p, val


def derivada_richardson(f, x0=0, r=0.5, h=0.5, x_var=symbols('x'), h_var=symbols('h'), Nf=None, m=None, grado=3):
    """Aplica el método de Richardson para el cálculo de la derivada de una función.

    Args:
        f (funcion): Función a derivar
        x0 (float): Punto de evaluación
        r (float, optional): Valor de r de aproximación (ideal cercano a 0.5). Defaults to 0.5.
        h (float, optional): Valor h de diferencia. Defaults to 0.5.
        x_var (variable, optional): Variable de f. Defaults to symbols('x').
        h_var (variable, optional): Variable h. Defaults to symbols('h').
        Nf (funcion, optional): Función de derivación. Por defecto es f(x+h) - f(x-h) / 2h.
        m (int, optional): Grado de error inicial en base a Nf. Para el caso por defecto, m = 2.
        grado (int, optional): Número de pasos del algoritmo. Defaults to 3.

    Returns:
        val (float): valor de la derivada
        matriz_r (matriz): matriz con los pasos del algoritmo
    """
    if Nf is None:
        Nf = (f.subs(x_var, x0 + h_var) - f.subs(x_var, x0 - h_var)) / (2 * h_var)
        m = 2

    matriz_r = zeros(grado, grado)

    # primero llenamos la matriz con los valores N(rih)
    for row in range(grado):
        rih = r ** row * h
        matriz_r[row, 0] = nsimplify(Nf.subs(h_var, rih), tolerance=1e-10, rational=True)

    for col in range(1, grado):
        for row in range(grado - col):
            r_exp = m + col - 1
            matriz_r[row, col] = nsimplify((r**r_exp * matriz_r[row, col - 1] - matriz_r[row + 1, col - 1]) / (r**r_exp - 1), rational=True)

    val = matriz_r[0, -1]

    return val, matriz_r

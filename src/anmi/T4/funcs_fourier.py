import numpy as np
from sympy import simplify, integrate, zeros, S, Matrix, symbols, pi, cos, sin
from .funcs_aproximacion import producto_asecas


def producto_escalar_trigono(f, g, var=symbols('x'), a=-pi, b=pi, I=None, numeric=False):
    """Aplica el producto escalar <f,g> = 1/(2pi) ∫_[-pi]^[pi] f.g

    Args:
        f (funcion): f
        g (funcion): g
        var (variable): variable de integración
        a (int, optional): limite inferior de integracion. Defaults to 0.
        b (int, optional): limite superior de integracion. Defaults to 1.
        I (list, optional): Si no es None, lista de valores sobre los que hacer un sumatorio discreto. Defaults to None.
        numeric (bool, optional): si True, realiza una aproximación numérica de la integral usando un método de sympy.

    Returns:
        funcion, float: Valor del producto escalar. Se devuelve como funcion si tiene variables.
    """
    prod = producto_asecas(f, g, var, a, b, I, numeric)
    return simplify(prod / (2 * pi))


def coefs_fourier(f, var=symbols('x'), I=[0, 1], n_coefs=2):
    """Genera los coeficientes de la serie de fourier. Esta es la versión continua, donde los coeficientes se calculan usando la expresión de la integral.

    Args:
        f (funcion): Función a aproximar
        var (variable, optional): Variable de la función. Defaults to symbols('x').
        I (list, optional): Intervalo de aproximación de la función. Defaults to [0, 1].
        n_coefs (int, optional): Número de coeficientes de la serie a generar. Defaults to 2.

    Returns:
        dict_coefs: {a_0, a_1, b_1, a_2, b_2, ...}
    """
    dict_coefs = {}
    dict_coefs['a0'] = simplify(1 / pi * integrate(f, (var, I[0], I[1])))
    for i in range(1, n_coefs):
        dict_coefs[f'a{i}'] = simplify(1 / pi * integrate(f * cos(i * var), (var, I[0], I[1])))
        dict_coefs[f'b{i}'] = simplify(1 / pi * integrate(f * sin(i * var), (var, I[0], I[1])))

    return dict_coefs


def coefs_fourier_discr(f, var=symbols('x'), I=[0, 1], n_coefs=2, m=10):
    """Genera los coeficientes de la serie de fourier. Esta es la versión donde la integral se aproxima como un sumatorio discreto de m términos sobre I.

    Args:
        f (funcion): Función a aproximar
        var (variable, optional): Variable de la función. Defaults to symbols('x').
        I (list, optional): Intervalo de aproximación de la función. Defaults to [0, 1].
        n_coefs (int, optional): Número de coeficientes de la serie a generar. Defaults to 2.
        m (int, optional): Número de elementos en los que dividir I para el sumatorio.

    Returns:
        dict_coefs: {a_0, a_1, b_1, a_2, b_2, ...}
    """
    dict_coefs = {}
    lista_xk = np.linspace(I[0], I[1], 2 * m)

    dict_coefs['a0'] = np.sum([f.subs(var, xk) * cos(0 * xk) for xk in lista_xk]) / m
    for i in range(1, n_coefs):
        dict_coefs[f'a{i}'] = np.sum([f.evalf(subs={var: S(xk)}) * cos(S(i) * xk) for xk in lista_xk]) / m
        dict_coefs[f'b{i}'] = np.sum([f.evalf(subs={var: S(xk)}) * sin(S(i) * xk) for xk in lista_xk]) / m

    return dict_coefs


def serie_fourier(f, var=symbols('x'), I=[0, 1], n_coefs=3, discreto=False, m=10):
    """Genera la serie de Fourier para la función f sobre un intervalo.

    Args:
        f (funcion): Función a aproximar
        var (variable, optional): Variable de la función. Defaults to symbols('x').
        I (list, optional): Intervalo de aproximación de la función. Defaults to [0, 1].
        n_coefs (int, optional): Número de coeficientes de la serie a generar. Defaults to 2.
        discreto (bool, optional): Si True, genera una aproximación discreta de los coeficientes empleando m términos.
        m (int, optional): Número de elementos en los que dividir I para el sumatorio.

    Returns:
        funcion: Función polinómica con la serie de Fourier.
    """
    if discreto:
        dict_coefs = coefs_fourier_discr(f, var, I, n_coefs, m)
    else:
        dict_coefs = coefs_fourier(f, var, I, n_coefs)

    serie_fourier = dict_coefs['a0'] / 2
    for i in range(1, n_coefs):
        serie_fourier += dict_coefs[f'a{i}'] * cos(i * var) + dict_coefs[f'b{i}'] * sin(i * var)

    return simplify(serie_fourier)

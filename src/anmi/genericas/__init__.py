from sympy import sqrt, simplify, symbols, integrate, solve, Matrix, eye, cbrt
import numpy as np
from numpy.linalg import inv


def norma(x):
    """Devuelve la norma de un vector <x,x>.

    Args:
        x (vector): Vector fila o columna

    Returns:
        float: Norma del vector
    """

    if x.shape[0] == 1:
        return sqrt((x * x.T)[0])
    else:
        return sqrt((x.T * x)[0])


def print_verbose(msg, verbose):
    """Printea el contenido de msg si verbose.

    Args:
        msg (str, list): mensaje a describir. Puede ser una list ade strings.
        verbose (bool): verbose
    """

    if type(msg) == str:
        msg = [msg]
    if verbose:
        for i in msg:
            print(i)


def norma_1(A, verbose=False):
    """Devuelve la norma 1, ||A||_1 de una matriz. (La suma de la columna con mayor suma).

    Args:
        A (matriz): Matriz
        verbose (bool, optional): Imprime cosas. Defaults to False.
    """
    norma_1 = max(np.sum(abs(A), axis=0))
    print_verbose(f"La norma 1 es: {norma_1}.", verbose)  # Mayor columna
    return norma_1


def norma_inf(A, verbose=False):
    """Devuelve la norma 1, ||A||_1 de una matriz. (La suma de la fila con mayor suma).

    Args:
        A (matriz): Matriz
        verbose (bool, optional): Imprime cosas. Defaults to False.
    """
    norma_inf = max(np.sum(abs(A), axis=1))
    print_verbose(f"La norma inf es: {norma_inf}.", verbose)  # Mayor fila
    return norma_inf


def norma_2(A, verbose=False):
    """Devuelve la norma 1, ||A||_1 de una matriz. Se define como la raíz del mayor de los autovalores de A*A.T.

    Args:
        A (matriz): Matriz
        verbose (bool, optional): Imprime cosas. Defaults to False.
    """
    eigs = sqrt([simplify(i) for i in list((A.T * A).eigenvals().keys())])
    norma_2 = np.max(eigs)
    print_verbose(f"Los eigs de A.T * A (sqrt) son: {eigs}", verbose)
    print_verbose(f"La norma 2 es: {norma_2}", verbose)
    return norma_2


x = symbols('x')


def norma_p_func(expr, var=x, p=2, a=0, b=1):
    """Calcula la norma p de una función

    Args:
        expr (expr): funcion
        var (variable, optional): variable de la funcion. Defaults to x.
        p (int, optional): p. Defaults to 2.
        a (int, optional): limite inferior. Defaults to 0.
        b (int, optional): limite superior. Defaults to 1.

    Returns:
        expr: expresión de la norma
    """
    exp_p = abs(expr ** p) if p % 2 == 1 else expr ** p

    if a is None or b is None:
        integral = integrate((expr) ** p, var)
    else:
        integral = integrate((expr) ** p, (var, a, b))

    if p == 2:
        integral = simplify(sqrt(integral))
    elif p == 3:
        integral = simplify(cbrt(integral))
    else:
        integral = simplify(integral ** (1 / p))

    return integral


def norma_inf_func(expr, var=x, a=None, b=None, return_sols=False):
    """Calcula la norma infinita de una función

    Args:
        expr (expr): funcion
        var (variable, optional): variable de la funcion. Defaults to x.
        a (int, optional): limite inferior. Defaults to 0.
        b (int, optional): limite superior. Defaults to 1.
        return_sols (bool, optional): si True, devuelve las soluciones y la localizacion

    Returns:
        expr: expresión de la norma
    """
    fprime = expr.diff(var)

    solutions = [(xx, abs(expr.subs(var, xx))) for xx in solve(fprime, var)]
    if a is None or b is None:
        pass
    else:
        solutions = [i for i in solutions if (a <= i[0]) & (b >= i[0])]

    max_val = max([i[1] for i in solutions])

    return max_val, solutions


def matriz_inversa(M, thresh=0.1):
    """
    Calcula la inversa de manera natural con sympy. Si no se satisface que M**(-1) * M = I (mediante la norma de M**(-1) * M - I), entonces calculamos la inversa con numpy.
    Aviso! Que la inversa nueva tenga menor norma que la original, no significa que sea lo suficientemente aceptable como para poder efectuar cálculos con ella.

    Args:
        M (Matriz): Matriz para calcular la inversa
        thresh (float, optional): Valor límite de la norma de M.inv * M - I. Defaults to 0.1.

    Returns:
        Matriz: Matriz inversa
    """

    M_inv = M.inv()

    norma = (simplify(M_inv * M) - eye(M.shape[0])).norm()

    if norma > thresh:
        print('AVISO! La inversa es inestable.')
        M_inv_np = Matrix(inv(np.array(M, dtype=float)))
        norma_numpy = (simplify(M_inv_np * M) - eye(M.shape[0])).norm()

        if norma_numpy < norma:
            print('Devolvemos la inversa usando numpy.')
            return M_inv_np

    return M_inv

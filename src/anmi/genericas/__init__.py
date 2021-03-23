from sympy import sqrt, simplify
import numpy as np


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

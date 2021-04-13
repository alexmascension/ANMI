from sympy import simplify, zeros
from sympy import Matrix as mat
import numpy as np

from ..genericas import print_verbose, matriz_inversa


def criterio_radio_espectral(H, verbose=True):
    eigs = [simplify(i) for i in list(H.eigenvals().keys())]

    print_verbose("||Criterio de radio espectral||", verbose)
    try:
        print_verbose(
            f"El mayor autovalor es {np.max(np.array(eigs, dtype=float))}. Si ese valor es < 1 entonces los métodos iterativos convergen.",
            verbose,
        )
    except:
        print_verbose(
            f"Los autovalores son {eigs}. Si el mayor autovalor es < 1, entonces el método converge.",
            verbose,
        )


def criterio_diagonal_dominante(A, verbose=True):
    print_verbose(
        "||Criterio de Diagonal Dominante||\n Si la matriz es dominante por filas, los métodos de Jacobi y Gauss-Seidel convergen.",
        verbose,
    )
    A_abs = abs(A)
    try:
        np.array(A_abs, dtype=float)
        for r in range(A.shape[0]):
            diff = 2 * A_abs[r, r] - sum(A_abs[r, :])
            if diff <= 0:
                print_verbose(
                    f"La fila {r} NO es dominante por filas: diff = {diff}.", verbose
                )
                return
        print_verbose("La matriz CUMPLE EL CRITERIO DIAGONAL DOMINANTE", verbose)
    except:
        print_verbose(
            "La matriz tiene complejos o simbolos. Hay que verificar el criterio a mano.",
            verbose,
        )


def criterio_simetrica_definida_positiva(A, verbose=True):
    print_verbose(
        "||Criterio de Sim Def Pos||\n Si la matriz es simétrica y definida positiva, el método de Gauss-Seidel es convergente.",
        verbose,
    )
    if A != A.T:
        print_verbose("La matriz NO es simétrica.", verbose)
        return

    det_A = A.det()
    print_verbose(f"El determinante de A es {det_A}.", verbose)
    try:
        if float(det_A) > 0:
            print_verbose(
                "La matriz es DEFINIDA POSITIVA (el determinante es positivo).",
                verbose,
            )
            print_verbose("La matriz CUMPLE EL CRITERIO SIM DEF POS", verbose)
        else:
            print_verbose(
                "La matriz NO es DEFINIDA POSITIVA (el determinante no es positivo).",
                verbose,
            )
    except:
        print_verbose(
            "No podemos determinar la positividad porque hay símbolos o complejos.",
            verbose,
        )


def criterio_SOR(verbose):
    print_verbose(
        "||Criterio SOR||\n Si la matriz es simétrica y definida positiva y w in (0, 2) el método SOR es convergente.\nSi w no (0, 2) el método SOR no converge.",
        verbose,
    )


def criterio_m_matriz(A, verbose):
    print_verbose(
        "||Criterio M matriz||\n Si la A es M-matriz entonces las descomposiciones de Jacobi y Gauss-Seidel son convergentes.\nA^-1 >= 0\naij < 0 para todo i =/= j",
        verbose,
    )
    A_inv = matriz_inversa(A)

    try:
        np.array(A, dtype=float)
        if np.min(A_inv) >= 0:
            print_verbose("A^-1 >= 0", verbose)
        else:
            print_verbose("A^-1 < 0. La matriz NO CUMPLE el criterio", verbose)

        A_null_diag = A.copy()
        for i in range(A.shape[0]):
            A_null_diag[i, i] = 0

        if np.max(A_null_diag) > 0:
            print_verbose(
                "La matriz tiene elementos no diagonales positivos. NO CUMPLE el criterio.",
                verbose,
            )
        else:
            print_verbose("Los elementos no diagonales son negativos.", verbose)
    except:
        print_verbose(
            "La matriz tiene complejos o símbolos, no podemos verificar le criterio.",
            verbose,
        )


def metodo_iterativo(
    A, b=None, x0=None, metodo="jacobi", w=1.5, n_iter=10, verbose=True,
):
    """Aplica el método iterativo designado

    Args:
        A (matriz): Matriz de valores
        b (vector, optional): Vector de rhs. Por defecto es 1, 1, ..., 1.
        x0 (vector, optional): Vector con elementos de la primera iteración. Por defecto es 1, 1, ..., 1.
        metodo (str, optional): método de resolución, puede ser "jacobi", "gs" o "sor".
        w (float, optional): Peso para método sor. Defaults to 1.5.
        n_iter (int, optional): Número de iteraciones del método. Defaults to 10.
        verbose (bool, optional): Imprime resultados intermedios. Defaults to True.

    Returns:
        dict: 'x': vector de resultados para Ax=b, 'diff': diferencia entre Ax y b para cada iteración.
    """
    if b is None:
        b = mat([[1] * A.shape[0]]).T
    if x0 is None:
        x0 = mat([[1] * A.shape[1]]).T

    D, L, U = (
        zeros(A.shape[0], A.shape[1]),
        zeros(A.shape[0], A.shape[1]),
        zeros(A.shape[0], A.shape[1]),
    )
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if r == c:
                D[r, c] = A[r, c]
            elif r < c:
                U[r, c] = -A[r, c]
            else:
                L[r, c] = -A[r, c]

    if metodo == "jacobi":
        M = D
    elif metodo == "gs":
        M = D - L
    elif metodo == "sor":
        M = D / w - L

    N = simplify(M - A)

    # Aplicamos criterios!
    criterio_radio_espectral(matriz_inversa(M) * N, verbose)
    criterio_diagonal_dominante(A, verbose)
    criterio_simetrica_definida_positiva(A, verbose)
    criterio_SOR(verbose)
    criterio_m_matriz(A, verbose)

    diff = []
    for iter in range(n_iter):  # Aplica el método
        x0 = (matriz_inversa(M)) * (N * x0 + b)
        diff.append(np.sum(np.abs(A * x0 - b)))

    return {"x": x0, "diff": diff}

from sympy import symbols, poly, zeros, simplify
from anmi.T2 import factorizacion_QR


def matriz_krylov(A, x, n_iters=None):
    """Genera una matriz de krylov dada una matriz A y un vector x. Cada columna de la matriz es la iteración i de A^i*x.

    Args:
        A (matriz): Matriz de aplicación
        x (vector): Vector base
        n_iters (int, optional): Número de iteraciones. Por defecto es el número de filas de A + 1 (garantiza que la matriz de krylov tiene una combinación lineal).

    Returns:
        m_krylov: Matriz con las aplicaciones de krylov por columna.
    """
    if n_iters is None:
        n_iters = A.shape[0] + 1

    m_krylov = zeros(A.shape[0], n_iters)
    m_krylov[:, 0] = x

    for i in range(1, n_iters):
        m_krylov[:, i] = A * m_krylov[:, i - 1]

    return simplify(m_krylov)


def sucesion_krylov(A, x):
    """
    La sucesión de krylov viene dada como la expresión (ejemplo para 3x3)
    p(l) = -l³ + a2 * l² + a1 * l + a0
    donde a = [a0, a1, a2] y es la solución del sistema de ecuaciones
    [x.T, Ax.T, Ax².T] * a.T = Ax³.T

    Args:
        A (matriz): Matriz para aplicar Krylov
        x (vector): Vector base

    Returns:
        poli_krylov (poly): Polinomio para la matriz de Krylov.
        a (vector): vector de coeficientes.
    """
    m_krylov = matriz_krylov(A, x)

    a = (m_krylov[:, : A.shape[0]] ** (-1)) * (
        ((-1) ** (A.shape[0] + 1)) * m_krylov[:, -1]
    )
    a = simplify(a)

    lambda_ = symbols("lambda")
    poli_krylov = poly(((-1) ** A.shape[0]) * (lambda_ ** A.shape[0]) + a[0])

    for i in range(1, A.shape[0]):
        poli_krylov += poly(a[i] * (lambda_ ** i))

    return poli_krylov, a


def potencia_iterada(A, x, n_iters, devolver_ultimo=True):
    """Aplica el método de la potencia iterada para calcular el mayor autovalor de la matriz, usando el método de Krylov.

    Args:
        A (matriz): Matriz aplicación
        x (vector): Vector base para el método. Si el vector es autovector dará fallo.
        n_iters (int): Número de iteraciones
        devolver_ultimo (bool, optional): Si True, devuelve el vector de la última iteración. Si False, devuelve todas las iteraciones.

    Returns:
        m_cocientes (matriz, vector): matriz/vector con el número de filas igual al de A, con los cocientes. Los números deberían tender al mayor autovalor de A.
    """

    m_krylov = matriz_krylov(A, x, n_iters=n_iters)

    if devolver_ultimo:
        m_cocientes = zeros(m_krylov.shape[0], 1)
        for row in range(m_cocientes.shape[0]):
            m_cocientes[row, 0] = (
                m_krylov[row, m_krylov.shape[1] - 1]
                / m_krylov[row, m_krylov.shape[1] - 2]
            )

    else:
        m_cocientes = zeros(m_krylov.shape[0], m_krylov.shape[1] - 1)
        for col in range(m_cocientes.shape[1]):
            for row in range(
                m_cocientes.shape[0]
            ):  # No hay element-wise division en sympy
                m_cocientes[row, col] = m_krylov[row, col + 1] / m_krylov[row, col]

    m_cocientes = simplify(m_cocientes)
    return m_cocientes


def metodo_autovals_QR(A, n_iters=3, verbose=True):
    """Aplica el método QR para el cálculo de autovalores de una matriz.

    Args:
        A (matriz): Matriz para el metodo
        n_iters (int, optional): Número de iteraciones. Defaults to 3.
        verbose (bool, optional): Imprime información intermedia.

    Returns:
        dict: "A": lista de valores de A = R*Q en cada iteración, "R" Y "Q": matrices Q y R del método.
    """
    list_A, list_Q, list_R = [], [], []

    # La lista de A tiene los A^{(2)} hasta A^{(k+1)}, porque A^{(1)} = A, y ya lo conocemos
    for i in range(n_iters):
        dict_QR = factorizacion_QR(A, verbose=verbose)
        Q, R = simplify(dict_QR["Q"]), simplify(dict_QR["R"])
        list_Q.append(Q)
        list_R.append(R)

        A = simplify(R * Q).copy()
        list_A.append(A)

    return {"A": list_A, "Q": list_Q, "R": list_R}

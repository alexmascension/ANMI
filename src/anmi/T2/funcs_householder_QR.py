import numpy as np
from sympy import zeros, eye, simplify, sqrt
from sympy import Matrix as mat

from ..genericas import print_verbose, norma, matriz_inversa
from .funcs_gram_schmidt import gram_schmidt


def householder(x, y, signo="+", normalizar=False):
    """Aplica la transformación de Householder. x e y pueden ser vectores fila o columna,
    en cuyo caso aplicamos la transformación de una forma u otra.

    Args:
        x (vector): vector origen
        y (vector): vector destino
        signo (str, optional): + o -, si - aplica la transformación inversa. Defaults to "+".
        normalizar (bool, optional): Normaliza los vectores para la transformación. Defaults to False.

    Returns:
        dict: {'H': la aplicación de transformación, 'e': el vector de eje de cambio.}
    """
    if signo == "+":
        mult = 1
    else:
        mult = -1

    if normalizar:
        x = simplify(x / norma(x))
        y = simplify(y / norma(y))

    if simplify(norma(x)) != simplify(norma(y)):
        print(
            f"AVISO!!! x tiene norma {norma(x)} e y tiene norma {norma(y)}. "
            "Para Householder las normas tienen que ser iguales."
        )

    if simplify(norma(x - y)) == 0:  # x==y
        e = zeros(x.shape[0], x.shape[1])
    else:
        e = mult * (x - y) / (norma(x - y))

    if (e.shape[0] == 1) and (e.shape[1] >= 1):  # El vector es fila:
        H = eye(x.shape[1]) - 2 * e.T * e
    else:
        H = eye(x.shape[0]) - 2 * e * e.T

    if (e.shape[0] == 1) and (e.shape[1] >= 1) and (simplify(x * H) != y):
        print(
            "AVISO FILAS!!! xH != y. Para Householder las normas tienen que ser iguales."
        )
    elif (e.shape[1] == 1) and (e.shape[0] >= 1) and (simplify(H * x) != y):
        print("AVISO!!! Hx != y. Para Householder las normas tienen que ser iguales.")

    return simplify(H), simplify(e)


def factorizacion_QR(m, verbose=True, metodo="gram_schmidt"):
    """Aplica el método QR para la factorización de la matriz m. m = QR, con Q ortogonal y R triangular superior.

    Args:
        m (matriz): matriz a factorizar.
        verbose (bool, optional): Imprime mensajes informativos. Defaults to True.
        metodo (str, optional): "gram_schmidt" o "householder".
        Householder funciona mejor para vectores proximos,
        donde gram-schmidt deja de ser estable. Defaults to "gram_schmidt".

    Returns:
        dict: {'Q', 'R' , 'D'} ('D' en gram-schmidt solo).
    """
    if metodo == "gram_schmidt":
        print_verbose("Aplicamos QR con Gram Schmidt", verbose)

        dict_gs = gram_schmidt(m)
        P, C = dict_gs["P"], dict_gs["c"]

        D = zeros(m.shape[0], m.shape[1])
        for col in range(m.shape[0]):
            D[col, col] = sqrt((P[:, col].T * P[:, col])[0])

        Q = simplify(P * (matriz_inversa(D)))
        R = simplify(D * (eye(m.shape[0]) + C))

        print_verbose(
            f"Q es \n{np.array(Q)}\nR es \n{np.array(R)}\nD es \n{np.array(D)}", verbose
        )
    elif metodo == "householder":
        print_verbose("Aplicamos QR con Householder", verbose)
        lista_p = [eye(m.shape[0]) for _ in range(m.shape[0])]
        lista_e = []

        Q, R = eye(m.shape[0]), m.copy()
        for col in range(m.shape[1]):
            x = R[col:, col]
            y = mat([1] + [0] * (m.shape[0] - col - 1)) * norma(x)
            p_col = lista_p[col]
            H, e = householder(x, y)
            p_col[col:, col:] = H
            lista_e.append(e)

            R = simplify(p_col * R)
            Q = simplify(Q * p_col)

        print_verbose(f"Q es \n{np.array(Q)}\nR es \n{np.array(R)}", verbose)
    else:
        raise "El método no existe. Los disponibles son gram_schmidt y householder"

    if m != simplify(Q * R):
        print("AVISO!!! A != QR")

    if metodo == "gram_schmidt":
        return {"Q": Q, "R": R, "D": D}
    elif metodo == "householder":
        return {"Q": Q, "R": R}

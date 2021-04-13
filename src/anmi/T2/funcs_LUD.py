import numpy as np
from ..genericas import print_verbose, matriz_inversa
from sympy import zeros, eye, simplify, sqrt


def permutacion_matriz(U, fila_i, idx_max, verbose=False, P=None, r=None):
    """Efectua una permutación por filas de una matriz

    Args:
        U (matriz): MAtriz a permutar
        fila_i (int): indice de fila origen
        idx_max (int): indice de fila a la que permutar
        verbose (bool, optional): verbose. Defaults to False.
        P (mat, optional): matriz de permutación. Defaults to None.
        r (mat, optional): rhs de ecuación. Defaults to None.

    Returns:
        (U, P, r): Matrices U, P y r cambiadas
    """
    print_verbose(
        [
            f"Permutamos fila {fila_i} con {idx_max}",
            f"U antes:\n {np.array(U)}",
            f"P antes:\n {np.array(P)}",
        ],
        verbose,
    )

    if fila_i != idx_max:
        fila_origen, fila_destino = U[fila_i, :].copy(), U[idx_max, :].copy()
        U[idx_max, :], U[fila_i, :] = fila_origen, fila_destino
        if P is not None:
            fila_origen, fila_destino = P[fila_i, :].copy(), P[idx_max, :].copy()
            P[idx_max, :], P[fila_i, :] = fila_origen, fila_destino
        if r is not None:
            fila_origen, fila_destino = r[fila_i, :].copy(), r[idx_max, :].copy()
            r[idx_max, :], r[fila_i, :] = fila_origen, fila_destino

    print_verbose(
        [f"U despues:\n {np.array(U)}", f"P despues:\n {np.array(P)}"], verbose
    )

    return U, P, r


def permutacion_L(L, perm, verbose=False):
    """
    Esta función la creo porque a la hora de hacer la permutación, hay que
    permutar los elementos de L pero no directamente. Solo hay que seleccionar
    los elementos de la diagonal inferior que correspondan con el menor número de
    columnas. Por ejemplo, si permutamos las filas 3 y 5, se tienen que mover solo
    los elementos de las 2 primeras columnas (2 -> 3) para luego continuar
    con las transformaciones del resto de columnas.

    Args:
        L (matriz): Matriz L de LDU
        perm (list): lista con indices inicial y final de permutación
        verbose (bool, optional): Verbose

    Returns:
        matriz: matriz L con la permutación aplicada correctamente a los elementos triangulares.
    """

    print_verbose(f"L antes:\n {np.array(L)}", verbose)

    fila_origen, fila_destino = (
        L[perm[0], : min(perm)].copy(),
        L[perm[1], : min(perm)].copy(),
    )
    L[perm[1], : min(perm)], L[perm[0], : min(perm)] = fila_origen, fila_destino

    print_verbose(f"L despues:\n {np.array(L)}", verbose)

    return L


def descomposicion_LU(m, rhs=None, verbose=True, permutar_max=False):
    """Esta función realiza el algoritmo de triangulación de Gauss.
        Para ello vamos a ir aplicando paso a paso el algoritmo tal
        cual se hace manualmente,
        y aplicamos los cambios de columnas necesarios si hay
        que aplicar permutaciones.
        Por defecto, si encontramos un 0 en la diagonal aplicamos
        permutar_max para esa
        fila, y devolvemos la matriz de permutaciones.

    Args:
        m (matriz): Matriz a descomponer
        rhs (matriz, optional): Lado derecho de la ecuación. Default es [1, 1, ..., 1]
        verbose (bool, optional): Aplicar verbose.
        permutar_max (bool, optional): Aplicar permutación por filas aun si no es necesario. Defaults to False.

    Returns:
        dict: {"P": matriz de permutaciones, "L": matriz L, U": matriz U, "r": matriz rhs adecuada a U}
    """

    if rhs is None:
        rhs = zeros(m.shape[0], 1)

    print_verbose(
        [
            "La matriz M|X es  (X = 0) si no se ha introducido",
            np.concatenate((np.array(m), np.array(rhs)), axis=1),
        ],
        verbose,
    )

    P, L, U, r = eye(m.shape[0]), eye(m.shape[0]), m.copy(), rhs.copy()
    fila_i, err, err_max = 0, 0, 3

    while fila_i < m.shape[0] and err < err_max:
        print_verbose(
            [
                f"\n=====================================\nFila {fila_i}",
                f"A {fila_i}, {err}",
            ],
            verbose,
        )

        if U[max(0, fila_i - 1), max(0, fila_i - 1)] == 0 or permutar_max:
            """
      Esta parte nos asegura que si un elemento diagonal es cero, permutamos
      la fila con su inmediata inferior y rehacemos los cálculos,
      y así tener una configuración viable.
      En cualquier permutación tenemos que cambiar la L acorde con el cambio.
      Al hacer la permutación, con los nuevos valores, repasamos la matriz para
      asegurarnos de que todos los puntos están cumplidos.

      Para la opción de permutar, buscamos un elemento de las filas no
      alteradas que sea el mayor. Si hay más de una fila, y una de
      ellas es la actual, no aplica la permutación.
      """
            sub_mat = U[fila_i:, fila_i:]
            max_el = np.max(np.array(sub_mat))
            idx_max = (
                np.min([i[0] for i in np.argwhere(np.array(sub_mat) == max_el)])
                + fila_i
            )  # Cogemos el primer elemento si hay varios

            if U[max(0, fila_i - 1), max(0, fila_i - 1)] == 0:
                U, P, r = permutacion_matriz(
                    U, max(0, fila_i - 1), idx_max, verbose, P, r
                )
                L = permutacion_L(L, [max(0, fila_i - 1), idx_max], verbose)
                fila_i = min(max(0, fila_i - 1), idx_max)

                err += 1
                print_verbose(f"Err {err}", verbose)
                continue

            elif permutar_max:
                if idx_max <= fila_i:  # No es necesario hacer la permutación
                    print_verbose(
                        "El índice de permutación es igual a la fila a permutar.",
                        verbose,
                    )
                else:
                    U, P, r = permutacion_matriz(U, fila_i, idx_max, verbose, P, r)
                    L = permutacion_L(L, [fila_i, idx_max], verbose)
                    continue

        # Ahora aplicamos el algoritmo de calculo de filas:
        for columna_j in range(fila_i):
            a_ij = U[fila_i, columna_j] / U[columna_j, columna_j]

            if a_ij != 0:
                L[fila_i, columna_j] = a_ij
                U[fila_i, :] = U[fila_i, :] - a_ij * U[columna_j, :]
                r[fila_i, :] = r[fila_i, :] - a_ij * r[columna_j, :]

            err = 0

            print_verbose(
                [
                    f"||||||||||||||||||||||||\n Columna {columna_j}",
                    f"a_{fila_i},{columna_j} = {a_ij}",
                    f"P = \n{np.array(P)}",
                    f"L = \n{np.array(L)}",
                    f"U = \n{np.array(U)}",
                    f"r = \n{np.array(r)}",
                ],
                verbose,
            )

        fila_i += 1

    if err == err_max:
        print("Algo ha ido mal... mira el log.")

    print_verbose(
        [
            "================= FORMA FINAL",
            f"a_{fila_i},{columna_j} = {a_ij}",
            f"P = \n{np.array(P)}\n",
            f"L = \n{np.array(L)}\n",
            f"U = \n{np.array(U)}\n",
            f"r = \n{np.array(r)}\n",
        ],
        verbose,
    )

    if simplify(L * U) != simplify(P * m):
        print("AVISO!!! LU != PA")

    return {"P": simplify(P), "L": simplify(L), "U": simplify(U), "r": simplify(r)}


def descomposicion_LDU(m, permutar_max=True, verbose=False):
    """Aplica la descomposición LDU = Pm a una matriz m.

    Args:
        m (matriz): matriz a permutar
        **args: args de descomposicion_LU

    Returns:
        dict: {"P": simplify(dict_LU["P"]),
        "L": simplify(L),
        "U": simplify(U),
        "D": simplify(D),}
    """
    dict_LU = descomposicion_LU(m, permutar_max=permutar_max, verbose=verbose)
    L, U = dict_LU["L"], dict_LU["U"]

    D = zeros(m.shape[0], m.shape[0])

    for i in range(U.shape[0]):
        D[i, i] = U[i, i]

    U = matriz_inversa(D) * U

    print_verbose(
        [
            "================ RESULTADOS LDU*",
            f"L = \n{np.array(L)}\n",
            f"D = \n{np.array(D)}\n",
            f"U* = \n{np.array(U)}\n",
        ],
        verbose,
    )

    if simplify(L * D * U) != simplify(dict_LU["P"] * m):
        print("AVISO!!! LDU != PA")

    return {
        "P": simplify(dict_LU["P"]),
        "L": simplify(L),
        "U": simplify(U),
        "D": simplify(D),
    }


def cholesky(m, verbose=False):
    """
  Primero comprobamos que los menores sean positivos. Eso es equivalente que
  sus autovalores sean positivos. Por simplificar, si el determinante es negativo
  ya descartamos que sea factorizable, y saltamos el warning.
  """

    if m != m.T:
        print(
            "AVISO! La matriz no es simétrica, y por tanto no factorizable por Cholesky."
        )

    print_verbose(f"|M| es {m.det()}. Si es < 0, no es factorizable.", verbose)

    m_chol = zeros(m.shape[0], m.shape[1])

    for col_j in range(m.shape[0]):
        for row_i in range(col_j, m.shape[0]):
            if col_j == 0:
                if row_i == 0:
                    m_chol[row_i, col_j] = sqrt(m[row_i, col_j])
                else:
                    m_chol[row_i, col_j] = m[row_i, col_j] / m_chol[0, 0]

            else:
                if col_j == row_i:
                    m_chol[row_i, col_j] = sqrt(
                        m[col_j, col_j]
                        - sum([m_chol[col_j, k] ** 2 for k in range(col_j)])
                    )
                else:
                    m_chol[row_i, col_j] = (
                        m[row_i, col_j]
                        - sum(
                            [m_chol[col_j, k] * m_chol[row_i, k] for k in range(col_j)]
                        )
                    ) / (m_chol[col_j, col_j])

    if simplify(m_chol * m_chol.T) != m:
        print(
            f"AVISO!!! La matriz no es Cholesky-zable. \n L = \n {np.array(m_chol)}"
            "\n\n L*L.T = \n {np.array(m_chol * m_chol.T)}"
        )

    return simplify(m_chol)

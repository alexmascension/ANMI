from ..genericas import print_verbose
from sympy import zeros, simplify


def suma_columnas(lista):
    if len(lista) > 0:
        m = zeros(lista[0].shape[0], 1)
        for i in lista:
            m += i
        return m
    else:
        return 0


def gram_schmidt(m, verbose=False):
    """
  La ortogonalización produce una matriz ortogonalizada por columnas.
  p es la matriz ortogonal y p_norm es la ortonormal
  c es la matriz triangular tal que cij = aj·pi/||pi||^2 (los coeficientes de ortogonalización).
  Estos coeficientes se usan para la factorización QR.
  """

    p = zeros(m.shape[0], m.shape[1])

    p[:, 0] = m[:, 0]

    for col in range(1, m.shape[1]):
        p[:, col] = m[:, col] - suma_columnas(
            [
                (m[:, col].T * p[:, i])[0] / (p[:, i].T * p[:, i])[0] * p[:, i]
                for i in range(0, col)
            ]
        )

    print_verbose(f"La matriz ortogonal es \n {p}", verbose)

    p_norm = zeros(m.shape[0], m.shape[1])
    for col in range(p.shape[1]):
        p_norm[:, col] = p[:, col] / (p[:, col].T * p[:, col])[0]

    print_verbose(f"La matriz ortonormal es \n {p_norm}", verbose)

    c = zeros(m.shape[0], m.shape[1])
    for col in range(1, m.shape[1]):
        for row in range(0, col):
            c[row, col] = (m[:, col].T * p[:, row])[0] / (p[:, row].T * p[:, row])[0]

    return {"P": simplify(p), "Pn": simplify(p_norm), "c": simplify(c)}

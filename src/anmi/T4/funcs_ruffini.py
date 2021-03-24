from sympy.matrices import Matrix as mat
from sympy import Poly


def metodo_ruffini(poli, x0):
    """Método de Ruffini. Aplica el método para la evaluación de un polinomio.

    Args:
        poli (poli): Polinomio a evaluar
        x0 (int, float): Punto en el que evaluar el polinomio

    Returns:
        dict: 'eval' - evaluación del polinomio, 'mat': matriz de ruffini
    """
    try:
        poli_coeffs = poli.all_coeffs()
    except AttributeError:
        poli = Poly(poli)
        poli_coeffs = poli.all_coeffs()

    varx = poli.gens[0]

    mat_ruf = mat([poli_coeffs, [0] * len(poli_coeffs), [0] * len(poli_coeffs)])

    for i in range(len(poli_coeffs)):
        mat_ruf[2, i] = mat_ruf[1, i] + mat_ruf[0, i]

        try:
            mat_ruf[1, i + 1] = x0 * mat_ruf[2, i]
        except IndexError:  # Ultimo elemento
            pass

    return {'eval': mat_ruf[2, -1], 'mat': mat_ruf}

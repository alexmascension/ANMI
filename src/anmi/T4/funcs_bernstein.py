from sympy import symbols, S, binomial, expand


def polinomios_bernstein(f, var=symbols('x'), I=[0, 1], grado=2):
    """Genera el polinomio de Bernstein de grado n para la función f.

    Args:
        f (funcion): Función a aproximar
        var (variable, optional): Variable de f. Defaults to symbols('x').
        I (list, optional): Intervalo de aproximación. Defaults to [0, 1].
        grado (int, optional): Grado del polinomio. Defaults to 2.

    Returns:
        funcion: Polinomio de aproximación
    """
    # Los polinomios funcionan en un intervalo de 0 a 1, así que si el intervalo base es [a, b] tenemos que
    # aplicar el cambio t = (x - a)/(b - a) -> (b-a)t + a = x
    # Así transformamos f(x) en g(t)

    # Si el polinomio Bngt = sum_i^n (n i) g(i/n) t^i (1-t)^(n-1)

    vart = var * (I[1] - I[0]) + I[0]
    g = f.subs(var, vart)

    B_nft = S(0)
    for grado_i in range(grado + 1):
        poli_i = binomial(grado, grado_i) * (var ** grado_i) * ((1 - var) ** (grado - grado_i)) * g.subs(var, S(grado_i) / S(grado))
        B_nft += poli_i

    B_nf = B_nft.subs(var, S(var - S(I[0])) / S(I[1] - I[0]))

    return expand(B_nf)

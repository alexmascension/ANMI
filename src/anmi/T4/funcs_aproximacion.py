from sympy import simplify, integrate, zeros, S, Matrix, symbols, Integral
from ..genericas import matriz_inversa


def producto_deriv(f, g, var=symbols('x'), a=0, b=1, I=None, numeric=False):
    """Aplica el producto escalar <f,g> = ∫ f.g + f'.g'

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
    fg = f * g + f.diff(var) * g.diff(var)

    if I is None:  # aplica el modo continuo
        if numeric:
            return Integral(fg(var, a, b)).evalf()
        else:
            return simplify(integrate(fg, (var, a, b)))
    else:
        sum_I = S(0)  # Para hacerlo como objeto de sympy
        for i in I:
            sum_I += fg.subs(var, i)
        return simplify(sum_I)


def producto_asecas(f, g, var=symbols('x'), a=0, b=1, I=None, numeric=False):
    """Aplica el producto escalar <f,g> = ∫ f.g

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
    fg = f * g

    if I is None:  # aplica el modo continuo
        if numeric:
            return Integral(fg(var, a, b)).evalf()
        else:
            return simplify(integrate(fg, (var, a, b)))
    else:
        sum_I = S(0)  # Para hacerlo como objeto de sympy
        for i in I:
            sum_I += fg.subs(var, i)
        return simplify(sum_I)


def producto_escalar_peso(f, g, var=symbols('x'), w=S(1), a=0, b=1, I=None, numeric=False):
    """Aplica el producto escalar <f,g> = ∫ f.g.w, donde w es una función de peso determinada

    Args:
        f (funcion): f
        g (funcion): g
        var (variable): variable de integración
        w (funcion): función de peso.
        a (int, optional): limite inferior de integracion. Defaults to 0.
        b (int, optional): limite superior de integracion. Defaults to 1.
        I (list, optional): Si no es None, lista de valores sobre los que hacer un sumatorio discreto. Defaults to None.
        numeric (bool, optional): si True, realiza una aproximación numérica de la integral usando un método de sympy.

    Returns:
        funcion, float: Valor del producto escalar. Se devuelve como funcion si tiene variables.
    """

    fg = simplify(f * g * w)
    if I is None:  # aplica el modo continuo
        if numeric:
            return Integral(fg, (var, a, b)).evalf()
        else:
            return simplify(integrate(fg, (var, a, b)))
    else:
        sum_I = S(0)  # Para hacerlo como objeto de sympy
        for i in I:
            sum_I += fg.subs(var, i)
        return simplify(sum_I)


def metodo_gram(f, U, var=symbols('x'), func_producto=producto_deriv, a=0, b=1, I=None, **func_producto_kwargs):
    """Aplica el método de gram para aplicar la aproximación de una función f empleando una base de funciones U.

    Args:
        f (funcion): función a aproximar
        U (list): lista de funciones que establecen la base de aproximación
        var (variable): variable de f
        func_producto (funcion): función con la que aplicar los productos escalares de funciones.
        a (int, optional): límite inferior de integración para el producto escalar. Defaults to 0.
        b (int, optional): límite superior de integración para el producto escalar. Defaults to 1.
        I (list, optional): lista de valores discretos para el producto escalar si queremos hacer una aproximación discreta. Defaults to None.

    Returns:
        dict: {'poly': polinomio de aproximación en base a la base U, 'alpha': vector alpha con los coeficientes optimizados para U, 'f_bar': matriz de productos escalares entre f y U, 'G': matriz de gram con los productos escalares de U}
    """
    G = zeros(len(U), len(U))
    f_bar = zeros(len(U), 1)

    for idx_i, u_i in enumerate(U):
        for idx_j, u_j in enumerate(U):
            if idx_i >= idx_j:
                g_ij = func_producto(u_i, u_j, var=var, a=a, b=b, I=I, **func_producto_kwargs)
                G[idx_i, idx_j] = g_ij
                G[idx_j, idx_i] = g_ij

    for idx_i, u_i in enumerate(U):
        f_i = func_producto(f, u_i, var=var, a=a, b=b, I=I, **func_producto_kwargs)
        f_bar[idx_i, 0] = f_i

    try:
        alpha = simplify(matriz_inversa(G) * f_bar)
    except:
        print('AVISO!!! La matriz G no es invertible. Aplicamos la pseudoinversa')
        from sympy.matrices.inverse import _pinv  # Aplica la pseudoinversa de Moore Penrose
        alpha = simplify(_pinv(G) * f_bar)

    expr_pol = simplify((alpha.T * Matrix([U]).T)[0])  # producto entre alpha y la base U

    return {'poly': expr_pol, 'alpha': alpha, 'f_bar': f_bar, 'G': G}


def gram_schmidt_f(base, var=symbols('x'), prod_esc=producto_asecas, *args, **kwargs):
    """Aplica el método de ortogonalización de Gram Schmidt para funciones. Dada una lista de funciones base, y un producto escalar de funciones, 
    genera una nueva base de funciones que son ortogonales entre ellas para ese producto.

    Args:
        base (list): Lista de funciones de la base.
        var (variable): Variable de la lista de funciones
        prod_esc (funcion): Función de producto escalar a usar
        *args, **kwargs: argumentos de prod_esc

    Returns:
        list_pols_GS (list): Lista de funciones ortogonales
    """

    list_pols_GS = []
    for i in range(len(base)):
        poli = base[i]
        for j in range(i):
            pj = list_pols_GS[j]
            poli -= (prod_esc(pj, base[i], var, *args, **kwargs)) / (prod_esc(pj, pj, var, *args, **kwargs)) * pj

        list_pols_GS.append(poli)
    return list_pols_GS


def polinomios_orto_peso(base, w=S(1), var=symbols('x'), prod_esc=producto_escalar_peso, a=-1, b=1, I=None, numeric=False):
    """Partiendo de una base ortiginal, genera una serie de funciones, también ortogonales, a las que se les aplica un proceso extra de ortogonalización con la funcion de peso.

    Args:
        base (list): Lista de funciones de la base.
        w (funcion, optional): Función de peso para el producto escalar. Defaults to S(1).
        var (variable): Variable de la lista de funciones
        prod_esc (funcion): Función de producto escalar a usar
        a (int, optional): límite inferior de integración para el producto escalar. Defaults to 0.
        b (int, optional): límite superior de integración para el producto escalar. Defaults to 1.
        I (list, optional): lista de valores discretos para el producto escalar si queremos hacer una aproximación discreta. Defaults to None.
        numeric (bool, optional): si True, realiza una aproximación numérica de la integral usando un método de sympy.

    Returns:
        dict: {'p': base nueva de polinomios ortogonales con peso, 'a': coeficientes a para los polinomios, 'b': coeficiones b para los polinomios}
    """
    list_p, list_a, list_b = [], [], []

    # n = 0
    list_p.append(S(1))
    list_a.append(S(0))
    list_b.append(S(0))

    # n = 1
    a_i = simplify(prod_esc(base[0] * var, base[0], var=var, w=w, a=a, b=b, I=None, numeric=False) /
                   prod_esc(base[0], base[0], var=var, w=w, a=a, b=b, I=None, numeric=False))
    list_p.append(var - a_i)
    list_a.append(a_i)
    list_b.append(S(0))

    # n > 1
    for i in range(2, len(base)):
        a_i = simplify(prod_esc(base[i - 1] * var, base[i - 1], var=var, w=w, a=a, b=b, I=None, numeric=False) /
                       prod_esc(base[i - 1], base[i - 1], var=var, w=w, a=a, b=b, I=None, numeric=False))

        b_i = simplify(prod_esc(base[i - 1] * var, base[i - 2], var=var, w=w, a=a, b=b, I=None, numeric=False) /
                       prod_esc(base[i - 2], base[i - 2], var=var, w=w, a=a, b=b, I=None, numeric=False))

        list_a.append(a_i)
        list_b.append(b_i)
        list_p.append(simplify((var - a_i) * list_p[i - 1] - b_i * list_p[i - 2]))

    return {'p': list_p, 'a': list_a, 'b': list_b}

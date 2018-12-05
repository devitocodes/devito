def make_ops_ast(expr, nfops):
    """
    Transform an devito expression into an OPS expression.
    Only the interested nodes are changed.

    Parameters
    ----------
    expr : :class:`Node`
        Initial tree node.
    nfops : :class:`OPSNodeFactory`
        Generate OPS specific nodes.

    Returns
    -------
    :class:`Node`
        Expression alredy translated to OPS syntax.
    """

    if expr.is_Symbol or expr.is_Number:
        return expr
    elif expr.is_Indexed:
        return nfops.new_grid(expr)
    else:
        return expr.func(*[make_ops_ast(i, nfops) for i in expr.args])

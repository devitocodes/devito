def make_ops_ast(expr, nfops, is_write=False):
    """
    Transform a devito expression into an OPS expression.
    Only the interested nodes are rebuilt.

    Parameters
    ----------
    expr : Node
        Initial tree node.
    nfops : OPSNodeFactory
        Generate OPS specific nodes.

    Returns
    -------
    Node
        Expression alredy translated to OPS syntax.
    """

    if expr.is_Symbol or expr.is_Number:
        return expr
    elif expr.is_Indexed:
        return nfops.new_ops_arg(expr, is_write)
    elif expr.is_Equality:
        res = expr.func(
            make_ops_ast(expr.lhs, nfops, True),
            make_ops_ast(expr.rhs, nfops)
        )
        return res
    else:
        return expr.func(*[make_ops_ast(i, nfops) for i in expr.args])

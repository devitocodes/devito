from devito.equation import Eq

__all__ = ['to_expr']


def to_expr(expr):
    """
    Filter a Differential expression to return its sympy Expression
    """
    if expr.is_Equality:
        return Eq(to_expr(expr.lhs), to_expr(expr.rhs), region=expr._region)
    elif hasattr(expr, '_expr'):
        return expr._expr
    elif expr.is_Function:
        return expr
    elif expr.is_Atom:
        return expr
    else:
        return expr.func(*[to_expr(a) for a in expr.args])

import sympy

from devito.equation import Eq
from devito.symbolics.manipulation import pow_to_mul
__all__ = ['to_expr']

def to_expr(expr):
    if expr.is_Equality:
        return Eq(to_expr(expr.lhs), to_expr(expr.rhs))
    elif expr.is_Function:
        return expr
    elif hasattr(expr, '_expr'):
        return expr._expr
    elif expr.is_Atom:
        return expr
    else:
        return expr.func(*[to_expr(a) for a in expr.args])

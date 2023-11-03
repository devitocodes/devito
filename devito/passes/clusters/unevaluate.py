import sympy

from devito.ir import cluster_pass
from devito.symbolics import reuse_if_untouched, q_leaf
from devito.symbolics.unevaluation import Add, Mul, Pow

__all__ = ['unevaluate']


@cluster_pass
def unevaluate(cluster):
    exprs = [_unevaluate(e) for e in cluster.exprs]

    return cluster.rebuild(exprs=exprs)


mapper = {
    sympy.Add: Add,
    sympy.Mul: Mul,
    sympy.Pow: Pow
}


def _unevaluate(expr):
    if q_leaf(expr):
        return expr

    args = [_unevaluate(a) for a in expr.args]

    try:
        return mapper[expr.func](*args)
    except KeyError:
        return reuse_if_untouched(expr, args)

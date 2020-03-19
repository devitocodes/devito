from sympy import Add, Mul, collect, collect_const

from devito.passes.clusters.utils import cluster_pass
from devito.symbolics import estimate_cost, retrieve_scalars
from devito.tools import ReducerMap

__all__ = ['factorize']


MIN_COST_FACTORIZE = 100
"""
Minimum operation count of an expression so that aggressive factorization
is applied.
"""


@cluster_pass
def factorize(cluster, *args):
    """
    Factorize trascendental functions, symbolic powers, numeric coefficients.

    If the expression has an operation count greater than ``MIN_COST_FACTORIZE``,
    then the algorithm is applied recursively until no more factorization
    opportunities are detected.
    """
    processed = []
    for expr in cluster.exprs:
        handle = _collect_nested(expr)
        cost_handle = estimate_cost(handle)

        if cost_handle >= MIN_COST_FACTORIZE:
            handle_prev = handle
            cost_prev = estimate_cost(expr)
            while cost_handle < cost_prev:
                handle_prev, handle = handle, _collect_nested(handle)
                cost_prev, cost_handle = cost_handle, estimate_cost(handle)
            cost_handle, handle = cost_prev, handle_prev

        processed.append(handle)

    return cluster.rebuild(processed)


def _collect_nested(expr):
    """
    Collect numeric coefficients, trascendental functions, and symbolic powers,
    across all levels of the expression tree.

    The collection gives precedence to (in order of importance):

        1) Trascendental functions,
        2) Symbolic powers,
        3) Numeric coefficients.

    Parameters
    ----------
    expr : expr-like
        The expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Number:
            return expr, {'coeffs': expr}
        elif expr.is_Function:
            return expr, {'funcs': expr}
        elif expr.is_Pow:
            return expr, {'pows': expr}
        elif expr.is_Symbol or expr.is_Indexed or expr.is_Atom:
            return expr, {}
        elif expr.is_Add:
            args, candidates = zip(*[run(arg) for arg in expr.args])
            candidates = ReducerMap.fromdicts(*candidates)

            funcs = candidates.getall('funcs', [])
            pows = candidates.getall('pows', [])
            coeffs = candidates.getall('coeffs', [])

            # Functions/Pows are collected first, coefficients afterwards
            # Note: below we use sets, but SymPy will ensure determinism
            args = set(args)
            w_funcs = {i for i in args if any(j in funcs for j in i.args)}
            args -= w_funcs
            w_pows = {i for i in args if any(j in pows for j in i.args)}
            args -= w_pows
            w_coeffs = {i for i in args if any(j in coeffs for j in i.args)}
            args -= w_coeffs

            # Collect common funcs
            w_funcs = collect(expr.func(*w_funcs), funcs, evaluate=False)
            try:
                w_funcs = Add(*[Mul(k, collect_const(v)) for k, v in w_funcs.items()])
            except AttributeError:
                assert w_funcs == 0

            # Collect common pows
            w_pows = collect(expr.func(*w_pows), pows, evaluate=False)
            try:
                w_pows = Add(*[Mul(k, collect_const(v)) for k, v in w_pows.items()])
            except AttributeError:
                assert w_pows == 0

            # Collect common temporaries (r0, r1, ...)
            w_coeffs = collect(expr.func(*w_coeffs), tuple(retrieve_scalars(expr)),
                               evaluate=False)
            try:
                w_coeffs = Add(*[Mul(k, collect_const(v)) for k, v in w_coeffs.items()])
            except AttributeError:
                assert w_coeffs == 0

            # Collect common coefficients
            w_coeffs = collect_const(w_coeffs)

            rebuilt = Add(w_funcs, w_pows, w_coeffs, *args)

            return rebuilt, {}
        elif expr.is_Mul:
            args, candidates = zip(*[run(arg) for arg in expr.args])
            return Mul(*args), ReducerMap.fromdicts(*candidates)
        elif expr.is_Equality:
            args, candidates = zip(*[run(expr.lhs), run(expr.rhs)])
            return expr.func(*args, evaluate=False), ReducerMap.fromdicts(*candidates)
        else:
            args, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*args), ReducerMap.fromdicts(*candidates)

    return run(expr)[0]

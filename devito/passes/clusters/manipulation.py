from sympy import Add, Mul, collect, collect_const

from devito.symbolics import retrieve_scalars
from devito.tools import ReducerMap

__all__ = ['collect_nested']


def collect_nested(expr):
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

            # Always collect coefficients
            rebuilt = collect_const(expr.func(*args))
            try:
                if rebuilt.args:
                    # Note: Mul(*()) -> 1, and since sympy.S.Zero.args == (),
                    # the `if` prevents turning 0 into 1
                    rebuilt = Mul(*rebuilt.args)
            except AttributeError:
                pass

            return rebuilt, ReducerMap.fromdicts(*candidates)
        elif expr.is_Equality:
            args, candidates = zip(*[run(expr.lhs), run(expr.rhs)])
            return expr.func(*args, evaluate=False), ReducerMap.fromdicts(*candidates)
        else:
            args, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*args), ReducerMap.fromdicts(*candidates)

    return run(expr)[0]

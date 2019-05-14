from collections import OrderedDict

from sympy import collect, collect_const

from devito.ir import FlowGraph
from devito.symbolics import (Add, Mul, Eq, count, estimate_cost, q_op, q_leaf,
                              xreplace_constrained)
from devito.tools import ReducerMap

__all__ = ['collect_nested', 'common_subexprs_elimination', 'compact_temporaries']


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

            # Functions have precedence over coefficients
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
                w_funcs = Add(*[Mul(*i) for i in w_funcs.items()])
            except AttributeError:
                assert w_funcs == 0

            # Collect common pows
            w_pows = collect(expr.func(*w_pows), pows, evaluate=False)
            try:
                w_pows = Add(*[Mul(*i) for i in w_pows.items()])
            except AttributeError:
                assert w_pows == 0

            # Collect common coefficients
            w_coeffs = collect_const(expr.func(*w_coeffs))

            rebuilt = Add(w_funcs, w_pows, w_coeffs, *args)

            return rebuilt, {}
        elif expr.is_Mul:
            args, candidates = zip(*[run(arg) for arg in expr.args])

            # Always collect coefficients
            rebuilt = collect_const(expr.func(*args))
            try:
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


def common_subexprs_elimination(exprs, make, mode='default'):
    """
    Perform common sub-expressions elimination, or CSE.

    Note: the output is not guranteed to be topologically sorted.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions to which CSE is applied.
    make : callable
        Build symbols to store temporary, redundant values.
    mode : str, optional
        The CSE algorithm applied. Accepted: ['default'].
    """

    # Note: not defaulting to SymPy's CSE() function for three reasons:
    # - it also captures array index access functions (eg, i+1 in A[i+1] and B[i+1]);
    # - it sometimes "captures too much", losing factorization opportunities;
    # - very slow
    # TODO: a second "sympy" mode will be provided, relying on SymPy's CSE() but
    # also ensuring some sort of post-processing
    assert mode == 'default'  # Only supported mode ATM

    processed = list(exprs)
    mapped = []
    while True:
        # Detect redundancies
        counted = count(mapped + processed, q_op).items()
        targets = OrderedDict([(k, estimate_cost(k)) for k, v in counted if v > 1])
        if not targets:
            break

        # Create temporaries
        hit = max(targets.values())
        picked = [k for k, v in targets.items() if v == hit]
        mapper = OrderedDict([(e, make()) for i, e in enumerate(picked)])

        # Apply replacements
        processed = [e.xreplace(mapper) for e in processed]
        mapped = [e.xreplace(mapper) for e in mapped]
        mapped = [Eq(v, k) for k, v in reversed(list(mapper.items()))] + mapped

        # Prepare for the next round
        for k in picked:
            targets.pop(k)
    processed = mapped + processed

    # Simply renumber the temporaries in ascending order
    mapper = {i.lhs: j.lhs for i, j in zip(mapped, reversed(mapped))}
    processed = [e.xreplace(mapper) for e in processed]

    return processed


def compact_temporaries(temporaries, leaves):
    """Drop temporaries consisting of single symbols."""
    exprs = temporaries + leaves
    targets = {i.lhs for i in leaves}

    g = FlowGraph(exprs)

    mapper = {k: v.rhs for k, v in g.items()
              if v.is_Scalar and
              (q_leaf(v.rhs) or v.rhs.is_Function) and
              not v.readby.issubset(targets)}

    processed = []
    for k, v in g.items():
        if k not in mapper:
            # The temporary /v/ is retained, and substitutions may be applied
            handle, _ = xreplace_constrained(v, mapper, repeat=True)
            assert len(handle) == 1
            processed.extend(handle)

    return processed

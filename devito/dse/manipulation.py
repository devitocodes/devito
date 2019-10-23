from collections import OrderedDict

from sympy import Add, Mul, collect, collect_const

from devito.dse.flowgraph import FlowGraph
from devito.symbolics import (Eq, count, estimate_cost, q_xop, q_leaf, retrieve_scalars,
                              retrieve_terminals, xreplace_constrained)
from devito.tools import DAG, ReducerMap, split

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


def common_subexprs_elimination(exprs, make, mode='default'):
    """
    Perform common sub-expressions elimination, or CSE.

    Note: the output is guranteed to be topologically sorted.

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
        counted = count(mapped + processed, q_xop).items()
        targets = OrderedDict([(k, estimate_cost(k, True)) for k, v in counted if v > 1])
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

    # At this point we may have useless temporaries (e.g., r0=r1). Let's drop them.
    processed = compact_temporaries(processed)

    # Perform topological sorting so that reads-after-writes are honored
    processed = topological_sort(processed)

    return processed


def compact_temporaries(exprs):
    """Drop temporaries consisting of isolated symbols."""
    graph = FlowGraph(exprs)

    mapper = {k: v.rhs for k, v in graph.items()
              if v.is_Scalar and (q_leaf(v.rhs) or v.rhs.is_Function)}

    processed = []
    for k, v in graph.items():
        if k not in mapper:
            # The temporary /v/ is retained, and substitutions may be applied
            handle, _ = xreplace_constrained(v, mapper, repeat=True)
            assert len(handle) == 1
            processed.extend(handle)

    return processed


def topological_sort(exprs):
    """Topologically sort the temporaries in a list of equations."""
    mapper = {e.lhs: e for e in exprs}
    assert len(mapper) == len(exprs)  # Expect SSA

    # Build DAG and topologically-sort temporaries
    temporaries, tensors = split(exprs, lambda e: not e.lhs.is_Indexed)
    dag = DAG(nodes=temporaries)
    for e in temporaries:
        for r in retrieve_terminals(e.rhs):
            if r not in mapper:
                continue
            elif mapper[r] is e:
                # Avoid cyclic dependences, such as
                # Eq(f, f + 1)
                continue
            elif r.is_Indexed:
                # Only scalars enforce an ordering
                continue
            else:
                dag.add_edge(mapper[r], e, force_add=True)
    processed = dag.topological_sort()

    # Append tensor equations at the end in user-provided order
    processed.extend(tensors)

    return processed

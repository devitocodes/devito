from collections import OrderedDict

from sympy import collect, collect_const

from devito.ir import FlowGraph
from devito.symbolics import Eq, count, estimate_cost, q_op, q_leaf, xreplace_constrained
from devito.tools import as_mapper, flatten

__all__ = ['collect_nested', 'common_subexprs_elimination', 'compact_temporaries',
           'cross_cluster_cse']


def collect_nested(expr, aggressive=False):
    """
    Collect terms appearing in expr, checking all levels of the expression tree.

    :param expr: the expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Number or expr.is_Symbol:
            return expr, [expr]
        elif expr.is_Indexed or expr.is_Atom:
            return expr, []
        elif expr.is_Add:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])

            w_numbers = [i for i in rebuilt if any(j.is_Number for j in i.args)]
            wo_numbers = [i for i in rebuilt if i not in w_numbers]

            w_numbers = collect_const(expr.func(*w_numbers))
            wo_numbers = expr.func(*wo_numbers)

            if aggressive is True and wo_numbers:
                for i in flatten(candidates):
                    wo_numbers = collect(wo_numbers, i)

            rebuilt = expr.func(w_numbers, wo_numbers)
            return rebuilt, []
        elif expr.is_Mul:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            rebuilt = collect_const(expr.func(*rebuilt))
            return rebuilt, flatten(candidates)
        elif expr.is_Equality:
            rebuilt, candidates = zip(*[run(expr.lhs), run(expr.rhs)])
            return expr.func(*rebuilt, evaluate=False), flatten(candidates)
        else:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*rebuilt), flatten(candidates)

    return run(expr)[0]


def common_subexprs_elimination(exprs, make, mode='default'):
    """
    Perform common subexpressions elimination.

    Note: the output is not guranteed to be topologically sorted.

    :param exprs: The target SymPy expression, or a collection of SymPy expressions.
    :param make: A function to construct symbols used for replacement.
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
    """
    Drop temporaries consisting of single symbols.
    """
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


def cross_cluster_cse(clusters):
    """
    Perform common sub-expressions elimination across an iterable of Clusters.
    """
    clusters = clusters.unfreeze()

    # Detect redundancies
    mapper = {}
    for c in clusters:
        candidates = [i for i in c.trace.values() if i.is_unbound_temporary]
        for v in as_mapper(candidates, lambda i: i.rhs).values():
            for i in v[:-1]:
                mapper[i.lhs.base] = v[-1].lhs.base

    if not mapper:
        # Do not waste time reconstructing identical expressions
        return clusters

    # Apply substitutions
    for c in clusters:
        c.exprs = [i.xreplace(mapper) for i in c.trace.values()
                   if i.lhs.base not in mapper]

    return clusters

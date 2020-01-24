from collections import OrderedDict

from devito.ir import DummyEq
from devito.passes.clusters.utils import dse_pass, makeit_ssa
from devito.symbolics import (count, estimate_cost, q_xop, q_leaf, retrieve_terminals,
                              yreplace)
from devito.tools import DAG, split
from devito.types import Scalar

__all__ = ['cse']


@dse_pass
def cse(cluster, template, *args):
    """
    Common sub-expressions elimination (CSE).
    """
    make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
    processed = _cse(cluster.exprs, make)

    return cluster.rebuild(processed)


def _cse(exprs, make, mode='default'):
    """
    Main common sub-expressions elimination routine.

    Note: the output is guaranteed to be topologically sorted.

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
    # also ensuring some form of post-processing
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
        mapped = [DummyEq(v, k) for k, v in reversed(list(mapper.items()))] + mapped

        # Prepare for the next round
        for k in picked:
            targets.pop(k)
    processed = mapped + processed

    # At this point we may have useless temporaries (e.g., r0=r1). Let's drop them
    processed = _compact_temporaries(processed)

    # Perform topological sorting so that reads-after-writes are honored
    processed = _topological_sort(processed)

    return processed


def _compact_temporaries(exprs):
    """
    Drop temporaries consisting of isolated symbols.
    """
    # First of all, convert to SSA
    exprs = makeit_ssa(exprs)

    # What's gonna be dropped
    mapper = {e.lhs: e.rhs for e in exprs
              if e.lhs.is_Symbol and (q_leaf(e.rhs) or e.rhs.is_Function)}

    processed = []
    for e in exprs:
        if e.lhs not in mapper:
            # The temporary is retained, and substitutions may be applied
            handle, _ = yreplace(e, mapper, repeat=True)
            assert len(handle) == 1
            processed.extend(handle)

    return processed


def _topological_sort(exprs):
    """
    Topologically sort the temporaries in a list of equations.
    """
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

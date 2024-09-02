from devito.ir import Cluster
from devito.tools import as_tuple
from devito.types import CriticalRegion, Eq, Symbol

__all__ = ['is_memcpy', 'make_critical_sequence', 'in_critical_region']


def is_memcpy(expr):
    """
    True if `expr` implements a memcpy involving an Array, False otherwise.
    """
    a, b = expr.args

    if not (a.is_Indexed and b.is_Indexed):
        return False

    return a.function.is_Array or b.function.is_Array


def make_critical_sequence(ispace, sequence, **kwargs):
    sequence = as_tuple(sequence)
    assert len(sequence) >= 1

    processed = []

    # Opening
    expr = Eq(Symbol(name='⋈'), CriticalRegion(True))
    processed.append(Cluster(exprs=expr, ispace=ispace, **kwargs))

    processed.extend(sequence)

    # Closing
    expr = Eq(Symbol(name='⋈'), CriticalRegion(False))
    processed.append(Cluster(exprs=expr, ispace=ispace, **kwargs))

    return processed


def in_critical_region(cluster, clusters):
    """
    Return the opening Cluster of the critical sequence containing `cluster`,
    or None if `cluster` is not part of a critical sequence.
    """
    maybe_found = None
    for c in clusters:
        if c is cluster:
            return maybe_found
        elif c.is_critical_region and maybe_found:
            maybe_found = None
        elif c.is_critical_region:
            maybe_found = c
    return None

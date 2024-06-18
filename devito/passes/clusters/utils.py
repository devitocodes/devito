from devito.ir import Cluster
from devito.symbolics import uxreplace
from devito.tools import as_tuple
from devito.types import CriticalRegion, Eq, Symbol, Wildcard

__all__ = ['makeit_ssa', 'is_memcpy', 'make_critical_sequence',
           'in_critical_region']


def makeit_ssa(exprs):
    """
    Convert an iterable of Eqs into Static Single Assignment (SSA) form.
    """
    # Identify recurring LHSs
    seen = {}
    for i, e in enumerate(exprs):
        if not isinstance(e.lhs, Wildcard):
            seen.setdefault(e.lhs, []).append(i)
    # Optimization: don't waste time reconstructing stuff if already in SSA form
    if all(len(i) == 1 for i in seen.values()):
        return exprs
    # SSA conversion
    c = 0
    mapper = {}
    processed = []
    for i, e in enumerate(exprs):
        where = seen[e.lhs]
        rhs = uxreplace(e.rhs, mapper)
        if len(where) > 1:
            needssa = e.is_Scalar or where[-1] != i
            lhs = Symbol(name='ssa%d' % c, dtype=e.dtype) if needssa else e.lhs
            if e.is_Increment:
                # Turn AugmentedAssignment into Assignment
                processed.append(e.func(lhs, mapper[e.lhs] + rhs, operation=None))
            else:
                processed.append(e.func(lhs, rhs))
            mapper[e.lhs] = lhs
            c += 1
        else:
            processed.append(e.func(e.lhs, rhs))
    return processed


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

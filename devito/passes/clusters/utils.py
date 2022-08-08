from devito.symbolics import uxreplace
from devito.types import Symbol, Wildcard

__all__ = ['makeit_ssa']


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

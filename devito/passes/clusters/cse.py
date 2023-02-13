from collections import Counter, OrderedDict
from functools import singledispatch

from sympy import Add, Function, Indexed, Mul, Pow

from devito.finite_differences.differentiable import IndexDerivative
from devito.ir import Cluster, Scope, cluster_pass
from devito.passes.clusters.utils import makeit_ssa
from devito.symbolics import estimate_cost, q_leaf
from devito.symbolics.manipulation import _uxreplace
from devito.types import Eq, Temp as Temp0

__all__ = ['cse']


class Temp(Temp0):
    pass


@cluster_pass
def cse(cluster, sregistry, options, *args):
    """
    Common sub-expressions elimination (CSE).
    """
    make = lambda: Temp(name=sregistry.make_name(), dtype=cluster.dtype)
    exprs = _cse(cluster, make, min_cost=options['cse-min-cost'])

    return cluster.rebuild(exprs=exprs)


def _cse(maybe_exprs, make, min_cost=1, mode='default'):
    """
    Main common sub-expressions elimination routine.

    Note: the output is guaranteed to be topologically sorted.

    Parameters
    ----------
    maybe_exprs : expr-like or list of expr-like  or Cluster
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

    # Just for flexibility, accept either Clusters or exprs
    if isinstance(maybe_exprs, Cluster):
        processed = list(maybe_exprs.exprs)
        scope = maybe_exprs.scope
    else:
        processed = list(maybe_exprs)
        scope = Scope(maybe_exprs)

    # Some sub-expressions aren't really "common" -- that's the case of Dimension-
    # independent data dependences. For example:
    #
    # ... = ... a[i] + 1 ...
    # a[i] = ...
    # ... = ... a[i] + 1 ...
    #
    # `a[i] + 1` will be excluded, as there's a flow Dimension-independent data
    # dependence involving `a`
    d_flow = {i.source.access for i in scope.d_flow.independent()}
    d_anti = {i.source.access for i in scope.d_anti.independent()}
    exclude = d_flow & d_anti

    while True:
        # Detect redundancies
        counted = count(processed).items()
        targets = OrderedDict([(k, estimate_cost(k, True)) for k, v in counted if v > 1])

        # Rule out Dimension-independent data dependencies
        targets = OrderedDict([(k, v) for k, v in targets.items()
                               if not k.free_symbols & exclude])

        if not targets or max(targets.values()) < min_cost:
            break

        # Create temporaries
        hit = max(targets.values())
        temps = [Eq(make(), k) for k, v in targets.items() if v == hit]

        # Apply replacements
        # The extracted temporaries are inserted before the first expression
        # that contains it
        updated = []
        for e in processed:
            pe = e
            for t in temps:
                pe, changed = _uxreplace(pe, {t.rhs: t.lhs})
                if changed and t not in updated:
                    updated.append(t)
            updated.append(pe)
        processed = updated

        # Update `exclude` for the same reasons as above -- to rule out CSE across
        # Dimension-independent data dependences
        exclude.update({t.lhs for t in temps})

    # At this point we may have useless temporaries (e.g., r0=r1). Let's drop them
    processed = _compact_temporaries(processed, exclude)

    return processed


def _compact_temporaries(exprs, exclude):
    """
    Drop temporaries consisting of isolated symbols.
    """
    # First of all, convert to SSA
    exprs = makeit_ssa(exprs)

    # Drop candidates are all exprs in the form `t0 = s` where `s` is a symbol
    # Note: only CSE-captured Temps, which are by construction local objects, may
    # safely be compacted; a generic Symbol could instead be accessed in a subsequent
    # Cluster, for example: `for (i = ...) { a = b; for (j = a ...) ...`
    mapper = {e.lhs: e.rhs for e in exprs
              if isinstance(e.lhs, Temp) and q_leaf(e.rhs) and e.lhs not in exclude}

    processed = []
    for e in exprs:
        if e.lhs not in mapper:
            # The temporary is retained, and substitutions may be applied
            expr, changed = e, True
            while changed:
                expr, changed = _uxreplace(expr, mapper)
            processed.append(expr)

    return processed


@singledispatch
def count(expr):
    """
    Construct a mapper `expr -> #occurrences` for each sub-expression in `expr`.
    """
    mapper = Counter()
    for a in expr.args:
        mapper.update(count(a))
    return mapper


@count.register(list)
@count.register(tuple)
def _(exprs):
    mapper = Counter()
    for e in exprs:
        mapper.update(count(e))
    return mapper


@count.register(IndexDerivative)
@count.register(Indexed)
def _(expr):
    """
    Handler for objects preventing CSE to propagate through their arguments.
    """
    return Counter()


@count.register(Add)
@count.register(Mul)
@count.register(Pow)
@count.register(Function)
def _(expr):
    mapper = Counter()
    for a in expr.args:
        mapper.update(count(a))

    mapper[expr] += 1

    return mapper

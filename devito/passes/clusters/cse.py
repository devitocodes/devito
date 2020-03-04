from collections import OrderedDict

from devito.ir import DummyEq, Cluster, Scope
from devito.passes.clusters.utils import cluster_pass, makeit_ssa
from devito.symbolics import count, estimate_cost, q_xop, q_leaf, yreplace
from devito.types import Scalar

__all__ = ['cse']


@cluster_pass
def cse(cluster, template, *args):
    """
    Common sub-expressions elimination (CSE).
    """
    make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
    processed = _cse(cluster.exprs, make)

    return cluster.rebuild(processed)


def _cse(maybe_exprs, make, mode='default'):
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
        cluster = maybe_exprs
        processed = list(cluster.exprs)
        scope = cluster.scope
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
    exclude = {i.source.indexed for i in scope.d_flow.independent()}

    mapped = []
    while True:
        # Detect redundancies
        counted = count(mapped + processed, q_xop).items()
        targets = OrderedDict([(k, estimate_cost(k, True)) for k, v in counted if v > 1])

        # Rule out Dimension-independent data dependencies
        targets = OrderedDict([(k, v) for k, v in targets.items()
                               if not k.free_symbols & exclude])

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

        # Update `exclude` for the same reasons as above -- to rule out CSE across
        # Dimension-independent data dependences
        exclude.update({i for i in mapper.values()})

        # Prepare for the next round
        for k in picked:
            targets.pop(k)
    processed = mapped + processed

    # At this point we may have useless temporaries (e.g., r0=r1). Let's drop them
    processed = _compact_temporaries(processed)

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

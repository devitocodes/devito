"""
IET pass that lowers SparseOperation expressions (Interpolation/Injection)
into ElementalFunctions and replaces the bare Expression carried through
the cluster pipeline with a Call.

The SparseEq's LoweredEq carries the time-like Dimensions in its
IterationSpace, so the cluster builder produces the same outer time
loop it would produce for any other equation, and the parent's
argument-bound analysis works against the synthetic indexed lhs/rhs.

The efunc body is built directly via the IR lowering stages
(`_lower_exprs` -> `_lower_clusters` -> `_lower_stree` -> `_lower_uiet`),
without recursing into a full Operator compilation. The resulting
Callable is registered with the parent Graph so the parent's IET
specialisation passes (MPI, OpenMP, linearisation, ...) operate on it
as a sibling Callable.

The efunc carries its own `time_m, time_M` loop. The Call site, sitting
inside the parent's time iteration, collapses that loop to a single
iteration by passing `time_m=time_M=<parent's current time>`. Buffered
TimeFunction accesses (`t = time % bs`) are computed inside the efunc
exactly as the standard pipeline would emit them.
"""

from devito.ir.iet import (
    Call, EntryFunction, Expression, FindNodes, Transformer, make_callable
)
from devito.passes.iet.engine import iet_pass

__all__ = ['lower_sparse_ops']


def lower_sparse_ops(graph, **kwargs):
    """
    Replace each sparse-operation Expression in the IET with a Call to an
    ElementalFunction built from the interpolator-produced grid-level Eq
    list. The efunc is constructed directly via the IR lowering stages,
    without a full recursive Operator compilation.
    """
    _lower_sparse_ops(graph, **kwargs)


@iet_pass
def _lower_sparse_ops(iet, sregistry=None, **kwargs):
    if not isinstance(iet, EntryFunction):
        return iet, {}

    mapper = {}
    efuncs = []

    for expr in FindNodes(Expression).visit(iet):
        if not expr.expr.is_SparseOperation:
            continue
        sparse_op = expr.expr.sparse_op

        # Build the efunc body by running the interpolator-produced eqns
        # through the standard IR lowering stages up to (but not
        # including) IET specialisation
        eqns = sparse_op.operation()
        efunc_body = _build_sparse_iet(eqns, sregistry=sregistry, **kwargs)

        # Wrap in a Callable; the parent Graph picks it up and runs its
        # own _specialize_iet on it (MPI, parallelism, linearisation, ...)
        name = sregistry.make_name(prefix=_efunc_prefix(sparse_op))
        efunc = make_callable(name, efunc_body)
        efuncs.append(efunc)

        # Collapse the efunc's internal time loop to a single iteration
        # by passing the parent's current `time` for both `time_m` and
        # `time_M`. Buffered SteppingDimensions are resolved naturally
        # from the parent's scope.
        time_roots = _time_root_dims(sparse_op)
        bound_to_dim = {}
        for d in time_roots:
            bound_to_dim[f'{d.name}_m'] = d
            bound_to_dim[f'{d.name}_M'] = d
        call_args = [bound_to_dim.get(getattr(p, 'name', None), p)
                     for p in efunc.parameters]
        mapper[expr] = Call(efunc.name, call_args)

    if not mapper:
        return iet, {}

    iet = Transformer(mapper).visit(iet)

    return iet, {'efuncs': efuncs}


def _build_sparse_iet(eqns, **kwargs):
    """
    Lower `eqns` through the IR stages up to `iet_build` (no IET
    specialisation), returning the resulting IET body suitable for
    wrapping in a Callable.
    """
    from devito.operator.registry import operator_selector
    cls = operator_selector(**kwargs)
    expressions = cls._lower_exprs(eqns, **kwargs)
    clusters = cls._lower_clusters(expressions, **kwargs)
    stree = cls._lower_stree(clusters, **kwargs)
    uiet = cls._lower_uiet(stree, **kwargs)
    return uiet.body


def _efunc_prefix(sparse_op):
    """Pick an ElementalFunction name prefix based on the sparse-op kind."""
    sfname = sparse_op.interpolator.sfunction.name
    if sparse_op.kind == 'interpolate':
        return f'interpolate_{sfname}'
    if sparse_op.kind == 'inject':
        return f'inject_{sfname}'
    return 'sparse_op'


def _time_root_dims(sparse_op):
    """
    Root TimeDimensions touched by the sparse op, plus any
    implicit_dims supplied by the user (typically a SteppingDimension
    pinning the operation inside the parent's time loop).
    """
    from devito.symbolics import retrieve_functions
    from devito.tools import as_tuple
    roots = set()
    sfunc = sparse_op.interpolator.sfunction
    for d in getattr(sfunc, 'dimensions', ()):
        if d.is_Time:
            roots.add(d.root if d.is_Derived else d)
    for f in retrieve_functions(sparse_op.expr):
        for d in getattr(f, 'dimensions', ()):
            if d.is_Time:
                roots.add(d.root if d.is_Derived else d)
    for d in as_tuple(sparse_op.implicit_dims):
        roots.add(d.root if d.is_Derived else d)
    return roots

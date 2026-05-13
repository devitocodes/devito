"""
IET pass that lowers SparseOperation expressions (Interpolation/Injection)
into ElementalFunctions and replaces the bare Expression carried through
the cluster pipeline with a Call.

The SparseEq's LoweredEq carries the time-like Dimensions in its
IterationSpace, so the cluster builder produces the same outer time
loop it would produce for any other equation, and the data space sees
the right reads/writes for halo and argument-bound analysis. The
radius/sparse-dim iteration belongs to the efunc, which is generated
by recursively compiling the interpolator-produced equations.

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
    ElementalFunction built via `rcompile` from the interpolator-produced
    grid-level Eq list.
    """
    _lower_sparse_ops(graph, **kwargs)


@iet_pass
def _lower_sparse_ops(iet, rcompile=None, sregistry=None, **kwargs):
    if not isinstance(iet, EntryFunction):
        return iet, {}

    mapper = {}
    efuncs = []
    includes = []

    for expr in FindNodes(Expression).visit(iet):
        if not expr.expr.is_SparseOperation:
            continue
        sparse_op = expr.expr.sparse_op

        # Expand into the grid-level Eq sequence and recursively compile
        eqns = sparse_op.operation()
        irs, byproduct = rcompile(eqns)

        # Wrap the body in an ElementalFunction
        name = sregistry.make_name(prefix=_efunc_prefix(sparse_op))
        body = irs.iet.body.body
        efunc = make_callable(name, body)

        efuncs.extend([i.root for i in byproduct.funcs])
        efuncs.append(efunc)
        includes.extend(byproduct.includes)

        # Build the Call. Where the efunc declares `time_m`/`time_M`
        # bounds for a Dimension that the parent is already iterating,
        # pass the parent's current loop variable for both so the
        # efunc loops exactly once per parent iteration.
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

    return iet, {'efuncs': efuncs, 'includes': includes}


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

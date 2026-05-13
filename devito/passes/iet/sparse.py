"""
IET pass that lowers SparseOperation expressions (Interpolation/Injection)
into ElementalFunctions and replaces the synthetic loop nest emitted by the
expression-lowering pipeline with a single Call.

The synthetic shape carried by a SparseEq through clusterization is
`lhs <- rhs` (see `devito.operations.interpolators.UnevaluatedSparseOperation`);
its purpose is purely to anchor data flow for cluster scheduling. The real
expansion (positions, coefficient temporaries, radius loop, accumulation,
writeback) lives in `interpolator._interpolate` / `_inject` and is generated
here on demand, then handed to `rcompile` to obtain a self-contained
ElementalFunction.
"""

from devito.ir.iet import (
    Call, EntryFunction, Expression, FindNodes, Transformer, make_callable
)
from devito.passes.iet.engine import iet_pass

__all__ = ['lower_sparse_ops']


def lower_sparse_ops(graph, **kwargs):
    """
    Replace each sparse-operation Section in the IET with a Call to an
    ElementalFunction built via `rcompile` from the interpolator-produced
    grid-level Eq list.
    """
    _lower_sparse_ops(graph, **kwargs)


@iet_pass
def _lower_sparse_ops(iet, rcompile=None, sregistry=None, **kwargs):
    if not isinstance(iet, EntryFunction):
        return iet, {}

    # Find every bare Expression that represents a sparse operation. The
    # LoweredEq for a SparseEq has an empty IterationSpace, so the IET
    # carries it as a single Expression with no enclosing Iteration nest.
    mapper = {}
    efuncs = []
    includes = []

    for expr in FindNodes(Expression).visit(iet):
        if not expr.expr.is_SparseOperation:
            continue
        sparse_op = expr.expr.sparse_op

        # Expand into the grid-level Eq sequence
        eqns = sparse_op.operation()

        # Recursively compile into a self-contained IET
        irs, byproduct = rcompile(eqns)

        # Wrap the body in an ElementalFunction
        name = sregistry.make_name(prefix=_efunc_prefix(sparse_op))
        body = irs.iet.body.body
        efunc = make_callable(name, body)

        efuncs.extend([i.root for i in byproduct.funcs])
        efuncs.append(efunc)
        includes.extend(byproduct.includes)

        mapper[expr] = Call(efunc.name, efunc.parameters)

    if not mapper:
        return iet, {}

    iet = Transformer(mapper).visit(iet)

    return iet, {'efuncs': efuncs, 'includes': includes}


def _efunc_prefix(sparse_op):
    """Pick an ElementalFunction name prefix based on the sparse-op kind."""
    cls = type(sparse_op).__name__
    if cls == 'Interpolation':
        return f'interpolate_{sparse_op.interpolator.sfunction.name}'
    if cls == 'Injection':
        return f'inject_{sparse_op.interpolator.sfunction.name}'
    return 'sparse_op'

"""
IET pass that lowers SparseOperation expressions (Interpolation/Injection)
into ElementalFunctions and replaces the bare Expression carried through
the cluster pipeline with a Call.

The SparseEq's LoweredEq carries the time-like Dimensions in its
IterationSpace, so the cluster builder produces the same outer time
loop it would produce for any other equation, and the parent's
argument-bound analysis works against the synthetic indexed lhs/rhs.

The efunc body is built via the IR lowering stages
(`_lower_exprs` -> `_lower_clusters` -> `_lower_stree` -> `_lower_uiet`),
without recursing into a full Operator compilation. The resulting
ElementalFunction is registered with the parent Graph so the parent's
IET specialisation passes (MPI, OpenMP, linearisation, ...) operate on
it as a sibling Callable.

After construction the outermost time Iteration is stripped from the
body: the efunc is called once per parent time step, so iterating
time inside it would duplicate the iteration. The `uindices` carried
by the time Iteration (e.g. the SteppingDimension binding
`t = time % buffer_size`) are converted into scalar Expressions placed
at the top of the body, so buffered TimeFunction accesses keep working.
"""

from devito.ir.equations import DummyEq
from devito.ir.iet import (
    Call, EntryFunction, Expression, FindNodes, Iteration, List, Section, Transformer,
    make_efunc
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

        # Build the efunc body by running the interpolator-produced
        # eqns through the IR lowering stages up to (but not
        # including) IET specialisation. Pass `profiler=None` so the
        # recursive `profiler.analyze` does not register the efunc's
        # Sections alongside the parent's.
        eqns = sparse_op.operation()
        inner_kwargs = {**kwargs, 'sregistry': sregistry, 'profiler': None}
        efunc_body = _build_sparse_iet(eqns, **inner_kwargs)
        # Strip Section wrappers (the profiler shouldn't instrument
        # inside the efunc) and the outermost time Iteration, since
        # the parent's time loop already iterates time and the efunc
        # is called once per time step
        efunc_body, _ = _strip_sections(efunc_body)
        efunc_body = _strip_time_iteration(efunc_body)

        # Wrap in an ElementalFunction so the denormals pass (which
        # skips ElementalFunctions) leaves the body alone
        name = sregistry.make_name(prefix=_efunc_prefix(sparse_op))
        efunc = make_efunc(name, efunc_body)
        efuncs.append(efunc)

        mapper[expr] = Call(efunc.name, list(efunc.parameters))

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


def _strip_sections(body):
    """
    Replace each Section node in `body` with its inner body. Section
    wrappers exist for profiling instrumentation, which we don't want
    inside a per-sparse-point ElementalFunction.
    """
    sections = FindNodes(Section).visit(body)
    if not sections:
        return body, []
    mapper = {s: List(body=s.body) for s in sections}
    return Transformer(mapper, nested=True).visit(body), [s.name for s in sections]


def _strip_time_iteration(body):
    """
    Strip the outermost time Iteration from `body`. The efunc is
    called once per time step from inside the parent's time loop, so
    iterating time again here would produce nested loops and incorrect
    accumulation. The Iteration's `uindices` (e.g. the
    `t = time % buffer_size` SteppingDimension binding) are converted
    into scalar Expressions placed before the original body; the
    Iteration's loop variable (e.g. `time`) becomes a free symbol that
    derive_parameters lifts into the efunc signature.
    """
    iterations = FindNodes(Iteration).visit(body)
    time_iter = next((it for it in iterations if it.dim.is_Time), None)
    if time_iter is None:
        return body
    # Build scalar bindings for each uindex: e.g. `t = time % bs`
    bindings = []
    for u in time_iter.uindices:
        bindings.append(Expression(DummyEq(u, u.symbolic_min)))
    new_body = List(body=tuple(bindings) + tuple(time_iter.nodes))
    return Transformer({time_iter: new_body}, nested=True).visit(body)


def _efunc_prefix(sparse_op):
    """Pick an ElementalFunction name prefix based on the sparse-op kind."""
    sfname = sparse_op.interpolator.sfunction.name
    if sparse_op.kind == 'interpolate':
        return f'interpolate_{sfname}'
    if sparse_op.kind == 'inject':
        return f'inject_{sfname}'
    return 'sparse_op'

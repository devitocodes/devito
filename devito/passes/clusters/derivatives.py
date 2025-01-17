from functools import singledispatch

from sympy import S
import numpy as np

from devito.finite_differences import IndexDerivative
from devito.ir import Backward, Forward, Interval, IterationSpace, Queue
from devito.passes.clusters.misc import fuse
from devito.symbolics import BasicWrapperMixin, reuse_if_untouched, uxreplace
from devito.tools import infer_dtype, timed_pass
from devito.types import Eq, Inc, Indexed, Symbol

__all__ = ['lower_index_derivatives']


@timed_pass()
def lower_index_derivatives(clusters, mode=None, **kwargs):
    clusters, weights, mapper = _lower_index_derivatives(clusters, **kwargs)

    if not weights:
        return clusters

    if mode != 'noop':
        clusters = fuse(clusters, toposort='maximal')

    # At this point we can detect redundancies induced by inner derivatives that
    # previously were just not detectable via e.g. plain CSE. For example, if
    # there were two IndexDerivatives such as `(p.dx + m.dx).dx` and `m.dx.dx`
    # then it's only after `_lower_index_derivatives` that they're detectable!
    # TODO: see https://github.com/devitocodes/devito/issues/2306
    clusters = CDE(mapper).process(clusters)

    return clusters


def _lower_index_derivatives(clusters, **kwargs):
    weights = {}
    processed = []
    mapper = {}

    def dump(exprs, c):
        if exprs:
            processed.append(c.rebuild(exprs=exprs))
            exprs[:] = []

    for c in clusters:
        exprs = []
        for e in c.exprs:
            # Optimization 1: if the LHS is already a Symbol, then surely it's
            # usable as a temporary for one of the IndexDerivatives inside `e`
            if e.lhs.is_Symbol and e.operation is None:
                reusable = {e.lhs}
            else:
                reusable = set()

            expr, v = _core(e, c, c.ispace, weights, reusable, mapper, **kwargs)

            if v:
                dump(exprs, c)
                processed.extend(v)

            if e.lhs is expr.rhs:
                # Optimization 2: `e` is of the form
                # `r = IndexDerivative(...)`
                # Rather than say
                # `r = foo(IndexDerivative(...))`
                # Since `r` is reusable (Optimization 1), we now have `r = r`,
                # which can safely be discarded
                pass
            else:
                exprs.append(expr)

        dump(exprs, c)

    return processed, weights, mapper


@singledispatch
def _core(expr, c, ispace, weights, reusables, mapper, **kwargs):
    """
    Recursively carry out the core of `lower_index_derivatives` based
    on single-dispatch.
    """
    args = []
    processed = []
    for a in expr.args:
        e, clusters = _core(a, c, ispace, weights, reusables, mapper, **kwargs)
        args.append(e)
        processed.extend(clusters)

    expr = reuse_if_untouched(expr, args)

    return expr, processed


@_core.register(Symbol)
@_core.register(Indexed)
@_core.register(BasicWrapperMixin)
def _(expr, c, ispace, weights, reusables, mapper, **kwargs):
    return expr, []


@_core.register(IndexDerivative)
def _(expr, c, ispace, weights, reusables, mapper, **kwargs):
    sregistry = kwargs['sregistry']
    options = kwargs['options']
    subs_user = kwargs['subs']

    try:
        cbk0 = deriv_schedule_registry[options['deriv-schedule']]
        cbk1 = deriv_unroll_registry[options['deriv-unroll']]
    except KeyError:
        raise ValueError("Unknown derivative lowering mode")

    # Lower the IndexDerivative
    init, ideriv = cbk0(expr)

    # Create the concrete Weights array, or reuse an already existing one
    # if possible
    name = sregistry.make_name(prefix='w')
    w0 = ideriv.weights.function
    dtype = infer_dtype([w0.dtype, c.dtype])  # At least np.float32
    k = tuple(w0.weights)
    try:
        w = weights[k]
    except KeyError:
        initvalue = tuple(i.subs(subs_user) for i in k)
        w = weights[k] = w0._rebuild(name=name, dtype=dtype, initvalue=initvalue)

    # Replace the abstract Weights array with the concrete one
    subs = {w0.indexed: w.indexed}
    init = uxreplace(init, subs)
    ideriv = uxreplace(ideriv, subs)

    # The IterationSpace in which the IndexDerivative will be computed
    dims = ideriv.dimensions

    intervals = [Interval(d) for d in dims]
    directions = {d: Backward if d.backward else Forward for d in dims}
    ispace0 = IterationSpace(intervals, directions=directions)

    # Minimize the amount of integer arithmetic to calculate the various index
    # access functions by enforcing start at 0, e.g. `r0[x + i0 + 2] -> r0[x + i0]`
    base = ideriv.base
    for d in dims:
        ispace0 = ispace0.translate(d, -d._min)
        base = base.subs(d, d + d._min)
    ideriv = ideriv._subs(ideriv.base, base)

    # Should the IndexDerivative be unrolled?
    init, expr, ispace0 = cbk1(init, ideriv, ispace0)

    # The full IterationSpace
    extra = (ispace.itdims + ispace0.itdims,)
    ispace1 = IterationSpace.union(ispace, ispace0, relations=extra)

    # The Symbol that will hold the result of the IndexDerivative computation
    # NOTE: created before recurring so that we ultimately get a sound ordering
    try:
        s = reusables.pop()
        assert np.can_cast(s.dtype, dtype)
    except KeyError:
        name = sregistry.make_name(prefix='r')
        s = Symbol(name=name, dtype=dtype)

    # Go inside `expr` and recursively lower any nested IndexDerivatives
    expr, processed = _core(expr, c, ispace1, weights, reusables, mapper, **kwargs)

    # Finally inject the lowered IndexDerivative
    if init is not None:
        expr0 = Eq(s, init)
        processed.insert(0, c.rebuild(exprs=expr0, ispace=ispace))

        expr1 = Inc(s, expr)
        processed.append(c.rebuild(exprs=expr1, ispace=ispace1))
    else:
        expr1 = Eq(s, expr)
        processed.append(c.rebuild(exprs=expr1, ispace=ispace1))

    # Track the lowered IndexDerivative for subsequent optimization by the caller
    mapper.setdefault(expr, []).append(s)

    return s, processed


def _lower_index_derivative_base(ideriv):
    return S.Zero, ideriv


deriv_schedule_registry = {
    'basic': _lower_index_derivative_base,
}


deriv_unroll_registry = {
    False: lambda init, ideriv, ispace: (init, ideriv.expr, ispace)
}


class CDE(Queue):

    """
    Common derivative elimination.
    """

    _q_guards_in_key = True
    _q_syncs_in_key = True

    def __init__(self, mapper):
        super().__init__()

        self.mapper = {k: v for k, v in mapper.items() if len(v) > 1}

    def process(self, clusters):
        return self._process_fdta(clusters, 1, subs0={}, seen=set())

    def callback(self, clusters, prefix, subs0=None, seen=None):
        subs = {}
        processed = []
        for c in clusters:
            if c in seen:
                processed.append(c)
                continue

            exprs = []
            for e in c.exprs:
                k, v = e.args

                if k in subs0:
                    continue

                try:
                    subs0[k] = subs[v]
                    continue
                except KeyError:
                    pass

                if v in self.mapper:
                    subs[v] = k
                    exprs.append(e)
                else:
                    exprs.append(uxreplace(e, {**subs0, **subs}))

            processed.append(c.rebuild(exprs=exprs))

        seen.update(processed)

        return processed

"""
IET pass that lowers sparse-operation expressions (Interpolation/Injection)
into ElementalFunctions and replaces the iteration nest produced by the
cluster pipeline with a Call to the resulting efunc.

By the time this pass runs, the cluster pipeline has scheduled each
sparse op into a regular ``(.., p_sf, rd_x, rd_y, ..)`` iteration nest
with a single inner ``Expression`` carrying the synthetic
``LoweredSparseEq``. The pass:

* finds the outermost ``Iteration`` whose Dimension belongs to the
  sparse op (the SparseFunction's sparse Dimension or a radius
  Dimension);
* rewrites the body of the sparse Dimension's ``Iteration`` so the
  position/coefficient temporaries (``posx``, ``px``, ...) are
  computed once per sparse point, above the radius loops;
* for an interpolation, wraps the inner Expression with the scalar
  accumulator pattern (``acc = 0``; ``acc += weights * rhs`` inside
  the radius loops; ``sf[p_sf] = acc`` afterwards);
* for an injection, leaves the inner Expression as the existing
  weighted ``Inc(field[pos+rd], weights * rhs)``;
* wraps the resulting sub-tree in an ``ElementalFunction`` and
  replaces it in the parent IET with a ``Call``.
"""

from collections import OrderedDict

from devito.ir.equations import DummyEq
from devito.ir.equations.algorithms import lower_exprs
from devito.ir.iet import (
    Call, Conditional, EntryFunction, Expression, ExpressionBundle, FindNodes, HaloSpot,
    Increment, Iteration, List, Transformer, make_callable
)
from devito.passes.iet.engine import iet_pass
from devito.types import Eq, InjectionMixin, InterpolationMixin, Symbol

__all__ = ['lower_sparse_ops']


@iet_pass
def lower_sparse_ops(iet, sregistry=None, **kwargs):
    """
    Replace each sparse-op iteration nest in the IET with a Call to an
    ElementalFunction that materialises the position temporaries and
    the inner accumulator/increment pattern.
    """
    if not isinstance(iet, EntryFunction):
        return iet, {}

    # The "head" of a sparse op in the IET is the unique Expression
    # whose lhs is the SparseFunction (interpolation) or the target
    # field (injection); any auxiliary temporary expressions extracted
    # from the original SparseEq (e.g. by ``factorize``/``cse``) are
    # left where they are inside the radius nest.
    sparse_exprs = [e for e in FindNodes(Expression).visit(iet)
                    if e.expr.is_SparseOperation
                    and type(e.expr).is_head_eq(e.expr,
                                                e.expr.interpolator.sfunction)]
    if not sparse_exprs:
        return iet, {}

    # Group head Expressions by their enclosing outer (sparse-Dimension)
    # Iteration; all such Expressions inside the same outer nest share
    # one efunc.
    groups = OrderedDict()
    for expr in sparse_exprs:
        nest = _find_outer_iteration(iet, expr)
        if nest is None:
            continue
        groups.setdefault(nest, []).append(expr)

    # ``lower_sparse_ops`` runs before ``optimize_halospots``, so the
    # halo-exchange optimiser hasn't yet had a chance to drop the
    # reduction-only halo entries that the IR scheduler put around an
    # injection nest (e.g. an entry for ``u`` at ``loc_indices={time:
    # time+1}`` wrapping ``u[time+1] += ...``). Once the nest becomes a
    # Call those expressions are no longer visible to
    # ``_drop_reduction_halospots``, so we shed those entries here -- and
    # if that empties the HaloSpot, replace it whole so the MPI overlap
    # machinery doesn't wrap our Call with stale dynamic-args plumbing.
    parents = {nest: _enclosing_halospot(iet, nest) for nest in groups}

    mapper = {}
    efuncs = []
    for nest, exprs in groups.items():
        new_nest = _materialise_nest(nest, exprs)

        lse = exprs[0].expr
        prefix = f'{lse.efunc_prefix}_{lse.interpolator.sfunction.name}'
        efunc = make_callable(sregistry.make_name(prefix=prefix), new_nest)
        efuncs.append(efunc)

        call = Call(efunc.name, list(efunc.parameters))

        # Any HaloSpot living inside the nest (e.g. around the radius
        # loops reading a grid Function) becomes invisible once the
        # nest collapses into an opaque Call. Hoist its halo_scheme out
        # so the halo exchange still happens at the right point in the
        # parent IET, sitting next to the Call rather than buried in it.
        # An injection's lhs is *written* (not read) so its inner halo
        # entry for that Function is a reduction-only halo that the
        # caller has nothing to read back -- drop it before hoisting.
        reduced = {e.expr.lhs.function for e in exprs
                   if isinstance(e.expr, InjectionMixin)}
        prelude = _hoisted_halo_prelude(nest, reduced)

        parent = parents[nest]
        if parent is None:
            mapper[nest] = List(body=prelude + (call,)) if prelude else call
            continue

        # Drop fields that the (now-opaque) Call only writes/increments,
        # since the wrapping HaloSpot's purpose was to ensure read-side
        # coherency for them and the read no longer exists at the IET
        # level. Interpolation reads its target field, so its entries
        # stay.
        reduced = {e.expr.lhs.function for e in exprs
                   if isinstance(e.expr, InjectionMixin)}
        hs = parent.halo_scheme.drop(reduced) if reduced else parent.halo_scheme
        if hs.is_void:
            mapper[parent] = call
        elif hs is parent.halo_scheme:
            mapper[nest] = call
        else:
            mapper[parent] = parent._rebuild(halo_scheme=hs, body=call)

    if not mapper:
        return iet, {}

    return Transformer(mapper).visit(iet), {'efuncs': efuncs}


def _find_outer_iteration(iet, expr):
    """
    Walk up the IET from ``expr`` and return the outermost Iteration
    whose ``dim.root`` is the SparseFunction's sparse Dimension. This
    is the entry point of the sparse-op nest in the parent IET; the
    full nest (including any block Iterations the cluster pipeline
    wrapped around the sparse loop) gets extracted into the efunc.
    """
    sparse_dim = expr.expr.interpolator.sfunction._sparse_dim
    for it in FindNodes(Iteration).visit(iet):
        if it.dim.root is sparse_dim and expr in FindNodes(Expression).visit(it):
            return it
    return None


def _enclosing_halospot(iet, nest):
    """
    Return the HaloSpot directly wrapping ``nest``, if any.
    """
    for hs in FindNodes(HaloSpot).visit(iet):
        if nest in FindNodes(Iteration).visit(hs):
            return hs
    return None


def _hoisted_halo_prelude(nest, reduced=None):
    """
    Build a tuple of nodes that recreate, outside the sparse nest, any
    HaloSpots that live inside it. Each hoisted HaloSpot is wrapped in
    the Iterations from ``nest`` whose dim is referenced by the halo's
    ``loc_indices`` (e.g. the ``ps`` loop wrapping a halo whose
    ``loc_indices = {ps}``), so the lowering pipeline that turns the
    HaloSpot into a halo-exchange Call still sees the indices it needs
    in scope.

    ``reduced``, when provided, lists Functions whose halo entries
    should be dropped from the hoisted scheme -- these are the
    Functions the sparse Call only writes/increments (e.g. an
    injection's lhs), so the parent never reads them back and the
    halo update would be redundant.
    """
    reduced = reduced or set()

    inner = []
    for hs in FindNodes(HaloSpot).visit(nest):
        scheme = hs.halo_scheme.drop(reduced) if reduced else hs.halo_scheme
        if not scheme.is_void:
            inner.append(scheme)
    if not inner:
        return ()

    iters = FindNodes(Iteration).visit(nest)
    prelude = []
    for scheme in inner:
        loc_dims = set().union(*(d._defines for d in scheme.loc_indices))
        wrappers = [it for it in iters if it.dim in loc_dims]
        body = HaloSpot(List(body=[]), scheme)
        for it in reversed(wrappers):
            body = it._rebuild(nodes=body)
        prelude.append(body)
    return tuple(prelude)


def _materialise_nest(nest, exprs):
    """
    Rewrite the sparse Dimension's Iteration body to compute the
    position/coefficient temps once per sparse point, then for any
    interpolation Expression wrap it with the scalar accumulator
    pattern. Multiple sparse-op Expressions sharing the same outer
    Iteration are materialised in one pass and reuse the same temps.

    ``nest`` is the *outermost* sparse-Dimension Iteration, so that the
    whole block-Iteration hierarchy (e.g. ``p_rec0_blk0`` -> ``p_rec``
    on the GPU pipeline) is extracted into the efunc and downstream GPU
    kernel synthesis can fold the block loops into a thread-grid
    wrapping the kernel body. The temps and the accumulator pattern,
    however, must live *inside* the innermost sparse Iteration -- one
    set per sparse point, sitting beneath any thread-index/OOB-guard
    prelude that the GPU kernel prep may have inserted.
    """
    # Position + coefficient temporaries as IET Expressions. These are
    # the same for every Expression in the group, so we emit them once.
    # The sample's leaf class (Interpolation/Injection) drives whether
    # the temps carry staggering shifts.
    sample = exprs[0].expr
    temp_exprs = tuple(Expression(DummyEq(e.lhs, e.rhs))
                       for e in lower_exprs(sample.sparse_temps()))

    # Find the innermost sparse-Dimension Iteration within ``nest`` --
    # that's where the head Expressions actually live, beneath any block
    # Iterations that the cluster pipeline wrapped around the sparse
    # loop.
    sparse_dim = sample.interpolator.sfunction._sparse_dim
    inner_iter = nest
    for it in FindNodes(Iteration).visit(nest):
        if it.dim.root is sparse_dim and \
                any(e in FindNodes(Expression).visit(it) for e in exprs):
            inner_iter = it

    # ``inner_iter`` may carry a GPU kernel prelude (thread-index
    # ``ExpressionBundle`` and OOB ``Conditional``) that downstream
    # kernel synthesis expects to find at the top of the block dim's
    # body. The temps and the accumulator pattern go *after* that
    # prelude.
    head, body_nodes = _split_kernel_prelude(inner_iter.nodes)

    radius_nest = body_nodes[0] if len(body_nodes) == 1 else List(body=body_nodes)
    interp_exprs = [e for e in exprs if isinstance(e.expr, InterpolationMixin)]
    inject_exprs = [e for e in exprs if isinstance(e.expr, InjectionMixin)]

    new_body = []
    for expr in interp_exprs:
        siblings = [e for e in exprs if e is not expr]
        new_body.append(_interp_inner_block(
            radius_nest, expr, expr.expr.interpolator, siblings))
    if inject_exprs:
        drop = {e: None for e in interp_exprs}
        new_body.append(Transformer(drop, nested=True).visit(radius_nest)
                        if drop else radius_nest)

    new_inner_iter = inner_iter._rebuild(
        nodes=head + temp_exprs + tuple(new_body)
    )
    if new_inner_iter is inner_iter:
        return nest
    return Transformer({inner_iter: new_inner_iter}, nested=True).visit(nest)


def _split_kernel_prelude(nodes):
    """
    Split the contents of a sparse-Dimension Iteration into the GPU
    kernel prelude (the thread-index ``ExpressionBundle`` and the
    optional OOB ``Conditional``) and the remaining body. On non-cuda
    pipelines the prelude is empty and the full ``nodes`` tuple is the
    body.
    """
    head = ()
    body = tuple(nodes)
    if body and isinstance(body[0], ExpressionBundle):
        head += (body[0],)
        body = body[1:]
        if body and isinstance(body[0], Conditional):
            head += (body[0],)
            body = body[1:]
    return head, body


def _interp_inner_block(inner, expr, interp, siblings):
    """
    Build the accumulator/radius/write-back triple for an interpolation:

        ``Eq(acc, 0)``
        radius_nest with ``Inc(acc, weights * rhs)`` in place of expr
        ``Eq(sf[p], acc)``     # ``Inc`` when the SparseEq is a SparseInc

    ``siblings`` are sparse-op Expression nodes in the same radius nest
    (e.g. a second interpolation or an injection sharing the outer
    sparse Iteration) that must be removed from this copy of the nest
    so the accumulator pattern wraps only ``expr``'s increment.

    Any Iteration between the outer sparse Dimension and the radius
    Dimensions (e.g. an extra non-grid Dimension on the SparseFunction)
    is preserved as an enclosing loop around the accumulator pattern,
    so the write-back ``sf[..., d, p_*]`` sees a fresh accumulator per
    iteration of ``d``.
    """
    eq = expr.expr
    sf_lhs = eq.lhs
    rhs = eq.rhs

    acc = Symbol(name=f'sum{interp.sfunction.name}', dtype=sf_lhs.dtype)

    # ``rhs`` is already indexified; only the weights need lowering
    # (e.g. coefficient Functions are indexed by the radius dim).
    # Pull the concretized radius ConditionalDimensions from ``rhs``
    # so the weights inherit the same (renamed) ``cond`` Thicknesses
    # and don't leak duplicate ``x_ltkn``/``x_rtkn`` parameters into
    # the efunc signature.
    rdims_concrete = {d.name: d for d in rhs.free_symbols
                      if getattr(d, 'is_Conditional', False)}
    weights = interp._weights()
    if rdims_concrete:
        weights = weights.xreplace({
            rd: rdims_concrete[rd.name]
            for rd in weights.free_symbols
            if getattr(rd, 'is_Conditional', False) and rd.name in rdims_concrete
        })
    weighted_rhs = lower_exprs(Eq(acc, weights)).rhs * rhs

    init = Expression(DummyEq(acc, 0))
    inc = Increment(DummyEq(acc, weighted_rhs))
    # Honour the synthetic Eq's flavour: an ``IncrInterpolation`` means
    # the user asked for ``sf[p_*] += sum`` (interpolation with
    # ``increment=True``); a plain ``Interpolation`` is just ``sf[p_*] =
    # sum``. We key off the leaf class' ``is_increment_writeback`` flag
    # rather than ``is_Reduction`` because both flavours are tagged as
    # reductions (``OpInc``) for dependence-analysis purposes -- the rhs
    # is implicitly summed over the radius dims -- but only the
    # ``IncrInterpolation`` flavour writes back with ``+=``.
    write_back_cls = Increment if eq.is_increment_writeback else Expression
    write_back = write_back_cls(DummyEq(sf_lhs, acc))

    # Single substitution: drop siblings, replace ``expr`` with ``inc``.
    mapper = {expr: inc}
    mapper.update({s: None for s in siblings})

    radius_root = _find_radius_root(inner, interp.sfunction)
    if radius_root is None or radius_root is inner:
        return List(body=(init,
                          Transformer(mapper, nested=True).visit(inner),
                          write_back))

    # Wrap the accumulator pattern around just the radius sub-tree,
    # leaving the enclosing non-radius Iterations in place.
    wrapped = List(body=(init, Transformer(mapper, nested=True).visit(radius_root),
                         write_back))
    return Transformer({radius_root: wrapped}, nested=True).visit(inner)


def _find_radius_root(inner, sfunction):
    """
    Return the outermost Iteration in ``inner`` over a radius
    CustomDimension of ``sfunction``, or None if no such Iteration is
    found. Radius Dimensions are named ``r<sparse_dim_name><dim_name>``
    (see ``AbstractSparseFunction._crdim``).
    """
    prefix = f'r{sfunction._sparse_dim.name}'
    for it in FindNodes(Iteration).visit(inner):
        if it.dim.name.startswith(prefix):
            return it
    return None

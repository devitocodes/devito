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
    Call, EntryFunction, Expression, FindNodes, HaloSpot, Increment, Iteration, List,
    Transformer, make_callable
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
        parent = parents[nest]
        if parent is None:
            mapper[nest] = call
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
    whose ``dim.root`` is the SparseFunction's sparse Dimension.
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


def _materialise_nest(nest, exprs):
    """
    Rewrite the sparse Dimension's Iteration body to compute the
    position/coefficient temps once per sparse point, then for any
    interpolation Expression wrap it with the scalar accumulator
    pattern. Multiple sparse-op Expressions sharing the same outer
    Iteration are materialised in one pass and reuse the same temps.
    """
    # Position + coefficient temporaries as IET Expressions. These are
    # the same for every Expression in the group, so we emit them once.
    # The sample's leaf class (Interpolation/Injection) drives whether
    # the temps carry staggering shifts.
    sample = exprs[0].expr
    temp_exprs = tuple(Expression(DummyEq(e.lhs, e.rhs))
                       for e in lower_exprs(sample.sparse_temps()))

    # The radius nest is what runs once per sparse point. For each
    # interpolation Expression in the group, build its
    # accumulator-wrapped copy of the radius nest. Injection Exprs
    # share a single copy of the radius nest (their ``Inc`` already
    # carries the right ``weights * rhs`` form).
    inner = nest.nodes[0] if len(nest.nodes) == 1 else List(body=nest.nodes)
    interp_exprs = [e for e in exprs if isinstance(e.expr, InterpolationMixin)]
    inject_exprs = [e for e in exprs if isinstance(e.expr, InjectionMixin)]

    body = []
    for expr in interp_exprs:
        siblings = [e for e in exprs if e is not expr]
        body.append(_interp_inner_block(inner, expr, expr.expr.interpolator, siblings))
    if inject_exprs:
        drop = {e: None for e in interp_exprs}
        body.append(Transformer(drop, nested=True).visit(inner) if drop else inner)

    return nest._rebuild(nodes=temp_exprs + tuple(body))


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
    # Honour the synthetic Eq's flavour: a SparseInc means the user
    # asked for ``sf[p_*] += sum`` (interpolation with ``increment=True``);
    # otherwise the standard write is ``sf[p_*] = sum``.
    write_back_cls = Increment if eq.is_Reduction else Expression
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

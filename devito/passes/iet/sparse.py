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
from devito.types import Eq, Symbol

__all__ = ['lower_sparse_ops']


def lower_sparse_ops(graph, **kwargs):
    """
    Replace each sparse-op iteration nest in the IET with a Call to an
    ElementalFunction that materialises the position temporaries and
    the inner accumulator/increment pattern.
    """
    _lower_sparse_ops(graph, **kwargs)


@iet_pass
def _lower_sparse_ops(iet, sregistry=None, **kwargs):
    if not isinstance(iet, EntryFunction):
        return iet, {}

    # The "head" of a sparse op in the IET is the unique Expression
    # whose lhs is the SparseFunction (interpolation) or the target
    # field (injection); any auxiliary temporary expressions extracted
    # from the original SparseEq (e.g. by ``factorize``/``cse``) are
    # left where they are inside the radius nest.
    sparse_exprs = [e for e in FindNodes(Expression).visit(iet)
                    if e.expr.is_SparseOperation and _is_head(e.expr)]
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

    # If a sparse-op nest sits inside a HaloSpot whose halo scheme is
    # void (e.g. the reduction-only halo got dropped by
    # ``_drop_reduction_halospots``), replace the HaloSpot rather than
    # just the nest so we don't leave behind an empty HaloSpot — the
    # MPI overlap machinery would otherwise try to wrap our Call with
    # its own dynamic-args plumbing.
    parents = {nest: _enclosing_void_halospot(iet, nest) for nest in groups}

    mapper = {}
    efuncs = []

    for nest, exprs in groups.items():
        new_nest = _materialise_nest(nest, exprs)

        name = sregistry.make_name(prefix=_efunc_prefix(exprs[0].expr))
        efunc = make_callable(name, new_nest)
        efuncs.append(efunc)

        call = Call(efunc.name, list(efunc.parameters))
        target = parents[nest] or nest
        mapper[target] = call

    if not mapper:
        return iet, {}

    iet = Transformer(mapper).visit(iet)

    return iet, {'efuncs': efuncs}


def _enclosing_void_halospot(iet, nest):
    """
    Return the HaloSpot directly wrapping ``nest`` if it carries an
    empty (void) HaloScheme, otherwise None. Such HaloSpots are leftover
    after ``_drop_reduction_halospots`` cleared all entries.
    """
    for hs in FindNodes(HaloSpot).visit(iet):
        if not hs.is_void:
            continue
        if nest in FindNodes(Iteration).visit(hs):
            return hs
    return None


def _is_head(eq):
    """
    True if ``eq`` is the "head" of its sparse op: the Expression
    whose lhs is the SparseFunction (interpolation) or a
    DiscreteFunction grid field (injection), as opposed to an
    auxiliary scalar temporary extracted from the original SparseEq by
    a cluster pass.
    """
    sf = eq.interpolator.sfunction
    f = eq.lhs.function
    if eq.kind == 'interpolate':
        return f is sf
    # 'inject': head writes into a DiscreteFunction (the grid field),
    # not into a scalar temporary
    return f.is_DiscreteFunction and f is not sf


def _find_outer_iteration(iet, expr):
    """
    Walk up the IET from ``expr`` and return the outermost Iteration
    whose ``dim.root`` is the SparseFunction's sparse Dimension.
    """
    sparse_dim = expr.expr.interpolator.sfunction._sparse_dim
    for it in FindNodes(Iteration).visit(iet):
        if it.dim.root is not sparse_dim:
            continue
        if expr in FindNodes(Expression).visit(it):
            return it
    return None


def _materialise_nest(nest, exprs):
    """
    Rewrite the sparse Dimension's Iteration body to compute the
    position/coefficient temps once per sparse point, then for any
    interpolation Expression wrap it with the scalar accumulator
    pattern. Multiple sparse-op Expressions sharing the same outer
    Iteration are materialised in one pass and reuse the same temps.
    """
    interp = exprs[0].expr.interpolator
    sample_lse = exprs[0].expr

    # Position + coefficient temporaries as IET Expressions. These are
    # the same for every Expression in the group, so we emit them once.
    temps = interp._sparse_temps(
        sample_lse.kind, _user_expr(sample_lse),
        field=_user_field(sample_lse),
        implicit_dims=sample_lse.implicit_dims,
    )
    temp_exprs = tuple(Expression(DummyEq(e.lhs, e.rhs))
                       for e in lower_exprs(temps))

    # For each interpolation Expression in the group, build its
    # accumulator-wrapped radius nest. Injection Exprs are left where
    # they are in the radius nest (their Inc is already the right
    # form); injection Exprs share a single copy of the radius nest.
    inner = _drop_outer(nest)
    interp_exprs = [e for e in exprs if e.expr.kind == 'interpolate']
    inject_exprs = [e for e in exprs if e.expr.kind == 'inject']

    body = []
    for expr in interp_exprs:
        # Build the per-interpolation accumulator: substitute siblings
        # out and replace ``expr`` with the increment in a single
        # Transformer pass so the radius sub-tree contains only the
        # head's increment.
        body.append(_interp_inner_block(inner, expr, expr.expr.interpolator,
                                        siblings=[e for e in exprs if e is not expr]))
    if inject_exprs:
        # Injections share one radius nest with no interpolation heads.
        others = {e: None for e in interp_exprs}
        local_inner = Transformer(others, nested=True).visit(inner) if others else inner
        body.append(local_inner)

    return nest._rebuild(nodes=temp_exprs + tuple(body))


def _interp_inner_block(inner, expr, interp, siblings=()):
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
    weights_expr = lower_exprs(_make_eq(acc, weights)).rhs
    weighted_rhs = weights_expr * rhs

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
        # No intermediate Iteration: wrap the whole ``inner`` directly.
        return List(body=(init,
                          Transformer(mapper, nested=True).visit(inner),
                          write_back))

    # Wrap the accumulator pattern around just the radius sub-tree,
    # leaving the enclosing non-radius Iterations in place.
    new_radius = Transformer(mapper, nested=True).visit(radius_root)
    wrapped = List(body=(init, new_radius, write_back))
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


def _drop_outer(nest):
    """
    Return the sub-IET inside ``nest`` (the Iteration over the sparse
    Dim) — i.e. the radius nest. ``nest.nodes`` is what runs once per
    sparse point.
    """
    if len(nest.nodes) == 1:
        return nest.nodes[0]
    return List(body=nest.nodes)


def _make_eq(lhs, rhs):
    """Helper to wrap a (lhs, rhs) pair as a symbolic Eq for ``lower_exprs``."""
    return Eq(lhs, rhs)


def _efunc_prefix(lse):
    """Pick an ElementalFunction name prefix based on the sparse-op kind."""
    return f'{lse.kind}_{lse.interpolator.sfunction.name}'


def _user_expr(lse):
    """The user-side expression to feed ``_sparse_temps`` (rhs of the LSE)."""
    return lse.rhs


def _user_field(lse):
    """For injection, the destination Function appearing in lhs."""
    if lse.kind == 'inject':
        return lse.lhs.function
    return None

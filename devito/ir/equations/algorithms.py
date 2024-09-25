from collections.abc import Iterable
from collections import defaultdict
from functools import singledispatch

from devito.symbolics import retrieve_indexed, uxreplace, retrieve_dimensions
from devito.tools import Ordering, as_tuple, flatten, filter_sorted, filter_ordered
from devito.types import (Dimension, Eq, IgnoreDimSort, SubDimension,
                          ConditionalDimension)
from devito.types.array import Array
from devito.types.basic import AbstractFunction
from devito.types.grid import MultiSubDimension

__all__ = ['dimension_sort', 'lower_exprs', 'concretize_subdims']


def dimension_sort(expr):
    """
    Topologically sort the Dimensions in ``expr``, based on the order in which they
    appear within Indexeds.
    """

    def handle_indexed(indexed):
        relation = []
        for i in indexed.indices:
            try:
                # Assume it's an AffineIndexAccessFunction...
                relation.append(i.d)
            except AttributeError:
                # It's not! Maybe there are some nested Indexeds (e.g., the
                # situation is A[B[i]])
                nested = flatten(handle_indexed(n) for n in retrieve_indexed(i))
                if nested:
                    relation.extend(nested)
                    continue

                # Fallback: Just insert all the Dimensions we find, regardless of
                # what the user is attempting to do
                relation.extend(filter_sorted(i.atoms(Dimension)))

        # StencilDimensions are lowered subsequently through special compiler
        # passes, so they can be ignored here
        relation = tuple(d for d in relation if not d.is_Stencil)

        return relation

    if isinstance(expr.implicit_dims, IgnoreDimSort):
        relations = set()
    else:
        relations = {handle_indexed(i) for i in retrieve_indexed(expr)}

    # Add in any implicit dimension (typical of scalar temporaries, or Step)
    relations.add(expr.implicit_dims)

    # Add in leftover free dimensions (not an Indexed' index)
    extra = set(retrieve_dimensions(expr, deep=True))

    # Add in pure data dimensions (e.g., those accessed only via explicit values,
    # such as A[3])
    indexeds = retrieve_indexed(expr, deep=True)
    for i in indexeds:
        extra.update({d for d in i.function.dimensions if i.indices[d].is_integer})

    # Enforce determinism
    extra = filter_sorted(extra)

    # Add in implicit relations for parent dimensions
    # -----------------------------------------------
    # 1) Note that (d.parent, d) is what we want, while (d, d.parent) would be
    # wrong; for example, in `((t, time), (t, x, y), (x, y))`, `x` could now
    # preceed `time`, while `t`, and therefore `time`, *must* appear before `x`,
    # as indicated by the second relation
    implicit_relations = {(d.parent, d) for d in extra if d.is_Derived and not d.indirect}

    # 2) To handle cases such as `((time, xi), (x,))`, where `xi` a SubDimension
    # of `x`, besides `(x, xi)`, we also have to add `(time, x)` so that we
    # obtain the desired ordering `(time, x, xi)`. W/o `(time, x)`, the ordering
    # `(x, time, xi)` might be returned instead, which would be non-sense
    for i in relations:
        dims = []
        for d in i:
            # Only add index if a different Dimension name to avoid dropping conditionals
            # with the same name as the parent
            if d.index.name == d.name:
                dims.append(d)
            else:
                dims.extend([d.index, d])

        implicit_relations.update({tuple(filter_ordered(dims))})

    ordering = Ordering(extra, relations=implicit_relations, mode='partial')

    return ordering


def lower_exprs(expressions, subs=None, **kwargs):
    """
    Lowering an expression consists of the following passes:

        * Indexify functions;
        * Align Indexeds with the computational domain;
        * Apply user-provided substitution;

    Examples
    --------
    f(x - 2*h_x, y) -> f[xi + 2, yi + 4]  (assuming halo_size=4)
    """
    return _lower_exprs(expressions, subs or {})


def _lower_exprs(expressions, subs):
    processed = []
    for expr in as_tuple(expressions):
        try:
            dimension_map = expr.subdomain.dimension_map
        except AttributeError:
            # Some Relationals may be pure SymPy objects, thus lacking the subdomain
            dimension_map = {}

        # Handle Functions (typical case)
        mapper = {f: _lower_exprs(f.indexify(subs=dimension_map), subs)
                  for f in expr.find(AbstractFunction)}

        # Handle Indexeds (from index notation)
        for i in retrieve_indexed(expr):
            f = i.function

            # Introduce shifting to align with the computational domain
            indices = [_lower_exprs(a, subs) + o for a, o in
                       zip(i.indices, f._size_nodomain.left)]

            # Substitute spacing (spacing only used in own dimension)
            indices = [i.xreplace({d.spacing: 1, -d.spacing: -1})
                       for i, d in zip(indices, f.dimensions)]

            # Apply substitutions, if necessary
            if dimension_map:
                indices = [j.xreplace(dimension_map) for j in indices]

            # Handle Array
            if isinstance(f, Array) and f.initvalue is not None:
                initvalue = [_lower_exprs(i, subs) for i in f.initvalue]
                # TODO: fix rebuild to avoid new name
                f = f._rebuild(name='%si' % f.name, initvalue=initvalue)

            mapper[i] = f.indexed[indices]
        # Add dimensions map to the mapper in case dimensions are used
        # as an expression, i.e. Eq(u, x, subdomain=xleft)
        mapper.update(dimension_map)
        # Add the user-supplied substitutions
        mapper.update(subs)
        # Apply mapper to expression
        processed.append(uxreplace(expr, mapper))

    if isinstance(expressions, Iterable):
        return processed
    else:
        assert len(processed) == 1
        return processed.pop()


def concretize_subdims(exprs, **kwargs):
    """
    Given a list of expressions, return a new list where all user-defined
    SubDimensions have been replaced by their concrete counterparts.

    A concrete SubDimension binds objects that are guaranteed to be unique
    across `exprs`, such as the thickness symbols.
    """
    sregistry = kwargs.get('sregistry')

    # {root Dimension -> {SubDimension -> concrete SubDimension}}
    mapper = defaultdict(dict)

    _concretize_subdims(exprs, mapper, sregistry)
    if not mapper:
        return exprs

    subs = {}
    for i in mapper.values():
        subs.update(i)

    processed = [uxreplace(e, subs) for e in exprs]

    return processed


@singledispatch
def _concretize_subdims(a, mapper, sregistry):
    pass


@_concretize_subdims.register(list)
@_concretize_subdims.register(tuple)
def _(v, mapper, sregistry):
    for i in v:
        _concretize_subdims(i, mapper, sregistry)


@_concretize_subdims.register(Eq)
def _(expr, mapper, sregistry):
    for d in expr.free_symbols:
        _concretize_subdims(d, mapper, sregistry)


@_concretize_subdims.register(SubDimension)
def _(d, mapper, sregistry):
    # TODO: to be implemented as soon as we drop the counter machinery in
    # Grid.__subdomain_finalize__
    pass


@_concretize_subdims.register(ConditionalDimension)
def _(d, mapper, sregistry):
    # TODO: to be implemented as soon as we drop the counter machinery in
    # Grid.__subdomain_finalize__
    # TODO: call `_concretize_subdims(d.parent, mapper)` as the parent might be
    # a SubDimension!
    pass


@_concretize_subdims.register(MultiSubDimension)
def _(d, mapper, sregistry):
    if not d.is_abstract:
        # TODO: for now Grid.__subdomain_finalize__ creates the thickness, but
        # soon it will be done here instead
        return

    pd = d.parent

    subs = mapper[pd]

    if d in subs:
        # Already have a substitution for this dimension
        return

    name = sregistry.make_name(prefix=d.name)
    ltkn, rtkn = MultiSubDimension._symbolic_thickness(name)

    thickness = (ltkn, rtkn)

    subs[d] = d._rebuild(d.name, pd, thickness)

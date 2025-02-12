from collections.abc import Iterable
from functools import singledispatch

from devito.symbolics import (retrieve_indexed, uxreplace, retrieve_dimensions,
                              retrieve_functions)
from devito.tools import (Ordering, as_tuple, flatten, filter_sorted, filter_ordered,
                          frozendict)
from devito.types import (Dimension, Eq, IgnoreDimSort, SubDimension,
                          ConditionalDimension)
from devito.types.array import Array
from devito.types.basic import AbstractFunction
from devito.types.dimension import MultiSubDimension, Thickness
from devito.data.allocators import DataReference
from devito.logger import warning

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
        dimension_map = _make_dimension_map(expr)

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


def _make_dimension_map(expr):
    """
    Make the dimension_map for an expression. In the basic case, this is extracted
    directly from the SubDomain attached to the expression.

    The indices of a Function defined on a SubDomain will all be the SubDimensions of
    that SubDomain. In this case, the dimension_map should be extended with
    `{ix_f: ix_i, iy_f: iy_i}` where `ix_f` is the SubDimension on which the Function is
    defined, and `ix_i` is the SubDimension to be iterated over.
    """
    try:
        dimension_map = {**expr.subdomain.dimension_map}
    except AttributeError:
        # Some Relationals may be pure SymPy objects, thus lacking the SubDomain
        dimension_map = {}
    else:
        functions = [f for f in retrieve_functions(expr) if f._is_on_subdomain]
        for f in functions:
            dimension_map.update({d: expr.subdomain.dimension_map[d.root]
                                  for d in f.space_dimensions if d.is_Sub})

    return frozendict(dimension_map)


def concretize_subdims(exprs, **kwargs):
    """
    Given a list of expressions, return a new list where all user-defined
    SubDimensions have been replaced by their concrete counterparts.

    A concrete SubDimension binds objects that are guaranteed to be unique
    across `exprs`, such as the thickness symbols.
    """
    sregistry = kwargs.get('sregistry')

    mapper = {}
    rebuilt = {}  # Rebuilt implicit dims etc which are shared between dimensions

    _concretize_subdims(exprs, mapper, rebuilt, sregistry)
    if not mapper:
        return exprs

    # There may be indexed Arrays defined on SubDimensions in the expressions
    # These must have their dimensions replaced and their .function attribute
    # reset to prevent recovery of the original SubDimensions
    functions = {f for f in retrieve_functions(exprs) if f.is_Array}
    for f in functions:
        dimensions = tuple(mapper.get(d, d) for d in f.dimensions)
        if dimensions != f.dimensions:
            # A dimension has been rebuilt, so build a mapper for Indexed
            mapper[f.indexed] = f._rebuild(dimensions=dimensions).indexed

    processed = [uxreplace(e, mapper) for e in exprs]

    return processed


@singledispatch
def _concretize_subdims(a, mapper, rebuilt, sregistry):
    pass


@_concretize_subdims.register(list)
@_concretize_subdims.register(tuple)
def _(v, mapper, rebuilt, sregistry):
    for i in v:
        _concretize_subdims(i, mapper, rebuilt, sregistry)


@_concretize_subdims.register(Eq)
def _(expr, mapper, rebuilt, sregistry):
    # Split and reorder symbols so SubDimensions are processed before lone Thicknesses
    # This means that if a Thickness appears both in the expression and attached to
    # a SubDimension, it gets concretised with the SubDimension.
    thicknesses = {i for i in expr.free_symbols if isinstance(i, Thickness)}
    symbols = expr.free_symbols.difference(thicknesses)

    # Iterate over all other symbols before iterating over standalone thicknesses
    for d in tuple(symbols) + tuple(thicknesses):
        _concretize_subdims(d, mapper, rebuilt, sregistry)

    # Subdimensions can be hiding in implicit dims
    _concretize_subdims(expr.implicit_dims, mapper, rebuilt, sregistry)


@_concretize_subdims.register(Thickness)
def _(tkn, mapper, rebuilt, sregistry):
    if tkn in mapper:
        # Already have a substitution for this thickness
        return

    mapper[tkn] = tkn._rebuild(name=sregistry.make_name(prefix=tkn.name))


@_concretize_subdims.register(SubDimension)
def _(d, mapper, rebuilt, sregistry):
    if d in mapper:
        # Already have a substitution for this dimension
        return

    _concretize_subdims(d.tkns, mapper, rebuilt, sregistry)
    mapper[d] = d._rebuild(thickness=tuple(mapper[tkn] for tkn in d.tkns))


@_concretize_subdims.register(ConditionalDimension)
def _(d, mapper, rebuilt, sregistry):
    if d in mapper:
        # Already have a substitution for this dimension
        return

    _concretize_subdims(d.parent, mapper, rebuilt, sregistry)

    kwargs = {}

    # Parent may be a subdimension
    if d.parent in mapper:
        kwargs['parent'] = mapper[d.parent]

    # Condition may contain subdimensions
    if d.condition is not None:
        for v in d.condition.free_symbols:
            _concretize_subdims(v, mapper, rebuilt, sregistry)

        if any(v in mapper for v in d.condition.free_symbols):
            # Substitute into condition
            kwargs['condition'] = d.condition.xreplace(mapper)

    if kwargs:
        # Rebuild if parent or condition need replacing
        mapper[d] = d._rebuild(**kwargs)


@_concretize_subdims.register(MultiSubDimension)
def _(d, mapper, rebuilt, sregistry):
    if d in mapper:
        # Already have a substitution for this dimension
        return

    tkns = tuple(tkn._rebuild(name=sregistry.make_name(prefix=tkn.name))
                 for tkn in d.thickness)
    kwargs = {'thickness': tkns}
    fkwargs = {}

    idim0 = d.implicit_dimension
    if idim0 is not None:
        if idim0 in rebuilt:
            idim1 = rebuilt[idim0]
        else:
            iname = sregistry.make_name(prefix='n')
            rebuilt[idim0] = idim1 = idim0._rebuild(name=iname)

        kwargs['implicit_dimension'] = idim1
        fkwargs['dimensions'] = (idim1,) + d.functions.dimensions[1:]

    if d.functions in rebuilt:
        functions = rebuilt[d.functions]
    else:
        # Increment every instance of this name after the first encountered
        fname = sregistry.make_name(prefix=d.functions.name, increment_first=False)
        # Warn the user if name has been changed, since this will affect overrides
        if fname != d.functions.name:
            fkwargs['name'] = fname
            warning("%s <%s> renamed as '%s'. Consider assigning a unique name to %s." %
                    (str(d.functions), id(d.functions), fname, d.functions.name))

        fkwargs.update({'function': None,
                        'halo': None,
                        'padding': None})

        # Data in MultiSubDimension function may not have been touched at this point,
        # in which case do not use an allocator, as there is nothing to allocate, and
        # doing so will petrify the `_data` attribute as None.
        if d.functions._data is not None:
            fkwargs.update({'allocator': DataReference(d.functions._data)})

        rebuilt[d.functions] = functions = d.functions._rebuild(**fkwargs)

    kwargs['functions'] = functions

    mapper[d] = d._rebuild(**kwargs)

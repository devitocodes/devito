from collections.abc import Iterable
from functools import singledispatch

from devito.symbolics import retrieve_indexed, uxreplace, retrieve_dimensions
from devito.tools import Ordering, as_tuple, flatten, filter_sorted, filter_ordered
from devito.types import Dimension, IgnoreDimSort
from devito.types.basic import AbstractFunction
from devito.types.dimension import Thickness, SubDimension
from devito.types.grid import MultiSubDimension

__all__ = ['dimension_sort', 'lower_exprs']


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
    return _lower_exprs(expressions, subs or {}, **kwargs)


def _lower_exprs(expressions, subs, **kwargs):
    # FIXME: Not sure why you can get this, but not access with index
    sregistry = kwargs.get('sregistry')

    processed = []
    rebuilt = {}
    for expr in as_tuple(expressions):
        try:
            dimension_map = expr.subdomain.dimension_map
        except AttributeError:
            # Some Relationals may be pure SymPy objects, thus lacking the subdomain
            dimension_map = {}

        if sregistry and dimension_map:
            # Give SubDimension thicknesses, SubDimensionSet functions and
            # SubDomainSet implicit dimensions unique names
            rename_thicknesses(dimension_map, sregistry, rebuilt)
            # Rebuild ConditionalDimensions using rebuilt subdimensions
            # The expression is then rebuilt with this ConditionalDimension
            expr = rebuild_cdims(expr, rebuilt)

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


@singledispatch
def _rename_thicknesses(dim, sregistry, rebuilt):
    # FIXME: Could return dim here, and remove the if statement
    pass


@_rename_thicknesses.register(SubDimension)
def _(dim, sregistry, rebuilt):
    ((lst, lst_v), (rst, rst_v)) = dim.thickness
    lst_name = sregistry.make_name(prefix=lst.name)
    rst_name = sregistry.make_name(prefix=rst.name)
    lst_new = lst._rebuild(name=lst_name)
    rst_new = rst._rebuild(name=rst_name)
    new_thickness = Thickness((lst_new, lst_v),
                              (rst_new, rst_v))

    interval = dim._interval.subs({lst: lst_new, rst: rst_new})
    return dim._rebuild(symbolic_min=interval.left,
                        symbolic_max=interval.right,
                        thickness=new_thickness)


@_rename_thicknesses.register(MultiSubDimension)
def _(dim, sregistry, rebuilt):
    try:
        # Get pre-existing rebuilt implicit dimension from mapper
        # Implicit dimension substitution means there is a function
        # substitution
        idim = rebuilt[dim.implicit_dimension]
        func = rebuilt[dim.functions]

    except KeyError:
        idim_name = sregistry.make_name(prefix=dim.implicit_dimension.name)
        idim = dim.implicit_dimension._rebuild(name=idim_name)
        rebuilt[dim.implicit_dimension] = idim

        dimensions = list(dim.functions.dimensions)
        dimensions[0] = idim

        f_name = sregistry.make_name(prefix=dim.functions.name)
        func = dim.functions._rebuild(name=f_name, dimensions=tuple(dimensions),
                                      halo=None, padding=None)
        # TODO: _rebuild nukes the Function data. It should not do this.
        func.data[:] = dim.functions.data[:]
        rebuilt[dim.functions] = func

    return dim._rebuild(functions=func, implicit_dimension=idim)


def rename_thicknesses(mapper, sregistry, rebuilt):
    """
    Rebuild SubDimensions in a mapper so that their thicknesses
    have unique names.

    Also rebuilds MultiSubDimensions such that their thicknesses
    and implicit dimension names are unique.
    """
    for k, v in mapper.items():
        if v.is_AbstractSub:
            try:
                # Use an existing renaming if one exists
                mapper[k] = rebuilt[v]
            except KeyError:
                mapper[k] = rebuilt[v] = _rename_thicknesses(v, sregistry, rebuilt)


def rebuild_cdims(expr, rebuilt):
    """
    Rebuild expression using ConditionalDimensions where their parent
    dimension is a MultiSubDimension which has been rebuilt to have a unique
    name.
    """
    i_dims = expr.implicit_dims
    rebuilt_dims = []
    for d in i_dims:
        try:
            parent = rebuilt[d.parent]
            # FIXME: Condition substitution should maybe be moved
            cond = d.condition.subs(rebuilt)
            rebuilt_dims.append(d._rebuild(parent=parent, condition=cond,
                                           factor=None))
        except KeyError:
            rebuilt_dims.append(d)
    return expr._rebuild(implicit_dims=tuple(rebuilt_dims))

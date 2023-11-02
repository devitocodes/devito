from collections.abc import Iterable
from itertools import groupby

from sympy import sympify

from devito.symbolics import retrieve_indexed, uxreplace, retrieve_dimensions
from devito.tools import Ordering, as_tuple, flatten, filter_sorted, filter_ordered
from devito.types import Dimension, IgnoreDimSort
from devito.types.basic import AbstractFunction
from devito.ir.support import pull_dims

__all__ = ['dimension_sort', 'lower_exprs', 'separate_dimensions']


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


def lower_exprs(expressions, **kwargs):
    """
    Lowering an expression consists of the following passes:

        * Indexify functions;
        * Align Indexeds with the computational domain;
        * Apply user-provided substitution;

    Examples
    --------
    f(x - 2*h_x, y) -> f[xi + 2, yi + 4]  (assuming halo_size=4)
    """
    # Normalize subs
    subs = {k: sympify(v) for k, v in kwargs.get('subs', {}).items()}

    processed = []
    for expr in as_tuple(expressions):
        try:
            dimension_map = expr.subdomain.dimension_map
        except AttributeError:
            # Some Relationals may be pure SymPy objects, thus lacking the subdomain
            dimension_map = {}

        # Handle Functions (typical case)
        mapper = {f: lower_exprs(f.indexify(subs=dimension_map), **kwargs)
                  for f in expr.find(AbstractFunction)}

        # Handle Indexeds (from index notation)
        for i in retrieve_indexed(expr):
            f = i.function

            # Introduce shifting to align with the computational domain
            indices = [(lower_exprs(a) + o) for a, o in
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


def separate_dimensions(expressions):
    """
    Rename Dimensions with clashing names within expressions.
    """
    resolutions = {}
    count = {}  # Keep track of increments on dim names
    processed = []
    from devito.symbolics import retrieve_indexed
    for e in expressions:
        # Extraction of dimensions is too effective
        # Group dimensions by matching names
        dims = sorted(pull_dims(e, flag=False), key=lambda x: x.name)
        # dims = sorted(set().union(*tuple(set(i.function.dimensions)
        #                                  for i in retrieve_indexed(e))),
        #               key=lambda x: x.name)

        # print("Dims pulled", dims)
        # print("New dims", dims)

        groups = tuple(tuple(g) for n, g in groupby(dims, key=lambda x: x.name))
        clashes = tuple(g for g in groups if len(g) > 1)

        subs = {}
        for c in clashes:
            # Preferentially rename dims that aren't root dims
            rdims = tuple(d for d in c if d.is_Root)
            ddims = tuple(d for d in c if not d.is_Root)

            for d in (ddims + rdims)[:-1]:
                if d in resolutions.keys():
                    subs[d] = resolutions[d]
                else:
                    try:
                        subs[d] = d._rebuild(d.name+str(count[d.name]))
                        count[d.name] += 1
                    except KeyError:
                        subs[d] = d._rebuild(d.name+'0')
                        count[d.name] = 1
                    resolutions[d] = subs[d]

        processed.append(uxreplace(e, subs))

    return processed

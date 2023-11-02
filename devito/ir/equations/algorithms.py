from collections.abc import Iterable

from sympy import sympify

from devito.symbolics import retrieve_indexed, uxreplace, retrieve_dimensions
from devito.tools import Ordering, as_tuple, flatten, filter_sorted, filter_ordered
from devito.types import Dimension, IgnoreDimSort
from devito.types.basic import AbstractFunction

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
            indices = []
            # import pdb;pdb.set_trace()
            for a, o in zip(i.indices, f._size_nodomain.left):
                # import pdb;pdb.set_trace()            
                indices.append(lower_exprs(a) + o)
                #try:
                #    assert (a+o-a.d) <= f._size_halo[a.d].left
                #except AttributeError:
                #    pass
                #except AssertionError:
                #    print("Illegal shifting that will cause OOB accesses")
                #    pass
                #except TypeError:
                #    import pdb;pdb.set_trace()
                #    pass


            # Substitute spacing (spacing only used in own dimension)
            indices = [i.xreplace({d.spacing: 1, -d.spacing: -1})
                       for i, d in zip(indices, f.dimensions)]

            # import pdb;pdb.set_trace()

            # Apply substitutions, if necessary
            if dimension_map:
                indices = [j.xreplace(dimension_map) for j in indices]

            mapper[i] = f.indexed[indices]

            print(indices)
            print(f._size_nodomain.left)
            for k, v in mapper.items():
                for n, ind in enumerate(k.indices):
                    # print(v.indices[n] - k.indices[n])
                    # import pdb;pdb.set_trace()
                    offsetv = [v.indices[n].subs(fs, 0) for fs in v.indices[n].free_symbols] or 0
                    offsetk = [k.indices[n].subs(fs, 0) for fs in k.indices[n].free_symbols] or 0
                    # shift = v.indices[n] - k.indices[n]
                    try:
                        if (offsetv[0] - offsetk[0]) > f._size_nodomain.left[n] and offsetv[0] > 0:
                        # if offset > f._size_nodomain.left[n] and f._offset_domain[n]:
                            print("problem!")
                    except:
                        pass
                        # import pdb;pdb.set_trace()

            # import pdb;pdb.set_trace()
            # need to find if the access is OOB

            # #for n, ind in enumerate(indices):
                # import pdb;pdb.set_trace()
                # offset = indices[n] - i.indices[n]
            #    offset = indices[n].subs(f.dimensions[n], 0) - f._offset_domain[n]
            #    try:
            #        if offset > f._size_nodomain.left[n]:
            #            print(offset)
                        # import pdb;pdb.set_trace()
            #    except:
            #        import pdb;pdb.set_trace()


        # Add dimensions map to the mapper in case dimensions are used
        # as an expression, i.e. Eq(u, x, subdomain=xleft)
        mapper.update(dimension_map)
        # Add the user-supplied substitutions
        mapper.update(subs)
        # Apply mapper to expression
        processed.append(uxreplace(expr, mapper))

    if isinstance(expressions, Iterable):
    #     import pdb;pdb.set_trace()
    #    for expr in processed:
    #        exlist = retrieve_indexed(expr)
    #        for ex in exlist:
    #            for n, ind in enumerate(ex.indices):
    #                ex = [ex.indices[n].subs(fs, 0) for fs in ex.indices[n].free_symbols] or 0
            
    #                import pdb;pdb.set_trace()


        # processed = check_offs(expressions=expressions)
        
        return processed
    else:
        assert len(processed) == 1
        return processed.pop()



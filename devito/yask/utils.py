import cgen as c

from devito.cgen_utils import INT, ccode
from devito.dse import FunctionFromPointer, ListInitializer, retrieve_indexed
from devito.nodes import Element, Expression
from devito.tools import as_tuple
from devito.visitors import FindNodes, Transformer

from devito.yask import namespace


def make_sharedptr_funcall(call, params, sharedptr):
    return FunctionFromPointer(call, FunctionFromPointer('get', sharedptr), params)


def make_grid_accesses(node):
    """
    Construct a new Iteration/Expression based on ``node``, in which all
    :class:`interfaces.Indexed` accesses have been converted into YASK grid
    accesses.
    """

    def make_grid_gets(expr):
        mapper = {}
        indexeds = retrieve_indexed(expr)
        data_carriers = [i for i in indexeds if i.base.function.from_YASK]
        for i in data_carriers:
            name = namespace['code-grid-name'](i.base.function.name)
            args = [ListInitializer([INT(make_grid_gets(j)) for j in i.indices])]
            mapper[i] = make_sharedptr_funcall(namespace['code-grid-get'], args, name)
        return expr.xreplace(mapper)

    mapper = {}
    for i, e in enumerate(FindNodes(Expression).visit(node)):
        lhs, rhs = e.expr.args

        # RHS translation
        rhs = make_grid_gets(rhs)

        # LHS translation
        if e.output_function.from_YASK:
            name = namespace['code-grid-name'](e.output_function.name)
            args = [rhs]
            args += [ListInitializer([INT(make_grid_gets(i)) for i in lhs.indices])]
            handle = make_sharedptr_funcall(namespace['code-grid-put'], args, name)
            processed = Element(c.Statement(ccode(handle)))
        else:
            # Writing to a scalar temporary
            processed = Expression(e.expr.func(lhs, rhs), dtype=e.dtype)

        mapper.update({e: processed})

    return Transformer(mapper).visit(node)


def convert_multislice(multislice, shape, offsets, mode='get'):
    """
    Convert a multislice into a format suitable to YASK's get_elements_{...}
    and set_elements_{...} grid routines.

    A multislice is the typical object received by NumPy ndarray's __getitem__
    and __setitem__ methods; this function, therefore, converts NumPy indices
    into YASK indices.

    In particular, a multislice is either a single element or an iterable of
    elements. An element can be a slice object, an integer index, or a tuple
    of integer indices.

    In the general case in which ``multislice`` is an iterable, each element in
    the iterable corresponds to a dimension in ``shape``. In this case, an element
    can be either a slice or an integer index, but not a tuple of integers.

    If ``multislice`` is a single element,  then it is interpreted as follows: ::

        * slice object: the slice spans the whole shape;
        * single (integer) index: shape is one-dimensional, and the index
          represents a specific entry;
        * a tuple of (integer) indices: it must be ``len(multislice) == len(shape)``,
          and each entry in ``multislice`` corresponds to a specific entry in a
          dimension in ``shape``.

    The returned value is a 3-tuple ``(starts, ends, shapes)``, where ``starts,
    ends, shapes`` are lists of length ``len(shape)``. By taking ``starts[i]`` and
    `` ends[i]``, one gets the start and end points of the section of elements to
    be accessed along dimension ``i``; ``shapes[i]`` gives the size of the section.
    """

    # Note: the '-1' below are because YASK uses '<=', rather than '<', to check
    # bounds when iterating over grid dimensions

    assert mode in ['get', 'set']
    multislice = as_tuple(multislice)

    # Convert dimensions
    cstart = []
    cstop = []
    cshape = []
    for i, v in enumerate(multislice):
        if isinstance(v, slice):
            if v.step is not None:
                raise NotImplementedError("Unsupported stepping != 1.")
            if v.start is None:
                start = 0
            elif v.start < 0:
                start = shape[i] + v.start
            else:
                start = v.start
            cstart.append(start)
            if v.stop is None:
                stop = shape[i] - 1
            elif v.stop < 0:
                stop = shape[i] + v.stop
            else:
                stop = v.stop
            cstop.append(stop)
            cshape.append(cstop[-1] - cstart[-1] + 1)
        else:
            if v is None:
                start = 0
                stop = shape[i] - 1
            elif v < 0:
                start = shape[i] + v
                stop = shape[i] + v
            else:
                start = v
                stop = v
            cstart.append(start)
            cstop.append(stop)
            if mode == 'set':
                cshape.append(1)

    # Remainder (e.g., requesting A[1] and A has shape (3,3))
    nremainder = len(shape) - len(multislice)
    cstart.extend([0]*nremainder)
    cstop.extend([shape[i + j] - 1 for j in range(1, nremainder + 1)])
    cshape.extend([shape[i + j] for j in range(1, nremainder + 1)])

    assert len(shape) == len(cstart) == len(cstop) == len(offsets)

    # Shift by the specified offsets
    cstart = [j + i for i, j in zip(offsets, cstart)]
    cstop = [j + i for i, j in zip(offsets, cstop)]

    return cstart, cstop, cshape

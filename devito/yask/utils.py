from devito.tools import as_tuple

__all__ = ['convert_multislice']


def convert_multislice(multislice, shape, mode='get'):
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
            cstart.append(v.start or 0)
            cstop.append((v.stop or shape[i]) - 1)
            cshape.append(cstop[-1] - cstart[-1] + 1)
        else:
            cstart.append(normalize_index(v if v is not None else 0, shape))
            cstop.append(normalize_index(v if v is not None else (shape[i]-1), shape))
            if mode == 'set':
                cshape.append(1)

    # Remainder (e.g., requesting A[1] and A has shape (3,3))
    nremainder = len(shape) - len(multislice)
    cstart.extend([0]*nremainder)
    cstop.extend([shape[i + j] - 1 for j in range(nremainder)])
    cshape.extend([shape[i + j] for j in range(nremainder)])

    return cstart, cstop, cshape


def normalize_index(index, shape):
    normalized = [i if i >= 0 else j + i for i, j in zip(as_tuple(index), shape)]
    return normalized[0] if len(normalized) == 1 else normalized

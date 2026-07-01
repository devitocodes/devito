import numpy as np

from devito.tools import Tag, as_tuple, is_integer

__all__ = [
    'NONLOCAL',
    'PROJECTED',
    'Index',
    'convert_index',
    'flip_idx',
    'index_apply_modulo',
    'index_dist_to_repl',
    'index_handle_oob',
    'index_is_basic',
    'loc_data_idx',
]


class Index(Tag):
    pass
NONLOCAL = Index('nonlocal')  # noqa
PROJECTED = Index('projected')


def index_is_basic(idx):
    if is_integer(idx):
        return True
    elif isinstance(idx, (slice, np.ndarray)):
        return False
    else:
        return all(is_integer(i) or (i is NONLOCAL) for i in idx)


def index_apply_modulo(idx, modulo):
    if is_integer(idx):
        return idx % modulo
    elif isinstance(idx, slice):
        if idx.start is None:
            start = idx.start
        elif idx.start >= 0:
            start = idx.start % modulo
        else:
            start = -(idx.start % modulo)
        if idx.stop is None:
            stop = idx.stop
        elif idx.stop >= 0:
            stop = idx.stop % (modulo + 1)
        else:
            stop = -(idx.stop % (modulo + 1))
        return slice(start, stop, idx.step)
    elif isinstance(idx, (tuple, list)):
        return [i % modulo for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx
    else:
        raise ValueError(f"Cannot apply modulo to index of type `{type(idx)}`")


def index_dist_to_repl(idx, decomposition):
    """Convert a distributed array index into a replicated array index."""
    if decomposition is None:
        return PROJECTED if is_integer(idx) else slice(None)

    # Derive shift value
    if isinstance(idx, slice):
        value = idx.start if idx.step is None or idx.step >= 0 else idx.stop
    else:
        value = idx
    if value is None:
        value = 0
    elif not is_integer(value):
        raise ValueError(f"Cannot derive shift value from type `{type(value)}`")

    if value < 0:
        value += decomposition.glb_max + 1

    # Convert into absolute local index
    idx = decomposition.index_glb_to_loc(idx, rel=False)

    if is_integer(idx):
        return PROJECTED
    elif idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i - value for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx - value
    elif isinstance(idx, slice):
        if idx.step is not None and idx.step < 0 and idx.stop is None:
            return slice(idx.start - value, None, idx.step)
        return slice(idx.start - value, idx.stop - value, idx.step)
    else:
        raise ValueError(f"Cannot apply shift to type `{type(idx)}`")


def convert_index(idx, decomposition, mode='glb_to_loc'):
    """Convert a global index into a local index or vise versa according to mode."""
    if is_integer(idx) or isinstance(idx, slice):
        return decomposition(idx, mode=mode)
    elif isinstance(idx, (tuple, list)):
        return [decomposition(i, mode=mode) for i in idx]
    elif isinstance(idx, np.ndarray):
        if mode == 'glb_to_loc' and len(decomposition) == 1 \
                and not decomposition.loc_empty:
            # A single (non-distributed) subdomain holds the whole axis, so the
            # global-to-local map is just a negative-index wrap. Vectorize it
            # instead of calling `index_glb_to_loc` per element via np.vectorize.
            n = decomposition.glb_max - decomposition.glb_min + 1
            return np.where(idx < 0, idx + n, idx).astype(idx.dtype)
        return np.vectorize(lambda i: decomposition(i, mode=mode))(idx).astype(idx.dtype)
    else:
        raise ValueError(f"Cannot convert index of type `{type(idx)}` ")


def index_handle_oob(idx):
    # distributed.index_glb_to_loc returns None when the index is globally
    # legal, but out-of-bounds for the calling MPI rank
    if idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i for i in idx if i is not None]
    elif isinstance(idx, np.ndarray):
        if idx.dtype == bool:
            # A boolean mask, nothing to do
            return idx
        elif idx.ndim == 1:
            # Only an object array can carry the None sentinels; a plain numeric
            # index has no out-of-bounds entries to drop, so skip the O(n) copy.
            if idx.dtype != object:
                return idx
            return np.delete(idx, np.where(idx == None))  # noqa
        else:
            raise ValueError("Cannot identify OOB accesses when using "
                             "multidimensional index arrays")
    else:
        return idx


def loc_data_idx(loc_idx):
    """
    Return tuple of slices containing the unflipped idx corresponding to loc_idx.
    By 'unflipped' we mean that if a slice has a negative step, we wish to retrieve
    the corresponding indices but not in reverse order.

    Examples
    --------
    >>> loc_data_idx(slice(11, None, -3))
    (slice(2, 12, 3),)
    """
    retval = []
    for i in as_tuple(loc_idx):
        if isinstance(i, slice) and i.step is not None and i.step == -1:
            if i.stop is None:
                retval.append(slice(0, i.start+1, -i.step))
            else:
                retval.append(slice(i.stop+1, i.start+1, -i.step))
        elif isinstance(i, slice) and i.step is not None and i.step < -1:
            if i.stop is None:
                lmin = i.start
                while lmin >= 0:
                    lmin += i.step
                retval.append(slice(lmin-i.step, i.start+1, -i.step))
            else:
                retval.append(slice(i.stop+1, i.start+1, -i.step))
        elif is_integer(i):
            retval.append(slice(i, i+1, 1))
        else:
            retval.append(i)
    return as_tuple(retval)


def flip_idx(idx, decomposition):
    """
    This function serves two purposes:
    1) To Convert a global index with containing a slice with step < 0 to a 'mirrored'
       index with all slice steps > 0.
    2) Normalize indices with slices containing negative start/stops.

    Parameters
    ----------
    idx: tuple of slices/ints/tuples
        Representation of the indices that require processing.
    decomposition : tuple of Decomposition
        The data decomposition, for each dimension.

    Examples
    --------
    In the following examples, the domain consists of 12 indices, split over
    four subdomains [0, 3]. We pick 2 as local subdomain.

    >>> from devito.data import Decomposition, flip_idx
    >>> d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)
    >>> d
    Decomposition([0,2], [3,4], <<[5,7]>>, [8,11])

    Example with negative stepped slices:

    >>> idx = (slice(4, None, -1))
    >>> fidx = flip_idx(idx, (d,))
    >>> fidx
    (slice(None, 5, 1),)

    Example with negative start/stops:

    >>> idx2 = (slice(-4, -1, 1))
    >>> fidx2 = flip_idx(idx2, (d,))
    >>> fidx2
    (slice(8, 11, 1),)
    """
    processed = []
    for i, j in zip(as_tuple(idx), decomposition, strict=False):
        if isinstance(i, slice) and i.step is not None and i.step < 0:
            if i.start is None:
                stop = None
            elif i.start > 0:
                stop = i.start + 1
            else:
                stop = i.start + j.glb_max + 2
            if i.stop is None:
                start = None
            elif i.stop > 0:
                start = i.stop + 1
            else:
                start = i.stop + j.glb_max + 2
            processed.append(slice(start, stop, -i.step))
        elif isinstance(i, slice):
            if i.start is not None and i.start < 0:
                start = i.start + j.glb_max + 1
            else:
                start = i.start
            if i.stop is not None and i.stop < 0:  # noqa: SIM108
                stop = i.stop + j.glb_max + 1
            else:
                stop = i.stop
            processed.append(slice(start, stop, i.step))
        else:
            processed.append(slice(i, i+1, 1))
    return as_tuple(processed)

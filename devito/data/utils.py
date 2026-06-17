import numpy as np

from devito.tools import (
    Tag, as_list, as_tuple, dtype_to_mpidtype, is_integer, prod
)

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
    'index_is_integer_sequence',
    'loc_data_idx',
    'mpi_advanced_1d_get',
    'mpi_advanced_1d_index',
    'mpi_advanced_1d_set',
    'mpi_index_maps',
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


def index_is_integer_sequence(idx):
    """
    Return True for a one-dimensional integer array-like index.

    NumPy treats ``a[[...]]`` and ``a[np.array([...])]`` as advanced indexing.
    This helper recognizes only the integer-sequence form used by
    ``mpi_advanced_1d_*``; slices and scalars continue through the existing
    global-to-local index conversion.
    """
    if not isinstance(idx, (list, tuple, np.ndarray)):
        return False

    arr = np.asarray(idx)
    return (arr.ndim == 1 and
            (arr.size == 0 or np.issubdtype(arr.dtype, np.integer)))


def _index_has_integer_sequence(idx):
    """
    Cheaply reject ordinary basic indexing before normalizing ``idx``.

    This keeps the new MPI advanced-indexing path out of the hot path for
    scalar/slice-only accesses.
    """
    if isinstance(idx, np.ndarray) or isinstance(idx, list):
        return index_is_integer_sequence(idx)
    elif isinstance(idx, tuple):
        return any(index_is_integer_sequence(i) for i in idx)
    else:
        return False


def mpi_advanced_1d_index(data, glb_idx):
    """
    Normalize the supported 1D MPI advanced-indexing subset.

    Returns ``None`` when ``glb_idx`` can be handled by the regular Data
    indexing path. Otherwise returns the normalized index, advanced axis,
    global integer indices, and the axis decomposition used by
    :func:`mpi_advanced_1d_get` or :func:`mpi_advanced_1d_set`.

    The supported case is deliberately narrow: exactly one MPI-distributed
    dimension, indexed by exactly one one-dimensional integer sequence. The
    integer sequence is interpreted as global indices in that distributed
    dimension. All other dimensions must use basic indexing, i.e. slices or
    scalar integers.
    """
    if not _index_has_integer_sequence(glb_idx):
        return None

    if not data._is_decomposed:
        return None

    glb_idx = data._normalize_index(glb_idx)
    if len(glb_idx) > data.ndim:
        return None
    elif len(glb_idx) < data.ndim:
        glb_idx = glb_idx + (slice(None),)*(data.ndim - len(glb_idx))

    distributed = []
    advanced = []
    for i, d in enumerate(data._decomposition):
        if d is None:
            continue

        distributed.append(i)
        if index_is_integer_sequence(glb_idx[i]):
            advanced.append(i)

    if not advanced:
        return None
    elif len(distributed) != 1:
        raise NotImplementedError(
            "Advanced indexing with MPI-distributed Data is currently "
            "supported only for data with a single distributed dimension"
        )
    elif len(advanced) != 1:
        raise NotImplementedError(
            "Advanced indexing with MPI-distributed Data supports a single "
            "integer index array"
        )

    axis = advanced[0]
    for i, idx in enumerate(glb_idx):
        if i != axis and index_is_integer_sequence(idx):
            raise NotImplementedError(
                "Advanced indexing with MPI-distributed Data supports a single "
                "integer index array"
            )

    indices = np.asarray(glb_idx[axis], dtype=np.int64)
    return glb_idx, axis, indices, data._decomposition[axis]


def mpi_advanced_1d_get(data, glb_idx, axis, indices, decomposition,
                        target_getter):
    """
    Read MPI-distributed ``data`` using one global integer index sequence.

    This implements the read side of the supported NumPy advanced-indexing
    subset. Each rank supplies the global indices it wants in its local output.
    The helper asks owner ranks for those entries and returns a normal NumPy
    array ordered exactly like the caller's index sequence.
    """
    indices, owners = _mpi_advanced_1d_owners(data, indices, decomposition)
    shape = _mpi_advanced_1d_result_shape(data, glb_idx, axis, indices)
    positions, scount, rcount, recv_indices = \
        _mpi_advanced_1d_indices_alltoall(data, indices, owners)

    source, source_axis = target_getter(glb_idx, axis)
    source = np.moveaxis(source, source_axis, 0)
    local_offsets = _mpi_advanced_1d_local_offsets(recv_indices, decomposition)
    send_data = np.ascontiguousarray(source[local_offsets])

    payload_shape = send_data.shape[1:]
    recv_data = _mpi_advanced_1d_data_alltoall(data, send_data, rcount, scount,
                                               payload_shape)

    ret = np.empty(shape, dtype=data.dtype)
    ret_view = np.moveaxis(ret, source_axis, 0)
    ret_view[np.concatenate(positions)] = recv_data
    return ret


def mpi_advanced_1d_set(data, glb_idx, val, axis, indices, decomposition,
                        target_getter):
    """
    Assign into MPI-distributed ``data`` using one global integer index sequence.

    ``val`` is interpreted as local to the calling rank and ordered according
    to ``indices``. The helper routes each value to the rank that owns the
    corresponding global index, preserving NumPy broadcasting for the
    non-distributed dimensions.
    """
    indices, owners = _mpi_advanced_1d_owners(data, indices, decomposition)
    shape = _mpi_advanced_1d_result_shape(data, glb_idx, axis, indices)
    val = np.asarray(val, dtype=data.dtype)
    val = np.broadcast_to(val, shape)
    value_axis = sum(not is_integer(i) for i in glb_idx[:axis])
    val = np.ascontiguousarray(np.moveaxis(val, value_axis, 0))
    payload_shape = val.shape[1:]

    positions, scount, rcount, recv_indices = \
        _mpi_advanced_1d_indices_alltoall(data, indices, owners)

    send_data = _mpi_advanced_1d_pack_axis0(val, positions)
    recv_data = _mpi_advanced_1d_data_alltoall(data, send_data, scount, rcount,
                                               payload_shape)

    error = None
    if recv_indices.size and np.unique(recv_indices).size != recv_indices.size:
        error = "Duplicate global indices in MPI-distributed advanced assignment"
    _mpi_advanced_1d_error(data, error)

    target, target_axis = target_getter(glb_idx, axis)
    target = np.moveaxis(target, target_axis, 0)
    local_offsets = _mpi_advanced_1d_local_offsets(recv_indices, decomposition)
    target[local_offsets] = recv_data


def _mpi_advanced_1d_error(data, error):
    """Raise the first error reported by any rank, on every rank."""
    if data._distributor.nprocs > 1:
        errors = data._distributor.comm.allgather(error)
        error = next((i for i in errors if i is not None), None)

    if error is not None:
        raise ValueError(error)


def _mpi_advanced_1d_owners(data, indices, decomposition):
    """Map global indices to owning ranks, normalizing negative indices."""
    indices = indices.copy()
    if decomposition.glb_max is not None:
        indices[indices < 0] += decomposition.glb_max + 1

    owners = np.full(indices.size, -1, dtype=np.int32)
    for i, r in enumerate(decomposition):
        if r.size:
            owners[np.isin(indices, r)] = i

    error = None
    if np.any(owners < 0):
        error = "Advanced index contains out-of-bounds global indices"
    _mpi_advanced_1d_error(data, error)

    return indices, owners


def _mpi_advanced_1d_result_shape(data, glb_idx, axis, indices):
    """Return the NumPy result shape for the supported advanced-index case."""
    shape = []
    for i, idx in enumerate(glb_idx):
        if is_integer(idx):
            continue
        elif i == axis:
            shape.append(indices.size)
        elif isinstance(idx, slice):
            shape.append(len(range(*idx.indices(data.shape[i]))))
        else:
            raise NotImplementedError(
                "Advanced indexing with MPI-distributed Data supports only "
                "integer arrays, integer indices, and slices"
            )
    return tuple(shape)


def _mpi_advanced_1d_local_offsets(indices, decomposition):
    """Convert global indices received by an owner rank to local offsets."""
    return np.array([decomposition.index_glb_to_loc(int(i)) for i in indices],
                    dtype=np.int64)


def _mpi_advanced_1d_positions(owners, nprocs):
    """
    Group local index positions by destination rank.

    ``positions[r]`` are the entries in the caller's advanced index whose data
    must be exchanged with rank ``r``.
    """
    return [np.where(owners == i)[0] for i in range(nprocs)]


def _mpi_advanced_1d_indices_alltoall(data, indices, owners):
    """
    Exchange requested global indices with their owner ranks.

    The returned counts describe the same exchange pattern used later for the
    payload values.
    """
    comm = data._distributor.comm
    nprocs = data._distributor.nprocs
    positions = _mpi_advanced_1d_positions(owners, nprocs)
    scount = np.array([i.size for i in positions], dtype=np.int32)
    rcount = _mpi_advanced_1d_count_alltoall(comm, scount)

    send_indices = _mpi_advanced_1d_pack_axis0(indices, positions)
    recv_indices = np.empty(int(np.sum(rcount)), dtype=np.int64)
    idtype = dtype_to_mpidtype(np.int64)
    _mpi_advanced_1d_alltoallv(comm, send_indices, recv_indices, scount,
                               rcount, idtype)

    return positions, scount, rcount, recv_indices


def _mpi_advanced_1d_data_alltoall(data, send_data, scount, rcount, payload_shape):
    """Exchange payload arrays using an already established index exchange."""
    comm = data._distributor.comm
    payload_size = prod(payload_shape)
    recv_data = np.empty((int(np.sum(rcount)), *payload_shape), dtype=data.dtype)

    dscount = scount * payload_size
    drcount = rcount * payload_size
    mpitype = dtype_to_mpidtype(data.dtype)
    _mpi_advanced_1d_alltoallv(comm, send_data, recv_data, dscount, drcount,
                               mpitype)

    return recv_data


def _mpi_advanced_1d_count_alltoall(comm, scount):
    rcount = np.empty_like(scount)
    comm.Alltoall(scount, rcount)
    return rcount


def _mpi_advanced_1d_alltoallv(comm, send, recv, scount, rcount, mpitype):
    sdisp = _mpi_advanced_1d_displacements(scount)
    rdisp = _mpi_advanced_1d_displacements(rcount)
    comm.Alltoallv([send, scount, sdisp, mpitype],
                   [recv, rcount, rdisp, mpitype])


def _mpi_advanced_1d_displacements(counts):
    displacements = np.empty_like(counts, dtype=np.int32)
    displacements[0] = 0
    displacements[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return displacements


def _mpi_advanced_1d_pack_axis0(array, positions):
    """Pack an axis-0 array in destination-rank order."""
    return np.ascontiguousarray(np.concatenate([array[i] for i in positions],
                                               axis=0))


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


def mpi_index_maps(loc_idx, shape, topology, coords, comm):
    """
    Generate various data structures used to determine what MPI communication
    is required. The function creates the following:

    owners: An array of shape ``shape`` where each index signifies the rank on which
    that data is stored.

    send: An array of shape ``shape`` where each index signifies the rank to which
    data belonging to that index should be sent.

    global_si: An array of ``shape`` shape where each index contains the global index
    to which that index should be sent.

    local_si: An array of shape ``shape`` where each index contains the local index
    (on the destination rank) to which that index should be sent.

    Parameters
    ----------
    loc_idx : tuple of slices
        The coordinates of interest to the current MPI rank.
    shape: np.array of tuples
        Array containing the local shape of data to each rank.
    topology: tuple
        Topology of the decomposed domain.
    coords: tuple of tuples
        The coordinates of each MPI rank in the decomposed domain, ordered
        based on the MPI rank.
    comm : MPI communicator

    Examples
    --------
    An array is given by A = [[  0,  1,  2,  3],
                              [  4,  5,  6,  7],
                              [  8,  9, 10, 11],
                              [ 12, 13, 14, 15]],
    which is then distributed over four ranks such that on rank 0:

    A = [[ 0, 1],
         [ 4, 5]],

    on rank 1:

    A = [[ 2, 3],
         [ 6, 7]],

    on rank 2:

    A = [[  8,  9],
         [ 12, 13]],

    on rank 3:

    A = [[ 10, 11],
         [ 14, 15]].

    Taking the slice A[2:0:-1, 2:0:-1] the expected output (in serial) is

    [[  0,  1,  2,  3],
     [  4, 10,  9,  7],
     [  8,  6,  5, 11],
     [ 12, 13, 14, 15]],

    Hence, in this case the following would be generated:

    owners = [[0, 1],
              [2, 3]],

    send = [[3, 2],
            [1, 0]],

    global_si = [[(2, 2), (2, 1)],
                 [(1, 2), (1, 1)]],

    local_si = [[(0, 0), (0, 1)],
                [(1, 0), (1, 1)]].
    """

    nprocs = comm.Get_size()

    # Gather data structures from all ranks in order to produce the
    # relevant mappings.
    dat_len = np.zeros(topology, dtype=tuple)
    for j in range(nprocs):
        dat_len[coords[j]] = comm.bcast(shape, root=j)
        if any(k == 0 for k in dat_len[coords[j]]):
            dat_len[coords[j]] = as_tuple([0]*len(dat_len[coords[j]]))

    # If necessary, add the time index to the `topology` as this will
    # be required to correctly construct various maps.
    if len(np.amax(dat_len)) > len(topology):
        topology = as_list(topology)
        coords = [as_list(l) for l in coords]
        for _ in range(len(np.amax(dat_len)) - len(topology)):
            topology.insert(0, 1)
            for e in coords:
                e.insert(0, 0)
        topology = as_tuple(topology)
        coords = as_tuple([as_tuple(i) for i in coords])
    dat_len = dat_len.reshape(topology)
    dat_len_cum = distributed_data_size(dat_len, coords, topology)

    # This 'transform' will be required to produce the required maps
    transform = []
    for i in as_tuple(loc_idx):
        if isinstance(i, slice):
            if i.step is not None:
                transform.append(slice(None, None, np.sign(i.step)))
            else:
                transform.append(slice(None, None, None))
        else:
            transform.append(slice(0, 1, None))
    transform = as_tuple(transform)

    global_size = dat_len_cum[coords[-1]]

    indices = np.zeros(global_size, dtype=tuple)
    global_si = np.zeros(global_size, dtype=tuple)
    it = np.nditer(indices, flags=['refs_ok', 'multi_index'])
    while not it.finished:
        index = it.multi_index
        indices[index] = index
        it.iternext()
    global_si[:] = indices[transform]

    # Create the 'rank' slices
    rank_slice = []
    for j in coords:
        this_rank = []
        for k in dat_len[j]:
            this_rank.append(slice(0, k, 1))
        rank_slice.append(this_rank)
    # Normalize the slices:
    n_rank_slice = []
    for i in range(len(rank_slice)):
        my_coords = coords[i]
        if any([j.stop == j.start for j in rank_slice[i]]):
            n_rank_slice.append(as_tuple([None]*len(rank_slice[i])))
            continue
        if i == 0:
            n_rank_slice.append(as_tuple(rank_slice[i]))
            continue
        left_neighbours = []
        for j in range(len(my_coords)):
            left_coord = list(my_coords)
            left_coord[j] -= 1
            left_neighbours.append(as_tuple(left_coord))
        left_neighbours = as_tuple(left_neighbours)
        n_slice = []
        for j in range(len(my_coords)):
            if left_neighbours[j][j] < 0:
                start = 0
                stop = dat_len_cum[my_coords][j]
            else:
                start = dat_len_cum[left_neighbours[j]][j]
                stop = dat_len_cum[my_coords][j]
            n_slice.append(slice(start, stop, 1))
        n_rank_slice.append(as_tuple(n_slice))
    n_rank_slice = as_tuple(n_rank_slice)

    # Now fill each elements owner:
    owners = np.zeros(global_size, dtype=np.int32)
    send = np.zeros(global_size, dtype=np.int32)
    for i in range(len(n_rank_slice)):
        if any([j is None for j in n_rank_slice[i]]):
            continue
        else:
            owners[n_rank_slice[i]] = i
    send[:] = owners[transform]

    # Construct local_si
    local_si = np.zeros(global_size, dtype=tuple)
    it = np.nditer(local_si, flags=['refs_ok', 'multi_index'])
    while not it.finished:
        index = it.multi_index
        owner = owners[index]
        my_slice = n_rank_slice[owner]
        rnorm_index = []
        for j, k in zip(my_slice, index, strict=True):
            rnorm_index.append(k-j.start)
        local_si[index] = as_tuple(rnorm_index)
        it.iternext()
    return owners, send, global_si, local_si


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


def distributed_data_size(shape, coords, topology):
    """
    Compute the cumulative shape of the distributed data (cshape).

    Parameters
    -----------
    shape: np.array of tuples
        Array containing the local shape of data to each rank.
    coords: tuple of tuples
        The coordinates of each MPI rank in the decomposed domain, ordered
        based on the MPI rank.
    topology: tuple
        Topology of the decomposed domain.

    Examples
    --------
    Given a set of distributed data such that:

    shape = [[ (2, 2), (2, 2)],
             [ (2, 2), (2, 2)]],

    (that is, there are 4 ranks and the data on each rank has shape (2, 2)).
    cshape will be returned as

    cshape = [[ (2, 2), (2, 4)],
              [ (4, 2), (4, 4)]].
    """
    cshape = np.zeros(topology, dtype=tuple)
    for i in range(len(coords)):
        my_coords = coords[i]
        if i == 0:
            cshape[my_coords] = shape[my_coords]
            continue
        left_neighbours = []
        for j in range(len(my_coords)):
            left_coord = list(my_coords)
            left_coord[j] -= 1
            left_neighbours.append(as_tuple(left_coord))
        left_neighbours = as_tuple(left_neighbours)
        n_dat = []  # Normalised data size
        if sum(shape[my_coords]) == 0:
            prev_dat_len = []
            for j in left_neighbours:
                if any(d < 0 for d in j):
                    pass
                else:
                    prev_dat_len.append(cshape[j])
            func = lambda a, b: max([d[b] for d in a])
            max_dat_len = []
            for j in range(len(my_coords)):
                max_dat_len.append(func(prev_dat_len, j))
            cshape[my_coords] = as_tuple(max_dat_len)
        else:
            for j in range(len(my_coords)):
                if left_neighbours[j][j] < 0:
                    c_dat = shape[my_coords][j]
                    n_dat.append(c_dat)
                else:
                    c_dat = shape[my_coords][j]  # Current length
                    p_dat = cshape[left_neighbours[j]][j]  # Previous length
                    n_dat.append(c_dat+p_dat)
            cshape[my_coords] = as_tuple(n_dat)
    return cshape

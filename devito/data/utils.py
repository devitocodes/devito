import numpy as np

from devito.tools import Tag, as_tuple, is_integer

__all__ = ['Index', 'NONLOCAL', 'PROJECTED', 'index_is_basic', 'index_apply_modulo',
           'index_dist_to_repl', 'convert_index', 'index_handle_oob',
           'loc_data_idx', 'mpi_index_maps']


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
        raise ValueError("Cannot apply modulo to index of type `%s`" % type(idx))


def index_dist_to_repl(idx, decomposition):
    """Convert a distributed array index into a replicated array index."""
    if decomposition is None:
        return PROJECTED if is_integer(idx) else slice(None)

    # Derive shift value
    if isinstance(idx, slice):
        if idx.step is None or idx.step >= 0:
            value = idx.start
        else:
            value = idx.stop
    else:
        value = idx
    if value is None:
        value = 0
    elif not is_integer(value):
        raise ValueError("Cannot derive shift value from type `%s`" % type(value))

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
        if idx.step is not None and idx.step < 0:
            if idx.stop is None:
                return slice(idx.start - value, None, idx.step)
        return slice(idx.start - value, idx.stop - value, idx.step)
    else:
        raise ValueError("Cannot apply shift to type `%s`" % type(idx))


def convert_index(idx, decomposition, mode='glb_to_loc'):
    """Convert a global index into a local index or vise versa according to mode."""
    if is_integer(idx) or isinstance(idx, slice):
        return decomposition(idx, mode=mode)
    elif isinstance(idx, (tuple, list)):
        return [decomposition(i, mode=mode) for i in idx]
    elif isinstance(idx, np.ndarray):
        return np.vectorize(lambda i: decomposition(i, mode=mode))(idx)
    else:
        raise ValueError("Cannot convert index of type `%s` " % type(idx))


def index_handle_oob(idx):
    # distributed.index_glb_to_loc returns None when the index is globally
    # legal, but out-of-bounds for the calling MPI rank
    if idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i for i in idx if i is not None]
    elif isinstance(idx, np.ndarray):
        if idx.dtype == np.bool_:
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
    data beloning to that index should be sent.

    global_si: An array of ``shape`` shape where each index contains the global index
    to which that index should be sent.

    local_si: An array of shape ``shape`` where each index contains the local index
    (on the destination rank) to which that index should be sent.

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
    # Work out the cumulative data shape
    dat_len_cum = np.zeros(topology, dtype=tuple)
    for i in range(nprocs):
        my_coords = coords[i]
        if i == 0:
            dat_len_cum[my_coords] = dat_len[my_coords]
            continue
        left_neighbours = []
        for j in range(len(my_coords)):
            left_coord = list(my_coords)
            left_coord[j] -= 1
            left_neighbours.append(as_tuple(left_coord))
        left_neighbours = as_tuple(left_neighbours)
        n_dat = []  # Normalised data size
        for j in range(len(my_coords)):
            if left_neighbours[j][j] < 0:
                c_dat = dat_len[my_coords][j]
                n_dat.append(c_dat)
            else:
                c_dat = dat_len[my_coords][j]  # Current length
                p_dat = dat_len[left_neighbours[j]][j]  # Previous length
                n_dat.append(c_dat+p_dat)
        dat_len_cum[my_coords] = as_tuple(n_dat)
    # This 'transform' will be required to produce the required maps
    transform = []
    for i in as_tuple(loc_idx):
        if isinstance(i, slice):
            if i.step is not None:
                transform.append(slice(None, None, np.sign(i.step)))
            else:
                transform.append(slice(None, None, None))
        else:
            transform.append(0)
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
        for j, k in zip(my_slice, index):
            rnorm_index.append(k-j.start)
        local_si[index] = as_tuple(rnorm_index)
        it.iternext()
    return owners, send, global_si, local_si

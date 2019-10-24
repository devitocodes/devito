from collections.abc import Iterable
from functools import wraps

import numpy as np

from devito.data.allocators import ALLOC_FLAT
from devito.data.utils import *
from devito.parameters import configuration
from devito.tools import Tag, as_tuple, is_integer

__all__ = ['Data']


class Data(np.ndarray):

    """
    A numpy.ndarray supporting distributed Dimensions.

    Parameters
    ----------
    shape : tuple of ints
        Shape of created array.
    dtype : numpy.dtype
        The data type of the raw data.
    decomposition : tuple of Decomposition, optional
        The data decomposition, for each dimension.
    modulo : tuple of bool, optional
        If the i-th entry is True, then the i-th array dimension uses modulo indexing.
    allocator : MemoryAllocator, optional
        Used to allocate memory. Defaults to ``ALLOC_FLAT``.
    distributor : Distributor, optional
        The distributor from which the original decomposition was produced. Note that
        the decomposition Parameter above may be different to distributor.decomposition.

    Notes
    -----
    NumPy array subclassing is described at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    Any view or copy created from ``self``, for instance via a slice operation
    or a universal function ("ufunc" in NumPy jargon), will still be of type
    Data.
    """

    def __new__(cls, shape, dtype, decomposition=None, modulo=None, allocator=ALLOC_FLAT,
                distributor=None):
        assert len(shape) == len(modulo)
        ndarray, memfree_args = allocator.alloc(shape, dtype)
        obj = ndarray.view(cls)
        obj._allocator = allocator
        obj._memfree_args = memfree_args
        obj._decomposition = decomposition or (None,)*len(shape)
        obj._modulo = modulo or (False,)*len(shape)
        obj._distributor = distributor

        # This cannot be a property, as Data objects constructed from this
        # object might not have any `decomposition`, but they would still be
        # distributed. Hence, in `__array_finalize__` we must copy this value
        obj._is_distributed = any(i is not None for i in obj._decomposition)

        # Saves the last index used in `__getitem__`. This allows `__array_finalize__`
        # to reconstruct information about the computed view (e.g., `decomposition`)
        obj._index_stash = None

        # Sanity check -- A Dimension can't be at the same time modulo-iterated
        # and MPI-distributed
        assert all(i is None for i, j in zip(obj._decomposition, obj._modulo)
                   if j is True)

        return obj

    def __del__(self):
        if self._memfree_args is None:
            return
        self._allocator.free(*self._memfree_args)
        self._memfree_args = None

    def __array_finalize__(self, obj):
        # `self` is the newly created object
        # `obj` is the object from which `self` was created
        if obj is None:
            # `self` was created through __new__()
            return

        self._distributor = None
        self._index_stash = None

        # Views or references created via operations on `obj` do not get an
        # explicit reference to the underlying data (`_memfree_args`). This makes sure
        # that only one object (the "root" Data) will free the C-allocated memory
        self._memfree_args = None

        if type(obj) != Data:
            # Definitely from view casting
            self._is_distributed = False
            self._modulo = tuple(False for i in range(self.ndim))
            self._decomposition = (None,)*self.ndim
        elif obj._index_stash is not None:
            # From `__getitem__`
            self._is_distributed = obj._is_distributed
            self._distributor = obj._distributor
            glb_idx = obj._normalize_index(obj._index_stash)
            self._modulo = tuple(m for i, m in zip(glb_idx, obj._modulo)
                                 if not is_integer(i))
            decomposition = []
            for i, dec in zip(glb_idx, obj._decomposition):
                if is_integer(i):
                    continue
                elif dec is None:
                    decomposition.append(None)
                else:
                    decomposition.append(dec.reshape(i))
            self._decomposition = tuple(decomposition)
        else:
            self._is_distributed = obj._is_distributed
            self._distributor = obj._distributor
            if self.ndim == obj.ndim:
                # E.g., from a ufunc, such as `np.add`
                self._modulo = obj._modulo
                self._decomposition = obj._decomposition
            else:
                # E.g., from a reduction operation such as `np.mean` or `np.all`
                self._modulo = tuple(False for i in range(self.ndim))
                self._decomposition = (None,)*self.ndim

    @property
    def _local(self):
        """A view of ``self`` with global indexing disabled."""
        ret = self.view()
        ret._is_distributed = False
        return ret

    def _global(self, glb_idx, decomposition):
        """A "global" view of ``self`` over a given Decomposition."""
        if self._is_distributed:
            raise ValueError("Cannot derive a decomposed view from a decomposed Data")
        if len(decomposition) != self.ndim:
            raise ValueError("`decomposition` should have ndim=%d entries" % self.ndim)
        ret = self[glb_idx]
        ret._decomposition = decomposition
        ret._is_distributed = any(i is not None for i in decomposition)
        return ret

    def _check_idx(func):
        """Check if __getitem__/__setitem__ may require communication across MPI ranks."""
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            glb_idx = args[0]
            if len(args) > 1 and isinstance(args[1], Data) \
                    and args[1]._is_mpi_distributed:
                comm_type = index_by_index
            elif data._is_mpi_distributed:
                for i in as_tuple(glb_idx):
                    if isinstance(i, slice) and i.step is not None and i.step < 0:
                        comm_type = index_by_index
                        break
                    else:
                        comm_type = serial
            else:
                comm_type = serial
            kwargs['comm_type'] = comm_type
            return func(data, *args, **kwargs)
        return wrapper

    @property
    def _is_mpi_distributed(self):
        return self._is_distributed and configuration['mpi']

    def __repr__(self):
        return super(Data, self._local).__repr__()

    @_check_idx
    def __getitem__(self, glb_idx, comm_type):
        loc_idx = self._index_glb_to_loc(glb_idx)
        if comm_type is index_by_index:
            # Retrieve the pertinent local data prior to mpi send/receive operations
            data_idx = loc_data_idx(loc_idx)
            self._index_stash = flip_idx(glb_idx, self._decomposition)
            local_val = super(Data, self).__getitem__(data_idx)
            self._index_stash = None

            comm = self._distributor.comm
            rank = comm.Get_rank()

            owners, send, global_si, local_si = \
                mpi_index_maps(loc_idx, local_val.shape, self._distributor.topology,
                               self._distributor.all_coords, comm)

            retval = Data(local_val.shape, local_val.dtype.type,
                          decomposition=local_val._decomposition,
                          modulo=(False,)*len(local_val.shape))
            it = np.nditer(owners, flags=['refs_ok', 'multi_index'])
            # Iterate over each element of data
            while not it.finished:
                index = it.multi_index
                if rank == owners[index] and rank == send[index]:
                    # Current index and destination index are on the same rank
                    loc_ind = local_si[index]
                    send_ind = local_si[global_si[index]]
                    retval.data[send_ind] = local_val.data[loc_ind]
                elif rank == owners[index]:
                    # Current index is on this rank and hence need to send
                    # the data to the appropriate rank
                    loc_ind = local_si[index]
                    send_rank = send[index]
                    send_ind = global_si[index]
                    send_val = local_val.data[loc_ind]
                    reqs = comm.isend([send_ind, send_val], dest=send_rank)
                    reqs.wait()
                elif rank == send[index]:
                    # Current rank is required to receive data from this index
                    recval = comm.irecv(source=owners[index])
                    local_dat = recval.wait()
                    loc_ind = local_si[local_dat[0]]
                    retval.data[loc_ind] = local_dat[1]
                else:
                    pass
                it.iternext()
            return retval
        elif loc_idx is NONLOCAL:
            # Caller expects a scalar. However, `glb_idx` doesn't belong to
            # self's data partition, so None is returned
            return None
        else:
            self._index_stash = glb_idx
            retval = super(Data, self).__getitem__(loc_idx)
            self._index_stash = None
            return retval

    @_check_idx
    def __setitem__(self, glb_idx, val, comm_type):
        loc_idx = self._index_glb_to_loc(glb_idx)
        if loc_idx is NONLOCAL:
            # no-op
            return
        elif np.isscalar(val):
            if index_is_basic(loc_idx):
                # Won't go through `__getitem__` as it's basic indexing mode,
                # so we should just propage `loc_idx`
                super(Data, self).__setitem__(loc_idx, val)
            else:
                super(Data, self).__setitem__(glb_idx, val)
        elif isinstance(val, Data) and val._is_distributed:
            if comm_type is index_by_index:
                glb_idx, val = self._process_args(glb_idx, val)
                val_idx = as_tuple([slice(i.glb_min, i.glb_max+1, 1) for
                                    i in val._decomposition])
                idx = self._set_global_idx(val, glb_idx, val_idx)
                comm = self._distributor.comm
                nprocs = self._distributor.nprocs
                # Prepare global lists:
                data_global = []
                idx_global = []
                for j in range(nprocs):
                    data_global.append(comm.bcast(np.array(val), root=j))
                    idx_global.append(comm.bcast(idx, root=j))
                # Set the data:
                for j in range(nprocs):
                    skip = any(i is None for i in idx_global[j]) \
                        or data_global[j].size == 0
                    if not skip:
                        self.__setitem__(idx_global[j], data_global[j])
            elif self._is_distributed:
                # `val` is decomposed, `self` is decomposed -> local set
                super(Data, self).__setitem__(glb_idx, val)
            else:
                # `val` is decomposed, `self` is replicated -> gatherall-like
                raise NotImplementedError
        elif isinstance(val, np.ndarray):
            if self._is_distributed:
                # `val` is replicated, `self` is decomposed -> `val` gets decomposed
                glb_idx = self._normalize_index(glb_idx)
                glb_idx, val = self._process_args(glb_idx, val)
                val_idx = [index_dist_to_repl(i, dec) for i, dec in
                           zip(glb_idx, self._decomposition)]
                if NONLOCAL in val_idx:
                    # no-op
                    return
                val_idx = tuple([i for i in val_idx if i is not PROJECTED])
                # NumPy broadcasting note:
                # When operating on two arrays, NumPy compares their shapes
                # element-wise. It starts with the trailing dimensions, and works
                # its way forward. Two dimensions are compatible when
                # * they are equal, or
                # * one of them is 1
                # Conceptually, below we apply the same rule
                val_idx = val_idx[len(val_idx)-val.ndim:]
                processed = []
                # Handle step size > 1
                for i, j in zip(glb_idx, val_idx):
                    if isinstance(i, slice) and i.step is not None and i.step > 1 and \
                            j.stop > j.start:
                        processed.append(slice(j.start, j.stop, 1))
                    else:
                        processed.append(j)
                val_idx = as_tuple(processed)
                val = val[val_idx]
            else:
                # `val` is replicated`, `self` is replicated -> plain ndarray.__setitem__
                pass
            super(Data, self).__setitem__(glb_idx, val)
        elif isinstance(val, Iterable):
            if self._is_mpi_distributed:
                raise NotImplementedError("With MPI, data can only be set "
                                          "via scalars, numpy arrays or "
                                          "other data ")
            super(Data, self).__setitem__(glb_idx, val)
        else:
            raise ValueError("Cannot insert obj of type `%s` into a Data" % type(val))

    def _normalize_index(self, idx):
        if isinstance(idx, np.ndarray):
            # Advanced indexing mode
            return (idx,)
        else:
            idx = as_tuple(idx)
            if any(i is Ellipsis for i in idx):
                # Explicitly replace the Ellipsis
                items = (slice(None),)*(self.ndim - len(idx) + 1)
                items = idx[:idx.index(Ellipsis)] + items + idx[idx.index(Ellipsis)+1:]
            else:
                items = idx + (slice(None),)*(self.ndim - len(idx))
            # Normalize slice steps:
            processed = [slice(i.start, i.stop, 1) if
                         (isinstance(i, slice) and i.step is None)
                         else i for i in items]
            return as_tuple(processed)

    def _process_args(self, idx, val):
        """If comm_type is parallel we need to first retrieve local unflipped data."""
        if any(isinstance(i, slice) and i.step is not None and i.step < 0
               for i in as_tuple(idx)):
            processed = []
            transform = []
            for j, k in zip(idx, self._distributor.glb_shape):
                if isinstance(j, slice) and j.step is not None and j.step < 0:
                    if j.start is None:
                        stop = None
                    else:
                        stop = j.start + 1
                    if j.stop is None and j.start is None:
                        start = int(np.mod(k-1, -j.step))
                    elif j.stop is None:
                        start = int(np.mod(j.start, -j.step))
                    else:
                        start = j.stop + 1
                    processed.append(slice(start, stop, -j.step))
                    transform.append(slice(None, None, np.sign(j.step)))
                else:
                    processed.append(j)
            if isinstance(val, Data) and len(transform) > 0 and \
                    len(val._distributor.shape) > len(val.shape):
                # Rebuild the distributor since the dimension of the slice
                # is different to that of the original array
                distributor = \
                    val._distributor._rebuild(val.shape,
                                              self._distributor.dimensions,
                                              self._distributor.comm)
                new_val = Data(val.shape, val.dtype.type,
                               decomposition=val._decomposition, modulo=val._modulo,
                               distributor=distributor)
                slc = as_tuple([slice(None, None, 1) for j in transform])
                new_val[slc] = val[slc]
                return as_tuple(processed), new_val[as_tuple(transform)]
            else:
                return as_tuple(processed), val[as_tuple(transform)]
        else:
            return idx, val

    def _index_glb_to_loc(self, glb_idx):
        glb_idx = self._normalize_index(glb_idx)
        if len(glb_idx) > self.ndim:
            # Maybe user code is trying to add a new axis (see np.newaxis),
            # so the resulting array will be higher dimensional than `self`
            if self._is_mpi_distributed:
                raise ValueError("Cannot increase dimensionality of MPI-distributed Data")
            else:
                # As by specification, we are forced to ignore modulo indexing
                return glb_idx

        loc_idx = []
        for i, s, mod, dec in zip(glb_idx, self.shape, self._modulo, self._decomposition):
            if mod is True:
                # Need to wrap index based on modulo
                v = index_apply_modulo(i, s)
            elif self._is_distributed is True and dec is not None:
                # Need to convert the user-provided global indices into local indices.
                # Obviously this will have no effect if MPI is not used
                try:
                    v = convert_index(i, dec, mode='glb_to_loc')
                except TypeError:
                    if self._is_mpi_distributed:
                        raise NotImplementedError("Unsupported advanced indexing with "
                                                  "MPI-distributed Data")
                    v = i
            else:
                v = i

            # Handle non-local, yet globally legal, indices
            v = index_handle_oob(v)

            loc_idx.append(v)

        # Deal with NONLOCAL accesses
        if NONLOCAL in loc_idx:
            if len(loc_idx) == self.ndim and index_is_basic(loc_idx):
                # Caller expecting a scalar -- it will eventually get None
                loc_idx = [NONLOCAL]
            else:
                # Caller expecting an array -- it will eventually get a 0-length array
                loc_idx = [slice(-1, -2) if i is NONLOCAL else i for i in loc_idx]

        return loc_idx[0] if len(loc_idx) == 1 else tuple(loc_idx)

    def _set_global_idx(self, val, idx, val_idx):
        """
        Compute the global indices to which val (the locally stored data) correspond.
        """
        data_loc_idx = as_tuple(val._index_glb_to_loc(val_idx))
        data_glb_idx = []
        # Convert integers to slices so that shape dims are preserved
        if is_integer(as_tuple(idx)[0]):
            data_glb_idx.append(slice(0, 1, 1))
        for i, j in zip(data_loc_idx, val._decomposition):
            if not j.loc_empty:
                data_glb_idx.append(j.index_loc_to_glb(i))
            else:
                data_glb_idx.append(None)
        mapped_idx = []
        # Based `data_glb_idx` the indices to which the locally stored data
        # block correspond can now be computed:
        for i, j, k in zip(data_glb_idx, as_tuple(idx), self._decomposition):
            if isinstance(j, slice) and j.start is None:
                norm = 0
            elif isinstance(j, slice) and j.start is not None:
                if j.start >= 0:
                    norm = j.start
                else:
                    norm = j.start+k.glb_max+1
            else:
                norm = j
            if i is not None:
                if isinstance(j, slice) and j.step is not None:
                    stop = j.step*i.stop+norm
                else:
                    stop = i.stop+norm
            if i is not None:
                if isinstance(j, slice) and j.step is not None:
                    mapped_idx.append(slice(j.step*i.start+norm,
                                            stop, j.step))
                else:
                    mapped_idx.append(slice(i.start+norm, stop, i.step))
            else:
                mapped_idx.append(None)
        return as_tuple(mapped_idx)

    def reset(self):
        """Set all Data entries to 0."""
        self[:] = 0.0


class CommType(Tag):
    pass
index_by_index = CommType('index_by_index')  # noqa
serial = CommType('serial')

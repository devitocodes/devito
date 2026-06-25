from collections.abc import Iterable

import numpy as np

from devito.data.allocators import ALLOC_ALIGNED
from devito.data.utils import *
from devito.logger import warning
from devito.parameters import configuration
from devito.tools import as_list, as_tuple, is_integer

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
        If the i-th entry is True, then the i-th array dimension uses modulo
        indexing.
    allocator : MemoryAllocator, optional
        Used to allocate memory. Defaults to `ALLOC_ALIGNED`.
    distributor : Distributor, optional
        The distributor from which the original decomposition was produced.
        Note that `decomposition` may differ from `distributor.decomposition`.
    padding : int or 2-tuple of ints, optional
        The number of points that are allocated before and after the data,
        that is in addition to the requested shape. Defaults to 0.

    Notes
    -----
    NumPy array subclassing is described at: ::

        https://numpy.org/doc/stable/user/basics.subclassing.html

    Any view or copy created from ``self``, for instance via a slice operation
    or a universal function ("ufunc" in NumPy jargon), will still be of type
    `Data`.
    """

    def __new__(cls, shape, dtype, decomposition=None, modulo=None,
                allocator=ALLOC_ALIGNED, distributor=None, padding=0):
        assert len(shape) == len(modulo)
        ndarray, memfree_args = allocator.alloc(shape, dtype, padding=padding)
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
        assert all(i is None for i, j in zip(obj._decomposition, obj._modulo, strict=True)
                   if j is True)

        return obj

    def __del__(self):
        if getattr(self, "_memfree_args", None) is None:
            # NOTE: The need for `getattr`, in place of `self._memfree_args`, was
            # suggested for the first time in issue #1184. However, it appears
            # that even though, as described in the issue, we initialize the
            # attribute in `__array_finalize__`, an AttributeError exception may
            # still be raised in some obscure situations. Our best explanation
            # so far is that this is due to (un)pickling (as often used in a
            # Dask/Distributed context), which may (re)create a Data object
            # without going through `__array_finalize__`
            return
        self._allocator.free(*self._memfree_args)
        self._memfree_args = None

    def __reduce__(self):
        warning("Pickling of `Data` objects is not supported. Casting to `numpy.ndarray`")
        return self.view(np.ndarray).__reduce__()

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

        if not issubclass(type(obj), Data):
            # Definitely from view casting
            self._is_distributed = False
            self._modulo = tuple(False for i in range(self.ndim))
            self._decomposition = (None,)*self.ndim
            self._allocator = ALLOC_ALIGNED
        elif obj._index_stash is not None:
            # From `__getitem__`
            self._distributor = obj._distributor
            glb_idx = obj._normalize_index(obj._index_stash)
            self._modulo = tuple(
                m
                for i, m in zip(glb_idx, obj._modulo, strict=False)
                if not is_integer(i)
            )
            decomposition = []
            for i, dec in zip(glb_idx, obj._decomposition, strict=False):
                if is_integer(i):
                    continue
                elif dec is None:
                    decomposition.append(None)
                elif isinstance(i, slice):
                    # Indexing is local: the induced decomposition follows exact
                    # NumPy slicing (handles strided/reversed slices), unlike the
                    # boundary-adjusting `reshape` used for halos.
                    decomposition.append(dec.index_decomposition(i))
                else:
                    decomposition.append(dec.reshape(i))
            self._decomposition = tuple(decomposition)
            self._allocator = obj._allocator
            decomp = any(i is not None for i in self._decomposition)
            self._is_distributed = decomp and obj._is_distributed
        else:
            self._distributor = obj._distributor
            self._allocator = obj._allocator
            self._is_distributed = obj._is_distributed
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
        if self._is_decomposed:
            raise ValueError("Cannot derive a decomposed view from a decomposed Data")
        if len(decomposition) != self.ndim:
            raise ValueError(f"`decomposition` should have ndim={self.ndim} entries")
        ret = self[glb_idx]
        ret._decomposition = decomposition
        ret._is_distributed = any(i is not None for i in decomposition)
        return ret

    def _prune_shape(self, shape):
        # Reduce distributed MPI `Data`'s shape to that of an equivalently
        # sliced numpy array.
        decomposition = tuple(d for d in self._decomposition if d.size > 1)
        retval = self.reshape(shape)
        retval._decomposition = decomposition
        return retval

    @property
    def _is_decomposed(self):
        return self._is_distributed and configuration['mpi'] and \
            self._distributor.is_parallel

    def _scattered_exchange(self, glb_idx):
        """
        Return an `Exchange` when `glb_idx` advanced-indexes a
        distributed axis (the "unbalanced"/scattered case, where the value is
        rank-local and ordered by `glb_idx`), otherwise `None` so the caller
        falls back to the regular (basic-indexing) path.
        """
        # Imported lazily: the redistribution engine depends on `devito.mpi`,
        # which depends on `devito.types`, which loads `data.py` -- a top-level
        # import here would be circular (same reason as `dtype_to_mpidtype`).
        from devito.data.distributed import cached_exchange
        from devito.data.distributed.selection import index_has_array

        if not self._is_decomposed or not index_has_array(glb_idx, self.ndim):
            return None
        exchange = cached_exchange(self, glb_idx)
        distributed = {a for a, d in enumerate(self._decomposition) if d is not None}
        if any(a in distributed for a in exchange._selection.advanced_axes):
            return exchange
        return None

    def __repr__(self):
        return super(Data, self._local).__repr__()

    def __str__(self):
        return super(Data, self._local).__str__()

    def transpose(self, *axes):
        """
        Return a view of ``self`` with permuted axes.

        Overridden so that ``_decomposition``, ``_modulo`` (and the convenience
        flag ``_is_distributed``) are permuted to match the new axis ordering,
        rather than copied verbatim from ``self`` as ``__array_finalize__``
        would otherwise leave them. Without this, a subsequent slice on the
        transposed view (e.g. ``f.data.T[::2, ::2]``) is computed against the
        wrong per-axis decomposition and silently returns a wrong-shaped
        result (see issue #2187).
        """
        # Accept the same axis-spec forms as ``numpy.ndarray.transpose``:
        # no args, a single ``None``, a single tuple/list, or per-arg.
        if len(axes) == 1:
            axes = as_tuple(axes[0])
        new_order = (
            tuple(range(self.ndim - 1, -1, -1)) if not axes
            else tuple(ax % self.ndim for ax in axes)
        )

        ret = super().transpose(*axes)
        ret._decomposition = tuple(self._decomposition[i] for i in new_order)
        ret._modulo = tuple(self._modulo[i] for i in new_order)
        ret._is_distributed = any(d is not None for d in ret._decomposition)
        return ret

    def swapaxes(self, axis1, axis2):
        """
        Return a view of ``self`` with ``axis1`` and ``axis2`` swapped, with
        ``_decomposition`` / ``_modulo`` swapped in the same way (see
        `transpose`).
        """
        axis1 = axis1 % self.ndim
        axis2 = axis2 % self.ndim
        ret = super().swapaxes(axis1, axis2)
        order = list(range(self.ndim))
        order[axis1], order[axis2] = order[axis2], order[axis1]
        ret._decomposition = tuple(self._decomposition[i] for i in order)
        ret._modulo = tuple(self._modulo[i] for i in order)
        ret._is_distributed = any(d is not None for d in ret._decomposition)
        return ret

    @property
    def T(self):
        """
        The transposed array. Overridden so the C-level ``ndarray.T`` shortcut
        also permutes the per-axis metadata (see `transpose`).
        """
        return self.transpose()

    def __getitem__(self, glb_idx):
        exchange = self._scattered_exchange(glb_idx)
        if exchange is not None:
            return exchange.get()
        loc_idx = self._index_glb_to_loc(glb_idx)
        if loc_idx is NONLOCAL:
            # Caller expects a scalar. However, `glb_idx` doesn't belong to
            # self's data partition, so None is returned
            return None
        else:
            self._index_stash = glb_idx
            retval = super().__getitem__(loc_idx)
            self._index_stash = None
            return retval

    def __setitem__(self, glb_idx, val):
        if not (isinstance(val, Data) and val._is_decomposed):
            exchange = self._scattered_exchange(glb_idx)
            if exchange is not None:
                exchange.put(val)
                return
        loc_idx = self._index_glb_to_loc(glb_idx)

        if loc_idx is NONLOCAL:
            # no-op
            return
        elif np.isscalar(val):
            if index_is_basic(loc_idx):
                # Won't go through `__getitem__` as it's basic indexing mode,
                # so we should just propagate `loc_idx`
                super().__setitem__(loc_idx, val)
            else:
                super().__setitem__(glb_idx, val)
        elif isinstance(val, Data) and val._is_decomposed:
            # Lazy import to avoid a circular dependency (see
            # `_scattered_exchange`).
            from devito.data.distributed import redistribute_set

            # Structured point-to-point redistribution covers the well-defined
            # cases. The legacy broadcast-based fallback is reached only when a
            # scalar indexes a distributed axis: `val` then has a rank-dependent
            # shape (it is owned by some ranks and empty on others), which the
            # structured engine cannot route, so every rank exchanges its block.
            if redistribute_set(self, glb_idx, val):
                return
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
        elif isinstance(val, np.ndarray):
            if self._is_decomposed:
                # `val` is replicated, `self` is decomposed -> `val` gets decomposed
                glb_idx = self._normalize_index(glb_idx)
                glb_idx, val = self._process_args(glb_idx, val)
                val_idx = [index_dist_to_repl(i, dec) for i, dec in
                           zip(glb_idx, self._decomposition, strict=True)]
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
                for i, j in zip(glb_idx, val_idx, strict=False):
                    if isinstance(i, slice) and i.step is not None and i.step > 1 and \
                            j.stop > j.start:
                        processed.append(slice(j.start, j.stop, 1))
                    else:
                        processed.append(j)
                val_idx = as_tuple(processed)
                val = val[val_idx]
            super().__setitem__(glb_idx, val)
        elif isinstance(val, Iterable):
            if self._is_decomposed:
                raise NotImplementedError("With MPI, data can only be set "
                                          "via scalars, numpy arrays or "
                                          "other data ")
            super().__setitem__(glb_idx, val)
        else:
            raise ValueError(f"Cannot insert obj of type `{type(val)}` into a Data")

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
        """Retrieve local unflipped data for a distributed value assignment."""
        if (len(as_tuple(idx)) < len(val.shape)) and (len(val.shape) <= len(self.shape)):
            idx_processed = as_list(idx)
            for _ in range(len(val.shape)-len(as_tuple(idx))):
                idx_processed.append(slice(None, None, 1))
            idx = as_tuple(idx_processed)
        if any(isinstance(i, slice) and i.step is not None and i.step < 0
               for i in as_tuple(idx)):
            processed = []
            transform = []
            for j, k in zip(idx, self._distributor.glb_shape, strict=True):
                if isinstance(j, slice) and j.step is not None and j.step < 0:
                    stop = None if j.start is None else j.start + 1
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
            if self._is_decomposed:
                raise ValueError("Cannot increase dimensionality of MPI-distributed Data")
            else:
                # As by specification, we are forced to ignore modulo indexing
                return glb_idx

        loc_idx = []
        for i, s, mod, dec in zip(
            glb_idx, self.shape, self._modulo, self._decomposition, strict=False
        ):
            if mod is True:
                # Need to wrap index based on modulo
                v = index_apply_modulo(i, s)
            elif self._is_distributed is True and dec is not None:
                # Convert the user-provided global indices into local indices.
                try:
                    v = convert_index(i, dec, mode='glb_to_loc')
                except TypeError as e:
                    if self._is_decomposed:
                        raise NotImplementedError(
                            "Unsupported advanced indexing with MPI-distributed Data"
                        ) from e
                    v = i
            else:
                v = i

            # Handle non-local, yet globally legal, indices
            v = index_handle_oob(v)

            loc_idx.append(v)

        # Deal with NONLOCAL accesses
        if any(j is NONLOCAL for j in loc_idx):
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
        for i, j in zip(data_loc_idx, val._decomposition, strict=True):
            if not j.loc_empty:
                data_glb_idx.append(j.index_loc_to_glb(i))
            else:
                data_glb_idx.append(None)
        mapped_idx = []
        # Add any integer indices that were not present in `val_idx`.
        if len(as_list(idx)) > len(data_glb_idx):
            for index, value in enumerate(idx):
                if is_integer(value) and index > 0:
                    data_glb_idx.insert(index, value)
        # Based on `data_glb_idx` the indices to which the locally stored data
        # block correspond can now be computed:
        for i, j, k in zip(
            data_glb_idx, as_tuple(idx), self._decomposition, strict=False
        ):
            if is_integer(j):
                mapped_idx.append(j)
                continue
            elif isinstance(j, slice) and j.start is None:
                norm = 0
            elif isinstance(j, slice) and j.start is not None:
                norm = j.start if j.start >= 0 else j.start+k.glb_max+1
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

    def _gather(self, start=None, stop=None, step=1, rank=0):
        """
        Gather (a slice of) the distributed data into a NumPy array on a single
        rank, returning `None` on the others. See the public `data_gather`
        method of `Function`.

        Indexing is local (each rank already holds its induced result block);
        gathering is the explicit collect step -- every rank sends its block and
        the result indices it owns, and `rank` reassembles the global array.
        """
        if not isinstance(rank, int):
            raise TypeError('rank must be passed as an integer value.')

        def as_axes(v):
            return [v]*self.ndim if (v is None or isinstance(v, int)) else list(v)
        start, stop, step = as_axes(start), as_axes(stop), as_axes(step)
        idx = tuple(slice(i, j, k) for i, j, k
                    in zip(start, stop, step, strict=True))

        if not (self._is_decomposed and self._distributor.nprocs > 1):
            return np.array(self[idx])

        local = self[idx]
        block = np.ascontiguousarray(np.asarray(local))
        # Result indices this rank owns, per axis (replicated axis -> full extent)
        owned = [
            np.arange(block.shape[ax], dtype=np.int64) if dec is None
            else np.asarray(dec.loc_abs_numb, dtype=np.int64) - (dec.glb_min or 0)
            for ax, dec in enumerate(local._decomposition)
        ]

        comm = self._distributor.comm
        gathered = comm.gather((block, owned), root=rank)
        if comm.Get_rank() != rank:
            return None

        shape = tuple(
            len(range(*sl.indices(d.size if d is not None else self.shape[ax])))
            for ax, (sl, d) in enumerate(zip(idx, self._decomposition, strict=True))
        )
        out = np.empty(shape, dtype=self.dtype)
        for blk, idxs in gathered:
            if blk.size:
                out[np.ix_(*idxs)] = blk
        return out

    def reset(self):
        """Set all Data entries to 0."""
        self[:] = 0.0

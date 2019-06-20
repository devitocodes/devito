from collections import Iterable

import numpy as np

from devito.data.allocators import ALLOC_FLAT
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

    Notes
    -----
    NumPy array subclassing is described at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    Any view or copy created from ``self``, for instance via a slice operation
    or a universal function ("ufunc" in NumPy jargon), will still be of type
    Data.
    """

    def __new__(cls, shape, dtype, decomposition=None, modulo=None, allocator=ALLOC_FLAT):
        assert len(shape) == len(modulo)
        ndarray, memfree_args = allocator.alloc(shape, dtype)
        obj = ndarray.view(cls)
        obj._allocator = allocator
        obj._memfree_args = memfree_args
        obj._decomposition = decomposition or (None,)*len(shape)
        obj._modulo = modulo or (False,)*len(shape)

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

    @property
    def _is_mpi_distributed(self):
        return self._is_distributed and configuration['mpi']

    def __repr__(self):
        return super(Data, self._local).__repr__()

    def __getitem__(self, glb_idx):
        loc_idx = self._convert_index(glb_idx)
        if loc_idx is NONLOCAL:
            # Caller expects a scalar. However, `glb_idx` doesn't belong to
            # self's data partition, so None is returned
            return None
        else:
            self._index_stash = glb_idx
            retval = super(Data, self).__getitem__(loc_idx)
            self._index_stash = None
            return retval

    def __setitem__(self, glb_idx, val):
        loc_idx = self._convert_index(glb_idx)
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
            if self._is_distributed:
                # `val` is decomposed, `self` is decomposed -> local set
                super(Data, self).__setitem__(glb_idx, val)
            else:
                # `val` is decomposed, `self` is replicated -> gatherall-like
                raise NotImplementedError
        elif isinstance(val, np.ndarray):
            if self._is_distributed:
                # `val` is replicated, `self` is decomposed -> `val` gets decomposed
                val_idx = self._normalize_index(glb_idx)
                val_idx = [index_dist_to_repl(i, dec) for i, dec in
                           zip(val_idx, self._decomposition)]
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
                val = val[val_idx]
            else:
                # `val` is replicated`, `self` is replicated -> plain ndarray.__setitem__
                pass
            super(Data, self).__setitem__(glb_idx, val)
        elif isinstance(val, Iterable):
            if self._is_mpi_distributed:
                raise NotImplementedError("With MPI data can only be set "
                                          "via scalars or numpy arrays")
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
                return idx[:idx.index(Ellipsis)] + items + idx[idx.index(Ellipsis)+1:]
            else:
                return idx + (slice(None),)*(self.ndim - len(idx))

    def _convert_index(self, glb_idx):
        glb_idx = self._normalize_index(glb_idx)

        if len(glb_idx) > self.ndim:
            # Maybe user code is trying to add a new axis (see np.newaxis),
            # so the resulting array will be higher dimensional than `self`,
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
                    v = index_glb_to_loc(i, dec)
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

    def reset(self):
        """Set all Data entries to 0."""
        self[:] = 0.0


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
    value = idx.start if isinstance(idx, slice) else idx
    if value is None:
        value = 0
    elif not is_integer(value):
        raise ValueError("Cannot derive shift value from type `%s`" % type(value))

    # Convert into absolute local index
    idx = decomposition.convert_index(idx, rel=False)

    if is_integer(idx):
        return PROJECTED
    elif idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i - value for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx - value
    elif isinstance(idx, slice):
        return slice(idx.start - value, idx.stop - value, idx.step)
    else:
        raise ValueError("Cannot apply shift to type `%s`" % type(idx))


def index_glb_to_loc(idx, decomposition):
    """Convert a global index into a local index."""
    if is_integer(idx) or isinstance(idx, slice):
        return decomposition(idx)
    elif isinstance(idx, (tuple, list)):
        return [decomposition(i) for i in idx]
    elif isinstance(idx, np.ndarray):
        return np.vectorize(lambda i: decomposition(i))(idx)
    else:
        raise ValueError("Cannot convert global index of type `%s` into a local index"
                         % type(idx))


def index_handle_oob(idx):
    # distributed.convert_index returns None when the index is globally
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

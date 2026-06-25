"""
Selection layer: the meaning of an index expression, independent of any layout.

A `Selection` normalizes an arbitrary NumPy index (scalars, slices,
negative steps, integer arrays, boolean masks) into a per-axis list of
selectors plus the bookkeeping needed to reconstruct the result shape. It is the
single place that encodes NumPy indexing semantics; it performs no
communication and knows nothing about MPI decomposition. This makes it fully
testable in serial against NumPy.
"""

from dataclasses import dataclass

import numpy as np

from devito.tools import is_integer

__all__ = ['Affine', 'Explicit', 'IndexScalar', 'Selection', 'index_has_array',
           'result_dims']


def index_has_array(idx, ndim):
    """
    True if `idx` contains an integer/boolean array component (advanced
    indexing), excluding Devito's legacy `data[[i, j, k]]` basic shorthand.
    Used as a cheap gate to keep basic indexing off the routing path.
    """
    if _is_legacy_multidim_basic(idx, ndim):
        return False
    elif isinstance(idx, tuple):
        return any(_is_advanced(i) for i in idx)
    return _is_advanced(idx)


@dataclass(frozen=True)
class IndexScalar:
    """A single global index on one axis. The axis is dropped from the result."""

    index: int


@dataclass(frozen=True)
class Affine:
    """
    A strided range of global indices on one axis (the encoding of a scalar-free
    slice, including negative steps). The axis is kept in the result.
    """

    start: int
    stop: int
    step: int

    @property
    def size(self):
        return len(range(self.start, self.stop, self.step))

    @property
    def coords(self):
        """The global indices addressed, in result order."""
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)


@dataclass(frozen=True)
class Explicit:
    """
    Arbitrary global indices on one advanced (array-indexed) axis. The stored
    `coords` are already broadcast against the other advanced axes and
    flattened, so all advanced axes share a common point ordering.
    """

    coords: np.ndarray


def _is_advanced(item):
    """True for an integer/boolean array-like index component."""
    if isinstance(item, np.ndarray):
        return item.ndim >= 1
    elif isinstance(item, (list, tuple)):
        arr = np.asarray(item)
        return arr.ndim >= 1 and (arr.size == 0 or
                                  np.issubdtype(arr.dtype, np.integer) or
                                  np.issubdtype(arr.dtype, np.bool_))
    return False


def _is_legacy_multidim_basic(idx, ndim):
    """
    Devito historically accepts a top-level `data[[0, 1, 2]]` on a 3D object as
    shorthand for `data[0, 1, 2]` (basic), rather than NumPy advanced indexing
    on axis 0. Preserve that for a bare list whose length matches the rank.
    """
    return (ndim > 1 and isinstance(idx, list) and len(idx) == ndim and
            all(is_integer(i) for i in idx))


def _axis_span(item):
    """Number of array axes consumed by one raw index component."""
    if item is np.newaxis:
        return 0
    elif isinstance(item, np.ndarray) and item.dtype == np.bool_:
        return item.ndim
    elif isinstance(item, (list, tuple)):
        arr = np.asarray(item)
        return arr.ndim if arr.dtype == np.bool_ else 1
    return 1


def _expand(idx, ndim):
    """
    Expand a raw index into exactly `ndim` per-axis components, replacing
    Ellipsis and padding with full slices. Boolean arrays are converted to their
    equivalent integer coordinate arrays (one per consumed axis) via nonzero.
    `np.newaxis` is not supported on distributed data.
    """
    if isinstance(idx, tuple) or _is_legacy_multidim_basic(idx, ndim):
        items = list(idx)
    else:
        items = [idx]

    if any(i is np.newaxis for i in items):
        raise NotImplementedError("np.newaxis is not supported on distributed Data")

    # Replace Ellipsis with the right number of full slices
    if any(i is Ellipsis for i in items):
        pos = next(k for k, i in enumerate(items) if i is Ellipsis)
        consumed = sum(_axis_span(i) for i in items if i is not Ellipsis)
        fill = [slice(None)] * (ndim - consumed)
        items = items[:pos] + fill + items[pos + 1:]

    # Expand boolean masks into integer coordinate arrays, one per consumed axis
    expanded = []
    for item in items:
        if isinstance(item, np.ndarray) and item.dtype == np.bool_:
            expanded.extend(np.nonzero(item))
        elif isinstance(item, (list, tuple)) and \
                np.asarray(item).dtype == np.bool_:
            expanded.extend(np.nonzero(np.asarray(item)))
        else:
            expanded.append(item)
    items = expanded

    # Pad trailing axes with full slices
    items += [slice(None)] * (ndim - len(items))
    if len(items) != ndim:
        raise IndexError(f"too many indices for array of dimension {ndim}")
    return items


class Selection:
    """
    Normalized, layout-independent meaning of an index expression.

    Attributes
    ----------
    selectors : tuple
        One `IndexScalar`, `Affine`, or `Explicit` per axis.
    advanced_axes : tuple of int
        The axes indexed by arrays (the single coupled advanced group). Their
        `Explicit` coords share one flattened point ordering.
    advanced_shape : tuple of int
        Broadcast shape of the advanced index arrays. `npoints` is its product.
    result_shape : tuple of int
        Shape of `data[idx]`.
    """

    def __init__(self, selectors, advanced_axes, advanced_shape,
                 advanced_at_front, result_shape):
        self.selectors = tuple(selectors)
        self.advanced_axes = tuple(advanced_axes)
        self.advanced_shape = tuple(advanced_shape)
        self.advanced_at_front = advanced_at_front
        self.result_shape = tuple(result_shape)

    @classmethod
    def from_index(cls, idx, shape):
        """
        Build a Selection for `idx` against a global array of `shape`.

        Parameters
        ----------
        idx : index expression
            Any NumPy index (scalar, slice, integer array, boolean mask, or a
            tuple thereof, with at most one Ellipsis).
        shape : tuple of int
            The global array shape the index is resolved against.

        Returns
        -------
        Selection
            The normalized, layout-independent meaning of `idx`.
        """
        ndim = len(shape)
        items = _expand(idx, ndim)

        # Classify each axis, collecting the advanced (array) ones
        selectors = [None] * ndim
        advanced_axes = []
        advanced_arrays = []
        for axis, (item, n) in enumerate(zip(items, shape, strict=True)):
            if is_integer(item):
                index = int(item)
                if index < 0:
                    index += n
                if not 0 <= index < n:
                    raise IndexError(f"index {item} is out of bounds for axis "
                                     f"{axis} with size {n}")
                selectors[axis] = IndexScalar(index)
            elif isinstance(item, slice):
                selectors[axis] = Affine(*item.indices(n))
            elif _is_advanced(item):
                arr = np.asarray(item, dtype=np.int64)
                advanced_axes.append(axis)
                advanced_arrays.append(arr)
            else:
                raise IndexError(f"unsupported index component {item!r} on axis "
                                 f"{axis}")

        # Broadcast the advanced index arrays into a common point ordering
        advanced_shape = ()
        if advanced_arrays:
            advanced_shape = np.broadcast_shapes(*(a.shape for a in advanced_arrays))
            for axis, arr in zip(advanced_axes, advanced_arrays, strict=True):
                coords = np.broadcast_to(arr, advanced_shape).reshape(-1).copy()
                n = shape[axis]
                coords[coords < 0] += n
                selectors[axis] = Explicit(coords)

        advanced_at_front = _advanced_at_front(advanced_axes)
        result_shape = _result_shape(selectors, advanced_axes, advanced_shape,
                                     advanced_at_front)
        return cls(selectors, advanced_axes, advanced_shape, advanced_at_front,
                   result_shape)

    @property
    def ndim(self):
        return len(self.selectors)

    @property
    def is_advanced(self):
        return len(self.advanced_axes) > 0

    @property
    def result_dims(self):
        """Tagged result dimensions in order (see module-level `result_dims`)."""
        return result_dims(self.selectors, self.advanced_axes,
                           self.advanced_shape, self.advanced_at_front)

    @property
    def npoints(self):
        """Number of coupled advanced points (1 if there is no advanced group)."""
        out = 1
        for s in self.advanced_shape:
            out *= int(s)
        return out

    def __repr__(self):
        return f"Selection({', '.join(repr(s) for s in self.selectors)})"


def _advanced_at_front(advanced_axes):
    """
    NumPy places the advanced-index block at the front of the result when the
    advanced axes are separated by a basic index, and in-place when they are
    contiguous.
    """
    if not advanced_axes:
        return False
    contiguous = list(range(advanced_axes[0], advanced_axes[0] + len(advanced_axes)))
    return list(advanced_axes) != contiguous


def result_dims(selectors, advanced_axes, advanced_shape, advanced_at_front):
    """
    Ordered result dimensions of `data[idx]`, each tagged `('basic', axis)`
    for a kept slice axis or `('adv', j)` for the j-th advanced (broadcast)
    dimension.

    This is the single definition of NumPy's advanced-index result ordering (the
    advanced block moves to the front when the advanced axes are separated by a
    basic index, otherwise it sits in place). Both the result shape and the
    plan's row/payload layout derive from it, so the ordering lives in one place.
    """
    dims = []
    if advanced_axes and advanced_at_front:
        dims += [('adv', j) for j in range(len(advanced_shape))]
    inserted = False
    for axis, s in enumerate(selectors):
        if isinstance(s, IndexScalar):
            continue
        elif isinstance(s, Explicit):
            if advanced_at_front:
                continue
            if not inserted:
                dims += [('adv', j) for j in range(len(advanced_shape))]
                inserted = True
        else:
            dims.append(('basic', axis))
    if advanced_axes and not advanced_at_front and not inserted:
        dims += [('adv', j) for j in range(len(advanced_shape))]
    return dims


def _result_shape(selectors, advanced_axes, advanced_shape, advanced_at_front):
    """Shape of `data[idx]`, derived from `result_dims`."""
    dims = result_dims(selectors, advanced_axes, advanced_shape, advanced_at_front)
    return tuple(selectors[v].size if kind == 'basic' else advanced_shape[v]
                 for kind, v in dims)

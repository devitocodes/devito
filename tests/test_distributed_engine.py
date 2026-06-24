"""
Serial unit tests for the distributed redistribution engine
(:mod:`devito.data.distributed`).

The engine's lower layers (selection, layout, plan) are pure and
communication-free by design, so they are exercised here in serial -- no MPI
ranks, no ``@pytest.mark.parallel``. The end-to-end transport and ``Data``
integration are covered by the MPI tests in ``test_data.py``/``test_mpi.py``.
"""

import numpy as np
import pytest

from devito.data import Decomposition
from devito.data.distributed.layout import Layout
from devito.data.distributed.selection import (
    Affine, Explicit, Scalar, Selection, index_has_array, result_dims
)

# A reference global shape used throughout; indices below are validated against
# the NumPy result on an array of this shape.
SHAPE = (4, 5, 6)


class TestSelection:

    """``Selection`` encodes NumPy indexing semantics, layout-independent."""

    @pytest.mark.parametrize('idx', [
        # basic indexing
        (2,),
        (slice(None),),
        (slice(1, 4, 2),),
        (slice(None, None, -1),),
        (slice(3, 0, -2),),
        (slice(None), slice(None), 3),
        (1, 2, 3),
        (-1, -2),
        (slice(-3, -1),),
        # ellipsis / padding
        (Ellipsis, 2),
        (1, Ellipsis),
        (Ellipsis,),
        # advanced (array) indexing
        (np.array([0, 2, 3]),),
        (np.array([0, 2]), slice(1, 4)),
        (slice(None), np.array([0, 1, 4])),
        (np.array([[0, 1], [2, 3]]),),
        # contiguous advanced axes (stay in place)
        (np.array([0, 1]), np.array([2, 3])),
        # advanced axes separated by a basic index (block moves to front)
        (np.array([0, 1]), 2, np.array([0, 3])),
        (np.array([0, 1]), slice(None), np.array([0, 3])),
        # broadcast advanced indices
        (np.array([[0], [1]]), np.array([0, 2, 4])),
        # boolean masks
        (np.array([True, False, True, True]),),
        (slice(None), np.array([True, False, True, True, False])),
    ])
    def test_result_shape_matches_numpy(self, idx):
        """The induced result shape matches NumPy's for the same index."""
        ref = np.empty(SHAPE)
        selection = Selection.from_index(idx, SHAPE)
        assert selection.result_shape == ref[idx].shape

    def test_selector_types(self):
        """Each axis is classified as Scalar / Affine / Explicit as expected."""
        selection = Selection.from_index((2, slice(1, 4), np.array([0, 1])), SHAPE)
        s0, s1, s2 = selection.selectors
        assert isinstance(s0, Scalar) and s0.index == 2
        assert isinstance(s1, Affine) and (s1.start, s1.stop, s1.step) == (1, 4, 1)
        assert isinstance(s2, Explicit) and list(s2.coords) == [0, 1]

    def test_negative_scalar_normalized(self):
        """A negative scalar index is normalized to a non-negative one."""
        selection = Selection.from_index((-1,), SHAPE)
        assert selection.selectors[0] == Scalar(SHAPE[0] - 1)

    def test_negative_array_normalized(self):
        """Negative entries in an advanced index are wrapped into range."""
        selection = Selection.from_index((np.array([-1, -2]),), SHAPE)
        assert list(selection.selectors[0].coords) == [SHAPE[0] - 1, SHAPE[0] - 2]

    def test_advanced_at_front_detection(self):
        """Separated advanced axes set ``advanced_at_front``; contiguous don't."""
        sep = Selection.from_index((np.array([0, 1]), 2, np.array([0, 3])), SHAPE)
        cont = Selection.from_index((np.array([0, 1]), np.array([0, 3])), SHAPE)
        assert sep.advanced_at_front is True
        assert cont.advanced_at_front is False

    def test_npoints_and_is_advanced(self):
        sel = Selection.from_index((np.array([[0], [1]]), np.array([0, 2, 4])), SHAPE)
        assert sel.is_advanced is True
        assert sel.advanced_shape == (2, 3)
        assert sel.npoints == 6
        basic = Selection.from_index((slice(None),), SHAPE)
        assert basic.is_advanced is False
        assert basic.npoints == 1

    def test_scalar_out_of_bounds(self):
        with pytest.raises(IndexError):
            Selection.from_index((4,), SHAPE)

    def test_too_many_indices(self):
        with pytest.raises(IndexError):
            Selection.from_index((1, 2, 3, 4), SHAPE)

    def test_newaxis_unsupported(self):
        with pytest.raises(NotImplementedError):
            Selection.from_index((np.newaxis, slice(None)), SHAPE)


class TestResultDims:

    """``result_dims`` is the single source of result-axis ordering."""

    def test_basic_only(self):
        sel = Selection.from_index((2, slice(None), slice(None)), SHAPE)
        assert sel.result_dims == [('basic', 1), ('basic', 2)]

    def test_contiguous_advanced_in_place(self):
        sel = Selection.from_index((np.array([0, 1]), np.array([0, 3]), slice(None)),
                                   SHAPE)
        # advanced block sits where the (contiguous) advanced axes are
        assert sel.result_dims == [('adv', 0), ('basic', 2)]

    def test_separated_advanced_moves_to_front(self):
        sel = Selection.from_index((np.array([0, 1]), slice(None), np.array([0, 3])),
                                   SHAPE)
        assert sel.result_dims == [('adv', 0), ('basic', 1)]

    def test_result_shape_derives_from_dims(self):
        """``result_shape`` is exactly the sizes of ``result_dims``, in order."""
        sel = Selection.from_index((np.array([0, 1]), slice(1, 4), np.array([0, 3])),
                                   SHAPE)
        sizes = tuple(sel.selectors[v].size if kind == 'basic'
                      else sel.advanced_shape[v] for kind, v in sel.result_dims)
        assert sizes == sel.result_shape

    def test_module_and_property_agree(self):
        sel = Selection.from_index((np.array([0, 1]), 2, np.array([0, 3])), SHAPE)
        assert sel.result_dims == result_dims(
            sel.selectors, sel.advanced_axes, sel.advanced_shape,
            sel.advanced_at_front)


class TestIndexHasArray:

    """The cheap gate that keeps basic indexing off the routing path."""

    @pytest.mark.parametrize('idx, ndim, expected', [
        ((slice(None), 2), 2, False),
        (2, 2, False),
        (slice(1, 4), 2, False),
        ((np.array([0, 1]), slice(None)), 2, True),
        (np.array([0, 1]), 1, True),
        (np.array([True, False]), 1, True),
        # legacy ``data[[i, j, k]]`` shorthand on an n-D object stays basic
        ([0, 1, 2], 3, False),
        # a genuine 1-D list index on a 1-D object is advanced
        ([0, 1, 2], 1, True),
    ])
    def test_gate(self, idx, ndim, expected):
        assert index_has_array(idx, ndim) is expected


class TestLayout:

    """Physical placement maps, computed locally from the decomposition."""

    @pytest.fixture
    def decomposition(self):
        # 12 indices over 4 sub-ranks, uneven: [0,1,2] [3,4] [5,6,7] [8..11]
        return Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

    def test_axis_maps(self, decomposition):
        layout = Layout(None, (decomposition, None), (12, 3))
        owner, local, sizes = layout.axis_maps(0)
        assert list(owner) == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        assert list(local) == [0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 3]
        assert list(sizes) == [3, 2, 3, 4]

    def test_distributed_and_replicated_axes(self, decomposition):
        layout = Layout(None, (decomposition, None, decomposition), (12, 3, 12))
        assert layout.distributed_axes == (0, 2)
        assert layout.replicated_axes == (1,)
        assert layout.replicated_size == 3
        assert layout.topology_shape == (4, 4)

    def test_single_axis_coord_to_rank(self, decomposition):
        # Without ``all_coords`` (e.g. a sparse distributor) ranks lay out linearly
        class _Dist:
            nprocs = 4
        layout = Layout(_Dist(), (decomposition,), (12,))
        assert layout.coord_to_rank == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}

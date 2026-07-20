import numpy as np
import pytest

from devito import (  # noqa
    ALLOC_ALIGNED, ALLOC_GUARD, Dimension, Eq, Function, Grid, Operator,
    PrecomputedSparseFunction, PrecomputedSparseTimeFunction, SparseFunction,
    SparseTimeFunction, TimeFunction, configuration, switchconfig
)
from devito.data import LEFT, RIGHT, Decomposition, convert_index, loc_data_idx
from devito.data.allocators import DataReference
from devito.data.distributed.layout import Layout
from devito.data.distributed.selection import (
    Affine, Explicit, IndexScalar, Selection, index_has_array, result_dims
)
from devito.ir import ccode
from devito.tools import as_tuple
from devito.types import Scalar
from devito.types.misc import TempArray


class TestDataBasic:

    def test_simple_indexing(self):
        """Test data packing/unpacking via basic indexing."""
        grid = Grid(shape=(16, 16, 16))
        u = Function(name='yu3D', grid=grid, space_order=0)

        # Test simple insertion and extraction
        u.data[0, 1, 1] = 1.
        assert u.data[0, 0, 0] == 0.
        assert u.data[0, 1, 1] == 1.
        assert np.all(u.data == u.data[:, :, :])
        assert 1. in u.data[0]
        assert 1. in u.data[0, 1]

        # Test negative indices
        assert u.data[0, -15, -15] == 1.
        u.data[6, 0, 0] = 1.
        assert u.data[-10, :, :].sum() == 1.

        # Test setting whole array to given value
        u.data[:] = 3.
        assert np.all(u.data == 3.)

        # Test insertion of single value into block
        u.data[5, :, 5] = 5.
        assert np.all(u.data[5, :, 5] == 5.)

        # Test extraction of block with negative indices
        sliced = u.data[-11, :, -11]
        assert sliced.shape == (16,)
        assert np.all(sliced == 5.)

        # Test insertion of block into block
        block = np.ndarray(shape=(1, 16, 1), dtype=np.float32)
        block.fill(4.)
        u.data[4:5, :, 4:5] = block
        assert np.all(u.data[4, :, 4] == block)

    def test_advanced_indexing(self):
        """Test data packing/unpacking via advanced indexing."""
        grid = Grid(shape=(4, 4, 4))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0, time_order=1)
        u.data[:] = 0.

        # Test slicing w/ negative indices, combined to explicit indexing
        u.data[1, 1:-1, 1:-1, 1:-1] = 6.
        assert np.all(u.data[0] == 0.)
        assert np.all(u.data[1, 1:-1, 1:-1, 1:-1] == 6.)
        assert np.all(u.data[1, :, 0] == 0.)
        assert np.all(u.data[1, :, -1] == 0.)
        assert np.all(u.data[1, :, :, 0] == 0.)
        assert np.all(u.data[1, :, :, -1] == 0.)

    def test_negative_step(self):
        """Test slicing with a negative step."""
        grid = Grid(shape=(6, 6, 6))
        u = TimeFunction(name='u', grid=grid, dtype=np.int32)
        u.data[:] = 0.
        dat = np.array([1, 2, 3, 4, 5, 6])
        u.data[0, :, 0, 0] = dat
        assert (np.array(u.data[0, 3::-1, 0, 0]) == dat[3::-1]).all()
        assert (np.array(u.data[0, 5:1:-1, 0, 0]) == dat[5:1:-1]).all()

    def test_negative_start(self):
        """Test slicing with a negative start."""
        grid = Grid(shape=(13,))
        f = Function(name='f', grid=grid)
        idx = slice(-4, None, 1)
        dat = np.array([1, 2, 3, 4])
        f.data[idx] = dat
        assert np.all(np.array(f.data[9:]) == dat)

    def test_halo_indexing(self):
        """Test data packing/unpacking in presence of a halo region."""
        domain_shape = (16, 16, 16)
        grid = Grid(shape=domain_shape)
        u = Function(name='yu3D', grid=grid, space_order=2)

        assert u.shape == u.data.shape == domain_shape
        assert u._shape_with_inhalo == u.data_with_halo.shape == (20, 20, 20)
        assert u.shape_with_halo == u._shape_with_inhalo  # W/o MPI, these two coincide

        # Test simple insertion and extraction
        u.data_with_halo[0, 0, 0] = 1.
        u.data[0, 0, 0] = 2.
        assert u.data_with_halo[0, 0, 0] == 1.
        assert u.data[0, 0, 0] == 2.
        assert u.data_with_halo[2, 2, 2] == 2.

        # Test negative indices
        u.data_with_halo[-1, -1, -1] = 3.
        assert u.data[-1, -1, -1] == 0.
        assert u.data_with_halo[-1, -1, -1] == 3.

    def test_broadcasting(self):
        """
        Test Data broadcasting, expected to behave as NumPy broadcasting.

        Notes
        -----
        Refer to https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html
        for more info about NumPy broadcasting rules.
        """
        grid = Grid(shape=(4, 4, 4))
        u = Function(name='yu3D', grid=grid)
        u.data[:] = 2.

        # Assign from array with lower-dimensional shape
        v = np.ones(shape=(4, 4), dtype=u.dtype)
        u.data[:] = v
        assert np.all(u.data == 1.)

        # Assign from array with higher-dimensional shape causes a ValueError exception
        v = np.zeros(shape=(4, 4, 4, 4), dtype=u.dtype)
        try:
            u.data[:] = v
        except ValueError:
            assert True
        except Exception as e:
            raise AssertionError('Assert False') from e

        # Assign from array having shape with some 1-valued entries
        v = np.zeros(shape=(4, 1, 4), dtype=u.dtype)
        u.data[:] = v
        assert np.all(u.data == 0.)

    def test_arithmetic(self):
        """Test arithmetic operations involving Data objects."""
        grid = Grid(shape=(16, 16, 16))
        u = Function(name='yu3D', grid=grid, space_order=0)
        u.data[:] = 1

        # Simple arithmetic
        assert np.all(u.data == 1)
        assert np.all(u.data + 2. == 3.)
        assert np.all(u.data - 2. == -1.)
        assert np.all(u.data * 2. == 2.)
        assert np.all(u.data / 2. == 0.5)
        assert np.all(u.data % 2 == 1.)

        # Increments and partial increments
        u.data[:] += 2.
        assert np.all(u.data == 3.)
        u.data[9, :, :] += 1.
        assert all(np.all(u.data[i, :, :] == 3.) for i in range(9))
        assert np.all(u.data[9, :, :] == 4.)

        # Right operations __rOP__
        u.data[:] = 1.
        arr = np.ndarray(shape=(16, 16, 16), dtype=np.float32)
        arr.fill(2.)
        assert np.all(arr - u.data == 1.)

    def test_illegal_indexing(self):
        """Tests that indexing into illegal entries throws an exception."""
        nt = 5
        grid = Grid(shape=(4, 4, 4))
        u = Function(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid, save=nt)

        try:
            u.data[5]
            raise AssertionError('Assert False')
        except IndexError:
            pass
        try:
            v.data[nt]
            raise AssertionError('Assert False')
        except IndexError:
            pass

    def test_logic_indexing(self):
        """Test logic indexing along stepping dimensions."""
        grid = Grid(shape=(4, 4, 4))
        v_mod = TimeFunction(name='v_mod', grid=grid)

        v_mod.data[0] = 1.
        v_mod.data[1] = 2.
        assert np.all(v_mod.data[0] == 1.)
        assert np.all(v_mod.data[1] == 2.)
        assert np.all(v_mod.data[2] == v_mod.data[0])
        assert np.all(v_mod.data[4] == v_mod.data[0])
        assert np.all(v_mod.data[3] == v_mod.data[1])
        assert np.all(v_mod.data[-1] == v_mod.data[1])
        assert np.all(v_mod.data[-2] == v_mod.data[0])

    def test_indexing_into_sparse(self):
        """
        Test indexing into SparseFunctions.
        """
        grid = Grid(shape=(4, 4))
        sf = SparseTimeFunction(name='sf', grid=grid, npoint=1, nt=10)

        sf.data[1:-1, 0] = np.arange(8)
        assert np.all(sf.data[1:-1, 0] == np.arange(8))

    def test_slice_after_transpose(self):
        """
        Slicing a ``Data`` view that has been transposed (via ``.T``,
        ``transpose`` or ``swapaxes``) must use the new axis ordering for
        per-axis metadata. Previously the metadata was copied verbatim from
        the un-transposed array, so a subsequent slice was computed against
        the wrong decomposition and silently returned a wrong-shaped result
        (see issue #2187).
        """
        grid = Grid(shape=(4, 6))
        f = Function(name='f', grid=grid)
        f.data[:] = np.arange(24).reshape((4, 6)).astype(np.float32)
        ref = np.array(f.data)

        # ``.T`` (C-level shortcut) then slice
        assert np.array_equal(f.data.T[::2, ::2], ref.T[::2, ::2])

        # Equivalent: slice then ``.T``
        assert np.array_equal(f.data[::2, ::2].T, ref[::2, ::2].T)

        # Explicit ``transpose`` call -- same behavior as ``.T``
        assert np.array_equal(f.data.transpose()[::2, ::2],
                              ref.transpose()[::2, ::2])

        # ``swapaxes`` between non-conforming dims
        assert np.array_equal(f.data.swapaxes(0, 1)[::2, ::2],
                              ref.swapaxes(0, 1)[::2, ::2])

        # 3D transpose with an explicit axis order, then per-axis slice
        grid3 = Grid(shape=(2, 4, 6))
        g = Function(name='g3', grid=grid3)
        g.data[:] = np.arange(48).reshape((2, 4, 6)).astype(np.float32)
        ref3 = np.array(g.data)

        assert np.array_equal(g.data.T[::2, ::2, ::2], ref3.T[::2, ::2, ::2])
        assert np.array_equal(g.data.transpose((1, 0, 2))[::2, ::1, ::3],
                              ref3.transpose((1, 0, 2))[::2, ::1, ::3])

    def test_transpose_permutes_data_metadata(self):
        """
        After a transpose-like operation, ``_decomposition`` and ``_modulo``
        must be permuted to match the new axis order so that subsequent
        ``__getitem__`` translations use the right per-axis ranges.
        """
        grid = Grid(shape=(4, 6))
        f = Function(name='f', grid=grid)

        original_decomp = f.data._decomposition
        assert len(original_decomp) == 2

        # ``.T`` reverses everything
        tdata = f.data.T
        assert tdata._decomposition == original_decomp[::-1]
        assert tdata._modulo == f.data._modulo[::-1]

        # ``transpose()`` with no args == ``.T``
        tdata2 = f.data.transpose()
        assert tdata2._decomposition == original_decomp[::-1]

        # ``swapaxes`` swaps the two named axes
        sdata = f.data.swapaxes(0, 1)
        assert sdata._decomposition == (original_decomp[1], original_decomp[0])

        # Explicit axis-order
        grid3 = Grid(shape=(2, 4, 6))
        g = Function(name='g3', grid=grid3)
        gdec = g.data._decomposition
        perm = g.data.transpose((1, 2, 0))
        assert perm._decomposition == (gdec[1], gdec[2], gdec[0])
        assert perm._modulo == (g.data._modulo[1], g.data._modulo[2],
                                g.data._modulo[0])

    @pytest.mark.parallel(mode=1)
    def test_indexing_into_sparse_subfunc_singlempi(self, mode):
        grid = Grid(shape=(4, 4))
        s = SparseFunction(name='sf', grid=grid, npoint=1)
        coords = np.random.rand(*s.coordinates.data.shape)
        s.coordinates.data[:] = coords

        s.coordinates.data[-1, :] = s.coordinates.data[-1, :] / 2

        assert np.allclose(s.coordinates.data[-1, :], coords[-1, :] / 2)


class TestLocDataIDX:
    """
    Test the support function loc_data_idx.
    """
    @pytest.mark.parametrize('idx, expected', [
        ('(slice(10, None, -1), slice(11, None, -3))',
         '(slice(0, 11, 1), slice(2, 12, 3))'),
        ('(2, 5)', '(slice(2, 3, 1), slice(5, 6, 1))')
    ])
    def test_loc_data_idx(self, idx, expected):
        """
        Test loc_data_idx located in devito/data/utils.py
        """
        idx = eval(idx)
        expected = eval(expected)

        result = loc_data_idx(idx)

        assert result == expected


class TestMetaData:

    """
    Test correctness of metadata describing size and offset of the various
    data regions, such as DOMAIN, HALO, etc.
    """

    def test_wo_halo_wo_padding(self):
        grid = Grid(shape=(4, 4, 4))
        u = Function(name='u', grid=grid, space_order=0)

        assert u.shape == u._shape_with_inhalo == u.shape_allocated
        assert u.shape_with_halo == u._shape_with_inhalo  # W/o MPI, these two coincide
        assert u._size_halo == u._size_owned == u._size_padding ==\
            ((0, 0), (0, 0), (0, 0))
        assert u._offset_domain == (0, 0, 0)
        assert u._offset_halo == u._offset_owned == ((0, 4), (0, 4), (0, 4))

    def test_w_halo_wo_padding(self):
        grid = Grid(shape=(4, 4, 4))
        u = Function(name='u', grid=grid, space_order=2)

        assert len(u.shape) == len(u._size_halo.left)
        assert u._size_halo == u._size_owned == ((2, 2), (2, 2), (2, 2))
        assert u._offset_domain == (2, 2, 2)
        assert u._offset_halo == ((0, 6), (0, 6), (0, 6))
        assert u._offset_owned == ((2, 4), (2, 4), (2, 4))
        assert tuple(
            i + j*2 for i, j in zip(u.shape, u._size_halo.left, strict=True)
        ) == u.shape_with_halo

        # Try with different grid shape and space_order
        grid2 = Grid(shape=(3, 3, 3))
        u2 = Function(name='u2', grid=grid2, space_order=4)
        assert u2.shape == (3, 3, 3)
        assert u2._offset_domain == (4, 4, 4)
        assert u2._offset_halo == ((0, 7), (0, 7), (0, 7))
        assert tuple(
            i + j*2 for i, j in zip(u2.shape, u2._size_halo.left, strict=True)
        ) == u2.shape_with_halo
        assert u2.shape_with_halo == (11, 11, 11)

    # Platform is used to fix the pad value
    # GPU is disabled to prevent GPU pad value from being used
    @switchconfig(autopadding_mode='cpu-only', autopadding=True, platform='bdw')
    def test_w_halo_w_autopadding(self):
        grid = Grid(shape=(4, 4, 4))
        u0 = Function(name='u0', grid=grid, space_order=0)
        u1 = Function(name='u1', grid=grid, space_order=3)

        mmts = configuration['platform'].max_mem_trans_size(u1.dtype)

        u0_pad = mmts - 4  # RoundUp(4, mmts) - 4
        assert u0._size_halo == ((0, 0), (0, 0), (0, 0))
        assert u0._size_padding == ((0, 0), (0, 0), (0, u0_pad))
        assert u0._size_nodomain == u0._size_padding
        assert u0.shape_allocated == (4, 4, 4 + u0_pad)

        u1_pad = mmts - 10  # RoundUp(3+4+3, mmts) - (3+4+3)
        assert u1._size_halo == ((3, 3), (3, 3), (3, 3))
        assert u1._size_padding == ((0, 0), (0, 0), (0, u1_pad))
        assert u1._size_nodomain == ((3, 3), (3, 3), (3, 3 + u1_pad))
        assert u1.shape_allocated == (10, 10, 10 + u1_pad)

    @switchconfig(autopadding=True, platform='bdw')
    def test_temp_array_smart_padding_no_overshoot(self):
        mmts = configuration['platform'].max_mem_trans_size(np.float32)
        halo = 4
        z_size = 2*mmts - 2*halo

        grid = Grid(shape=(4, 4, z_size))
        u = Function(name='u', grid=grid, space_order=halo)
        r = TempArray(name='r', dimensions=grid.dimensions, halo=u.halo, dtype=u.dtype)

        z = grid.dimensions[-1]
        mapper = {z.symbolic_size: z_size}

        assert r.padding[z][1].subs(mapper) == 0
        assert r.shape_allocated[-1].subs(mapper) == u.shape_allocated[-1]

    @switchconfig(autopadding=True, platform='bdw')
    def test_temp_array_smart_padding_codegen_avoids_negative_mod(self):
        grid = Grid(shape=(4, 4, 592))
        u = Function(name='u', grid=grid, space_order=0)
        r = TempArray(name='r', dimensions=grid.dimensions, halo=u.halo, dtype=u.dtype)

        code = ccode(r.shape_allocated[-1])

        assert 'ROUND_UP(' in code
        assert '(-z_size)' not in code
        assert 'z_size' in code

    def test_w_halo_custom(self):
        grid = Grid(shape=(4, 4))

        # Custom halo with not enougn entries raises an exception
        with pytest.raises(TypeError):
            Function(name='u', grid=grid, space_order=(8, (4, 3)))

        u = TimeFunction(name='u', grid=grid, space_order=(8, ((4, 3), (1, 1))))

        assert u._size_halo == ((0, 0), (4, 3), (1, 1))
        assert u.shape_allocated == (2, 11, 6)


class TestDecomposition:

    """
    Notes
    -----
    If these tests don't work, there is no chance that the tests in TestDataDistributed
    will pass.
    """

    def test_glb_to_loc_index_conversions(self):
        d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        # A global index as single argument
        assert d.index_glb_to_loc(5) == 0
        assert d.index_glb_to_loc(6) == 1
        assert d.index_glb_to_loc(7) == 2
        assert d.index_glb_to_loc(3) is None

        # Retrieve relative local min/man given global min/max
        assert d.index_glb_to_loc((5, 7)) == (0, 2)
        assert d.index_glb_to_loc((5, 9)) == (0, 2)
        assert d.index_glb_to_loc((1, 3)) == (-1, -3)
        assert d.index_glb_to_loc((1, 6)) == (0, 1)
        assert d.index_glb_to_loc((None, None)) == (0, 2)

        # Retrieve absolute local min/man given global min/max
        assert d.index_glb_to_loc((5, 7), rel=False) == (5, 7)
        assert d.index_glb_to_loc((5, 9), rel=False) == (5, 7)
        assert d.index_glb_to_loc((1, 3), rel=False) == (-1, -3)
        assert d.index_glb_to_loc((1, 6), rel=False) == (5, 6)
        assert d.index_glb_to_loc((None, None), rel=False) == (5, 7)

    def test_glb_to_loc_strided_start_in_subdomain(self):
        """
        Regression: a strided slice whose start (or, for a negative step, its
        high end) falls strictly inside the owning subdomain must begin the
        stride phase at the start, not at the subdomain boundary. Previously the
        local slice picked spurious leading/trailing elements (e.g. global
        `2:8:2` gave rank 0 local `[0, 2]` instead of `[2]`).
        """
        d0 = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 0)
        d2 = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        # Positive step, start strictly inside the owning subdomain
        assert d0.index_glb_to_loc(slice(2, 12, 2)) == slice(2, 3, 2)   # -> [2]
        assert d2.index_glb_to_loc(slice(6, 12, 2)) == slice(1, 3, 2)   # -> [6]

        # Negative step, high end (the start) strictly inside the subdomain
        assert d2.index_glb_to_loc(slice(5, 0, -2)) == slice(0, None, -2)   # -> [5]
        assert d0.index_glb_to_loc(slice(2, None, -2)) == slice(2, None, -2)

    def test_glb_to_loc_w_side(self):
        d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        # A global index as single argument
        assert d.index_glb_to_loc(5, LEFT) == 0
        assert d.index_glb_to_loc(6, RIGHT) == 2
        assert d.index_glb_to_loc(7, LEFT) == 2
        assert d.index_glb_to_loc(4, RIGHT) == 0
        assert d.index_glb_to_loc(6, LEFT) == 1
        assert d.index_glb_to_loc(5, RIGHT) == 1
        assert d.index_glb_to_loc(2, LEFT) is None
        assert d.index_glb_to_loc(3, RIGHT) is None

    def test_loc_to_glb_index_conversions(self):
        d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        # Convert local indices to global indices
        assert d.index_loc_to_glb((0, 2)) == (5, 7)

        d2 = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 0)
        assert d2.index_loc_to_glb((0, 2)) == (0, 2)
        d3 = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 3)
        assert d3.index_loc_to_glb((1, 3)) == (9, 11)

    def test_convert_index(self):
        d0 = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)
        d1 = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 3)
        decomposition = (d0, d1)

        idx0 = (5, slice(8, 11, 1))
        result0 = []
        for i, j in zip(idx0, decomposition, strict=True):
            result0.append(convert_index(i, j))
        expected0 = (0, slice(0, 3, 1))
        assert as_tuple(result0) == expected0

    def test_reshape_identity(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Identity decomposition
        assert len(d.reshape(0, 0)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(0, 0), [[0, 1], [2, 3]], strict=True)
        )

    def test_reshape_right_only(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Extension at right only
        assert len(d.reshape(0, 2)) == 2
        assert all(
            list(i) == j for i, j in zip(
                d.reshape(0, 2), [[0, 1], [2, 3, 4, 5]], strict=True
            )
        )
        # Reduction at right affecting one sub-domain only, but not the whole subdomain
        assert len(d.reshape(0, -1)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(0, -1), [[0, 1], [2]], strict=True)
        )
        # Reduction at right over one whole sub-domain
        assert len(d.reshape(0, -2)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(0, -2), [[0, 1], []], strict=True)
        )
        # Reduction at right over multiple sub-domains
        assert len(d.reshape(0, -3)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(0, -3), [[0], []], strict=True)
        )

    def test_reshape_left_only(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Extension at left only
        assert len(d.reshape(2, 0)) == 2
        assert all(
            list(i) == j for i, j in zip(
                d.reshape(2, 0), [[0, 1, 2, 3], [4, 5]], strict=True
            )
        )
        # Reduction at left affecting one sub-domain only, but not the whole subdomain
        assert len(d.reshape(-1, 0)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-1, 0), [[0], [1, 2]], strict=True)
        )
        # Reduction at left over one whole sub-domain
        assert len(d.reshape(-2, 0)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-2, 0), [[], [0, 1]], strict=True)
        )
        # Reduction at right over multiple sub-domains
        assert len(d.reshape(-3, 0)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-3, 0), [[], [0]], strict=True)
        )

    def test_reshape_left_right(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Extension at both left and right
        assert len(d.reshape(1, 1)) == 2
        assert all(
            list(i) == j for i, j in zip(
                d.reshape(1, 1), [[0, 1, 2], [3, 4, 5]], strict=True
            )
        )
        # Reduction at both left and right
        assert len(d.reshape(-1, -1)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-1, -1), [[0], [1]], strict=True)
        )
        # Reduction at both left and right, with the right one obliterating one subdomain
        assert len(d.reshape(-1, -2)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-1, -2), [[0], []], strict=True)
        )
        # Reduction at both left and right obliterating all subdomains
        # triggering an exception
        assert len(d.reshape(-1, -3)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-1, -3), [[], []], strict=True)
        )
        assert len(d.reshape(-2, -2)) == 2
        assert all(
            list(i) == j for i, j in zip(d.reshape(-1, -3), [[], []], strict=True)
        )

    def test_reshape_slice(self):
        d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        assert d.reshape(slice(None)) == d
        assert d.reshape(slice(2, 9)) == Decomposition([[0], [1, 2], [3, 4, 5], [6]], 2)
        assert d.reshape(slice(3, 5)) == Decomposition([[], [0, 1], [], []], 2)
        assert d.reshape(slice(3, 3)) == Decomposition([[], [], [], []], 2)
        assert d.reshape(slice(13, 13)) == Decomposition([[], [], [], []], 2)
        assert d.reshape(slice(2, None)) == Decomposition([[0], [1, 2], [3, 4, 5],
                                                           [6, 7, 8, 9]], 2)
        assert d.reshape(slice(4)) == Decomposition([[0, 1, 2], [3], [], []], 2)
        assert d.reshape(slice(-2, 2)) == Decomposition([[0, 1, 2, 3], [], [], []], 2)
        assert d.reshape(slice(-2)) == Decomposition([[0, 1, 2], [3, 4], [5, 6, 7],
                                                      [8, 9]], 2)
        assert d.reshape(slice(3, -1)) == Decomposition([[], [0, 1], [2, 3, 4],
                                                         [5, 6, 7]], 2)

    def test_reshape_iterable(self):
        d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        assert d.reshape(()) == Decomposition([[], [], [], []], 2)
        assert d.reshape((1, 3, 5)) == Decomposition([[0], [1], [2], []], 2)
        assert d.reshape((1, 3, 10, 11)) == Decomposition([[0], [1], [], [2, 3]], 2)
        assert d.reshape((1, 3, 10, 11, 14)) == Decomposition([[0], [1], [], [2, 3]], 2)


# A reference global shape for the serial engine tests below; indices are
# validated against the NumPy result on an array of this shape.
ENGINE_SHAPE = (4, 5, 6)


class TestSelection:

    """`Selection` encodes NumPy indexing semantics, layout-independent."""

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
        ref = np.empty(ENGINE_SHAPE)
        selection = Selection.from_index(idx, ENGINE_SHAPE)
        assert selection.result_shape == ref[idx].shape

    def test_selector_types(self):
        """Each axis is classified as Scalar / Affine / Explicit as expected."""
        selection = Selection.from_index((2, slice(1, 4), np.array([0, 1])),
                                         ENGINE_SHAPE)
        s0, s1, s2 = selection.selectors
        assert isinstance(s0, IndexScalar) and s0.index == 2
        assert isinstance(s1, Affine) and (s1.start, s1.stop, s1.step) == (1, 4, 1)
        assert isinstance(s2, Explicit) and list(s2.coords) == [0, 1]

    def test_negative_scalar_normalized(self):
        """A negative scalar index is normalized to a non-negative one."""
        selection = Selection.from_index((-1,), ENGINE_SHAPE)
        assert selection.selectors[0] == IndexScalar(ENGINE_SHAPE[0] - 1)

    def test_negative_array_normalized(self):
        """Negative entries in an advanced index are wrapped into range."""
        selection = Selection.from_index((np.array([-1, -2]),), ENGINE_SHAPE)
        assert list(selection.selectors[0].coords) == [ENGINE_SHAPE[0] - 1,
                                                       ENGINE_SHAPE[0] - 2]

    def test_advanced_at_front_detection(self):
        """Separated advanced axes set `advanced_at_front`; contiguous don't."""
        sep = Selection.from_index((np.array([0, 1]), 2, np.array([0, 3])),
                                   ENGINE_SHAPE)
        cont = Selection.from_index((np.array([0, 1]), np.array([0, 3])),
                                    ENGINE_SHAPE)
        assert sep.advanced_at_front is True
        assert cont.advanced_at_front is False

    def test_npoints_and_is_advanced(self):
        sel = Selection.from_index((np.array([[0], [1]]), np.array([0, 2, 4])),
                                   ENGINE_SHAPE)
        assert sel.is_advanced is True
        assert sel.advanced_shape == (2, 3)
        assert sel.npoints == 6
        basic = Selection.from_index((slice(None),), ENGINE_SHAPE)
        assert basic.is_advanced is False
        assert basic.npoints == 1

    def test_scalar_out_of_bounds(self):
        with pytest.raises(IndexError):
            Selection.from_index((4,), ENGINE_SHAPE)

    def test_too_many_indices(self):
        with pytest.raises(IndexError):
            Selection.from_index((1, 2, 3, 4), ENGINE_SHAPE)

    def test_newaxis_unsupported(self):
        with pytest.raises(NotImplementedError):
            Selection.from_index((np.newaxis, slice(None)), ENGINE_SHAPE)


class TestResultDims:

    """`result_dims` is the single source of result-axis ordering."""

    def test_basic_only(self):
        sel = Selection.from_index((2, slice(None), slice(None)), ENGINE_SHAPE)
        assert sel.result_dims == [('basic', 1), ('basic', 2)]

    def test_contiguous_advanced_in_place(self):
        sel = Selection.from_index((np.array([0, 1]), np.array([0, 3]), slice(None)),
                                   ENGINE_SHAPE)
        # advanced block sits where the (contiguous) advanced axes are
        assert sel.result_dims == [('adv', 0), ('basic', 2)]

    def test_separated_advanced_moves_to_front(self):
        sel = Selection.from_index((np.array([0, 1]), slice(None), np.array([0, 3])),
                                   ENGINE_SHAPE)
        assert sel.result_dims == [('adv', 0), ('basic', 1)]

    def test_result_shape_derives_from_dims(self):
        """`result_shape` is exactly the sizes of `result_dims`, in order."""
        sel = Selection.from_index((np.array([0, 1]), slice(1, 4), np.array([0, 3])),
                                   ENGINE_SHAPE)
        sizes = tuple(sel.selectors[v].size if kind == 'basic'
                      else sel.advanced_shape[v] for kind, v in sel.result_dims)
        assert sizes == sel.result_shape

    def test_module_and_property_agree(self):
        sel = Selection.from_index((np.array([0, 1]), 2, np.array([0, 3])),
                                   ENGINE_SHAPE)
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
        # legacy `data[[i, j, k]]` shorthand on an n-D object stays basic
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
        # Without `all_coords` (e.g. a sparse distributor) ranks lay out linearly
        class _Dist:
            nprocs = 4
        layout = Layout(_Dist(), (decomposition,), (12,))
        assert layout.coord_to_rank == {(0,): 0, (1,): 1, (2,): 2, (3,): 3}


class TestDataDistributed:

    """
    Test Data indexing and manipulation when distributed over a set of MPI processes.
    """

    @staticmethod
    def _assert_induced(res, global_result):
        """
        Indexing is local: each rank holds exactly the elements of the global
        numpy result that it already owns (the induced decomposition), with no
        communication. Verify by reassembling every rank's block, at the result
        indices it owns, into the full global result.
        """
        res_arr = np.array(res)
        owned = []
        empty = res_arr.size == 0
        if not empty:
            for dec, n in zip(res._decomposition, global_result.shape, strict=True):
                if dec is None:
                    owned.append(np.arange(n, dtype=np.int64))
                else:
                    owned.append(np.asarray(dec.loc_abs_numb, dtype=np.int64)
                                 - (dec.glb_min or 0))

        comm = res._distributor.comm
        gathered = comm.gather((None if empty else res_arr, owned), root=0)
        if comm.Get_rank() == 0:
            out = np.full(global_result.shape, -10**9, dtype=global_result.dtype)
            for blk, idxs in gathered:
                if blk is not None and blk.size:
                    out[np.ix_(*idxs)] = blk
            assert np.array_equal(out, global_result)

    @pytest.mark.parallel(mode=4)
    def test_localviews(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        myrank = grid.distributor.myrank
        u = Function(name='u', grid=grid)

        u.data[:] = grid.distributor.myrank

        assert u.data_ro_domain._local[0, 0] == grid.distributor.myrank
        assert u.data_ro_domain._local[1, 1] == grid.distributor.myrank
        assert u.data_ro_domain._local[-1, -1] == grid.distributor.myrank

        assert u.data_ro_with_halo._local[1, 1] == grid.distributor.myrank
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_with_halo._local[1:, 1:] == myrank)
            assert np.all(u.data_ro_with_halo._local[0] == 0.)
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_with_halo._local[1:3, :2] == myrank)
            assert np.all(u.data_ro_with_halo._local[0] == 0.)
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_with_halo._local[:2, 1:3] == myrank)
            assert np.all(u.data_ro_with_halo._local[2] == 0.)
        else:
            assert np.all(u.data_ro_with_halo._local[:2, :2] == myrank)
            assert np.all(u.data_ro_with_halo._local[2] == 0.)

    @pytest.mark.parallel(mode=4)
    def test_trivial_insertion(self, mode):
        grid = Grid(shape=(4, 4))
        u = Function(name='u', grid=grid, space_order=0)
        v = Function(name='v', grid=grid, space_order=1)

        u.data[:] = 1.
        assert np.all(u.data == 1.)
        assert np.all(u.data._local == 1.)

        u.data_local[:] = 2.
        assert np.all(u.data == 2.)
        assert np.all(u.data_local == 2.)

        v.data_with_halo[:] = 1.
        assert v.data_with_halo[:].shape == (3, 3)
        assert np.all(v.data_with_halo == 1.)
        assert np.all(v.data_with_halo[:] == 1.)
        assert np.all(v.data_with_halo._local == 1.)

    @pytest.mark.parallel(mode=4)
    def test_local_indices_roundtrip(self, mode):
        """
        The public `local_indices` slices map a rank's local data into the
        global array, enabling the natural no-comm idiom::

            f.data_local[:] = global_array[f.local_indices]
        """
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)
        nt = 3
        g = TimeFunction(name='g', grid=grid, save=nt, time_order=1,
                         space_order=0, dtype=np.int32)

        global_f = np.arange(8, dtype=f.dtype)
        global_g = np.arange(nt*8, dtype=g.dtype).reshape(nt, 8)

        # Slice the (replicated) global arrays down to this rank's block
        f.data_local[:] = global_f[f.local_indices]
        g.data_local[:] = global_g[g.local_indices]      # time axis -> full slice

        assert np.all(f.data == global_f[f.local_indices])
        assert np.all(g.data == global_g[g.local_indices])

    @pytest.mark.parallel(mode=4)
    def test_advanced_indexing_set_get_1d(self, mode):
        """Each rank labels rank-local values with arbitrary global indices."""
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)

        global_data = np.arange(8, dtype=f.dtype)
        rank = grid.distributor.myrank
        source_index = np.array_split(np.arange(8)[::-1],
                                      grid.distributor.nprocs)[rank]
        source_data = global_data[source_index]

        f.data[:] = 0
        f.data[source_index] = source_data

        assert np.all(f.data_local == global_data[f.local_indices])
        assert np.all(f.data[source_index] == source_data)

    @pytest.mark.parallel(mode=4)
    def test_advanced_indexing_negative_and_empty(self, mode):
        """Negative indices normalize; ranks contributing nothing are a no-op."""
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)

        rank = grid.distributor.myrank
        index = [-1, 0] if rank == 0 else []
        values = np.array([70, 10], dtype=f.dtype) if rank == 0 else \
            np.empty(0, dtype=f.dtype)

        f.data[:] = 0
        f.data[index] = values

        expected = np.zeros(8, dtype=f.dtype)
        expected[[-1, 0]] = [70, 10]
        assert np.all(f.data_local == expected[f.local_indices])
        assert np.all(f.data[index] == values)

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_advanced_indexing_with_time_window(self, mode, order):
        """A replicated (time) axis rides along as the exchanged payload."""
        grid = Grid(shape=(8,))
        nt = 3
        f = TimeFunction(name='f', grid=grid, save=nt, time_order=1,
                         space_order=0, dtype=np.int32)

        global_data = np.arange(nt*8, dtype=f.dtype).reshape(nt, 8)
        rank = grid.distributor.myrank
        source_index = np.array_split(np.arange(8)[::-1],
                                      grid.distributor.nprocs)[rank]
        time_window = slice(1, None)
        source_data = np.array(global_data[time_window, source_index], order=order)

        f.data[:] = 0
        f.data[time_window, source_index] = source_data

        expected = np.zeros_like(global_data)
        expected[time_window, :] = global_data[time_window, :]
        assert np.all(f.data == expected[f.local_indices])
        assert np.all(f.data[time_window, source_index] == source_data)

    @pytest.mark.parallel(mode=4)
    def test_advanced_indexing_two_distributed_dims(self, mode):
        """
        Scatter over two distributed dimensions at once -- the case the previous
        implementation rejected with NotImplementedError.
        """
        grid = Grid(shape=(4, 4))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)
        global_data = np.arange(16, dtype=f.dtype).reshape(4, 4)

        rank = grid.distributor.myrank
        points = {0: ([0, 3], [0, 3]), 1: ([1, 2], [3, 0]),
                  2: ([3, 0], [1, 2]), 3: ([2, 1], [2, 1])}
        xs, ys = (np.array(p) for p in points[rank])
        values = global_data[xs, ys]

        f.data[:] = -1
        f.data[xs, ys] = values
        assert np.all(f.data[xs, ys] == values)

        # Every assigned global point now holds its global value
        gathered = np.array(f.data_gather())
        if rank == 0:
            for r in range(grid.distributor.nprocs):
                rx, ry = (np.array(p) for p in points[r])
                assert np.all(gathered[rx, ry] == global_data[rx, ry])

    @pytest.mark.parallel(mode=4)
    def test_advanced_indexing_errors(self, mode):
        """Duplicate and out-of-bounds indices raise consistently on all ranks."""
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)

        rank = grid.distributor.myrank
        duplicate_index = np.array([1, 1]) if rank == 0 else \
            np.empty(0, dtype=np.int64)

        with pytest.raises(ValueError, match="rank 0:.*Duplicate global indices"):
            f.data[duplicate_index] = np.ones(duplicate_index.size, dtype=f.dtype)

        oob_index = np.array([8]) if rank == 0 else np.empty(0, dtype=np.int64)
        with pytest.raises(ValueError, match="rank 0:.*out-of-bounds"):
            f.data[oob_index] = np.ones(oob_index.size, dtype=f.dtype)
        with pytest.raises(ValueError, match="rank 0:.*out-of-bounds"):
            f.data[oob_index]

    @pytest.mark.parallel(mode=4)
    def test_advanced_indexing_local_only_no_comm(self, mode):
        """
        Case 1: each rank labels rank-local values with its own global indices, so
        nothing crosses ranks. `b` matches the local size; data stays put. The
        same effect is achievable comm-free via `data_local` + `local_indices`.
        """
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)

        rank = grid.distributor.myrank
        # This rank's own global indices, as an explicit array
        local_index = np.arange(*f.local_indices[0].indices(grid.shape[0]))
        b = np.arange(local_index.size, dtype=f.dtype) + 10*rank

        f.data[:] = 0
        f.data[local_index] = b              # routed, but resolves to self only
        assert np.all(f.data_local == b)
        assert np.all(f.data[local_index] == b)

        # The idiomatic comm-free equivalent
        g = Function(name='g', grid=grid, space_order=0, dtype=np.int32)
        g.data_local[:] = b
        assert np.all(g.data == f.data)

    @pytest.mark.parallel(mode=4)
    def test_advanced_indexing_generic_size_crossing_ranks(self, mode):
        """
        Case 2: a single rank assigns an arbitrary number of values to global
        indices spread across every rank; `len(b)` differs from the local size.
        """
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)

        rank = grid.distributor.myrank
        index = np.array([7, 5, 2, 0]) if rank == 0 else np.empty(0, dtype=np.int64)
        values = np.array([77, 55, 22, 0], dtype=f.dtype) if rank == 0 else \
            np.empty(0, dtype=f.dtype)

        f.data[:] = -1
        f.data[index] = values

        expected = np.full(8, -1, dtype=f.dtype)
        expected[[7, 5, 2, 0]] = [77, 55, 22, 0]
        assert np.all(f.data_local == expected[f.local_indices])
        assert np.all(f.data[index] == values)

    @pytest.mark.parallel(mode=4)
    def test_full_assignment_replicated_rhs(self, mode):
        """
        Case 3: `a.data[:] = b` with `b` the full global array replicated on
        every rank. Basic indexing, so each rank slices its own piece (no comm).
        """
        grid = Grid(shape=(8,))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)

        b = np.arange(8, dtype=f.dtype)   # identical (replicated) on all ranks
        f.data[:] = b

        assert np.all(f.data_local == b[f.local_indices])
        assert np.all(f.data[:]._local == b[f.local_indices])

    @pytest.mark.parallel(mode=4)
    def test_indexing(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        myrank = grid.distributor.myrank
        u = Function(name='u', grid=grid, space_order=0)

        u.data[:] = myrank

        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert u.data[0, 0] == myrank
            assert u.data[2, 2] is None
            assert u.data[2].size == 0
            assert u.data[:, 2].size == 0
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert u.data[0, 0] is None
            assert u.data[2, 2] is None
            assert u.data[2].size == 0
            assert np.all(u.data[:, 2] == [myrank, myrank])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert u.data[0, 0] is None
            assert u.data[2, 2] is None
            assert np.all(u.data[2] == [myrank, myrank])
            assert u.data[:, 2].size == 0
        else:
            assert u.data[0, 0] is None
            assert u.data[2, 2] == myrank
            assert np.all(u.data[2] == [myrank, myrank])
            assert np.all(u.data[:, 2] == [myrank, myrank])

    @pytest.mark.parallel(mode=4)
    def test_slicing(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        myrank = grid.distributor.myrank
        u = Function(name='u', grid=grid, space_order=0)

        u.data[:] = myrank

        # `u.data` is a view of the global data array restricted, on each rank,
        # to the local rank domain, so it must be == myrank
        assert np.all(u.data == myrank)
        assert np.all(u.data._local == myrank)
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data[:2, :2] == myrank)
            assert u.data[:2, 2:].size == u.data[2:, :2].size == u.data[2:, 2:].size == 0
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data[:2, 2:] == myrank)
            assert u.data[:2, :2].size == u.data[2:, :2].size == u.data[2:, 2:].size == 0
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data[2:, :2] == myrank)
            assert u.data[:2, 2:].size == u.data[:2, :2].size == u.data[2:, 2:].size == 0
        else:
            assert np.all(u.data[2:, 2:] == myrank)
            assert u.data[:2, 2:].size == u.data[2:, :2].size == u.data[:2, :2].size == 0

    @pytest.mark.parallel(mode=4)
    def test_slicing_ns(self, mode):
        # Test slicing with a negative step
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        myrank = grid.distributor.myrank
        u = Function(name='u', grid=grid, space_order=0)

        u.data[:] = myrank

        dat = np.arange(16, dtype=np.int32)
        dat = dat.reshape(grid.shape)

        u.data[::-1, ::-1] = dat

        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data == [[15, 14], [11, 10]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data == [[13, 12], [9, 8]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data == [[7, 6], [3, 2]])
        else:
            assert np.all(u.data == [[5, 4], [1, 0]])

    @pytest.mark.parallel(mode=4)
    def test_getitem(self, mode):
        # __getitem__ MPI slicing: indexing is local, never communicates --
        # each rank returns exactly the part of the global result it owns.
        grid = Grid(shape=(8, 8))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)
        a = np.arange(64, dtype=np.int32).reshape(grid.shape)
        f.data[:] = a

        for gslice in [(slice(None, None, -1), slice(None, None, -1)),
                       (5, slice(6, 1, -1)),
                       (slice(6, 4, -1), slice(6, 1, -1)),
                       (slice(6, 4, -1), slice(2, 7)),
                       (slice(4, 2, -1), slice(6, 1, -1))]:
            self._assert_induced(f.data[gslice], a[gslice])

    @pytest.mark.parallel(mode=4)
    def test_big_steps(self, mode):
        # Slicing with |step| > 1 (positive and negative) is local: each rank
        # returns the part of the global strided result it owns.
        grid = Grid(shape=(8, 8))
        f = Function(name='f', grid=grid, space_order=0, dtype=np.int32)
        a = np.arange(64, dtype=np.int32).reshape(grid.shape)
        f.data[:] = a

        for gslice in [(slice(None, None, 3), slice(None, None, 3)),
                       (slice(1, None, 3), slice(1, None, 3)),
                       (slice(None, None, -3), slice(None, None, -3)),
                       (slice(6, None, -3), slice(6, None, -3))]:
            self._assert_induced(f.data[gslice], a[gslice])

    @pytest.mark.parallel(mode=4)
    def test_setitem(self, mode):
        # __setitem__ mpi slicing tests
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        g = Function(name='g', grid=grid, space_order=0, dtype=np.int32)
        h = Function(name='h', grid=grid, space_order=0, dtype=np.int32)

        grid1 = Grid(shape=(8, 8))
        f = Function(name='f', grid=grid1, space_order=0, dtype=np.int32)
        test_dat = np.arange(64, dtype=np.int32)
        a = test_dat.reshape(grid1.shape)

        f.data[:] = a

        g.data[0, 0:3] = f.data[7, 4:7]
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(np.array(g.data) == [[60, 61, 62, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y] \
                or RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(np.array(g.data)) == 0
        else:
            assert np.all(np.array(g.data)) == 0

        h.data[2:10, 2:10] = f.data[::-1, ::-1]
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(np.array(h.data) == [[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 63, 62, 61, 60],
                                               [0, 0, 55, 54, 53, 52],
                                               [0, 0, 47, 46, 45, 44],
                                               [0, 0, 39, 38, 37, 36]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(np.array(h.data) == [[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [59, 58, 57, 56, 0, 0],
                                               [51, 50, 49, 48, 0, 0],
                                               [43, 42, 41, 40, 0, 0],
                                               [35, 34, 33, 32, 0, 0]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(np.array(h.data) == [[0, 0, 31, 30, 29, 28],
                                               [0, 0, 23, 22, 21, 20],
                                               [0, 0, 15, 14, 13, 12],
                                               [0, 0, 7, 6, 5, 4],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]])
        else:
            assert np.all(np.array(h.data) == [[27, 26, 25, 24, 0, 0],
                                               [19, 18, 17, 16, 0, 0],
                                               [11, 10, 9, 8, 0, 0],
                                               [3, 2, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]])

    @pytest.mark.parallel(mode=4)
    def test_hd_slicing(self, mode):
        # Higher-dimensional negative-step slicing is local (induced).
        grid = Grid(shape=(4, 4, 4))
        t = Function(name='t', grid=grid, space_order=0)
        b = np.arange(64, dtype=np.int32).reshape(grid.shape)
        t.data[:] = b

        for gslice in [(slice(None, None, -1),)*3,
                       (slice(None, None, -1),),
                       (slice(None), slice(3, 2, -1))]:
            self._assert_induced(t.data[gslice], b[gslice])

    @pytest.mark.parallel(mode=4)
    def test_niche_slicing(self, mode):
        grid0 = Grid(shape=(8, 8))
        x0, y0 = grid0.dimensions
        glb_pos_map0 = grid0.distributor.glb_pos_map
        f = Function(name='f', grid=grid0, space_order=0, dtype=np.int32)
        dat = np.arange(64, dtype=np.int32)
        a = dat.reshape(grid0.shape)
        f.data[:] = a

        grid1 = Grid(shape=(12, 12))
        x1, y1 = grid1.dimensions
        glb_pos_map1 = grid1.distributor.glb_pos_map
        h = Function(name='h', grid=grid1, space_order=0, dtype=np.int32)

        grid2 = Grid(shape=(4, 4, 4))
        t = Function(name='t', grid=grid2, space_order=0)
        b = dat.reshape(grid2.shape)
        t.data[:] = b

        tdat0 = np.array(f.data[-2::, -2::])
        if LEFT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0] \
                or LEFT in glb_pos_map0[x0] and RIGHT in glb_pos_map0[y0] \
                or RIGHT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert tdat0.size == 0
        else:
            assert np.all(tdat0 == [[54, 55],
                                    [62, 63]])

        h.data[9:1:-1, 9:1:-1] = f.data[:, :]
        if LEFT in glb_pos_map1[x1] and LEFT in glb_pos_map1[y1]:
            assert np.all(np.array(h.data) == [[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 63, 62, 61, 60],
                                               [0, 0, 55, 54, 53, 52],
                                               [0, 0, 47, 46, 45, 44],
                                               [0, 0, 39, 38, 37, 36]])
        elif LEFT in glb_pos_map1[x1] and RIGHT in glb_pos_map1[y1]:
            assert np.all(np.array(h.data) == [[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [59, 58, 57, 56, 0, 0],
                                               [51, 50, 49, 48, 0, 0],
                                               [43, 42, 41, 40, 0, 0],
                                               [35, 34, 33, 32, 0, 0]])
        elif RIGHT in glb_pos_map1[x1] and LEFT in glb_pos_map1[y1]:
            assert np.all(np.array(h.data) == [[0, 0, 31, 30, 29, 28],
                                               [0, 0, 23, 22, 21, 20],
                                               [0, 0, 15, 14, 13, 12],
                                               [0, 0, 7, 6, 5, 4],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]])
        else:
            assert np.all(np.array(h.data) == [[27, 26, 25, 24, 0, 0],
                                               [19, 18, 17, 16, 0, 0],
                                               [11, 10, 9, 8, 0, 0],
                                               [3, 2, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]])

        f.data[:] = 0
        f.data[::2, ::2] = t.data[:, :, 0]
        if LEFT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[0, 0, 4, 0],
                                               [0, 0, 0, 0],
                                               [16, 0, 20, 0],
                                               [0, 0, 0, 0]])
        elif LEFT in glb_pos_map0[x0] and RIGHT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[8, 0, 12, 0],
                                               [0, 0, 0, 0],
                                               [24, 0, 28, 0],
                                               [0, 0, 0, 0]])
        elif RIGHT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[32, 0, 36, 0],
                                               [0, 0, 0, 0],
                                               [48, 0, 52, 0],
                                               [0, 0, 0, 0]])
        else:
            assert np.all(np.array(f.data) == [[40, 0, 44, 0],
                                               [0, 0, 0, 0],
                                               [56, 0, 60, 0],
                                               [0, 0, 0, 0]])

        f.data[:] = 0
        f.data[1::2, 1::2] = t.data[:, :, 0]
        if LEFT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[0, 0, 0, 0],
                                               [0, 0, 0, 4],
                                               [0, 0, 0, 0],
                                               [0, 16, 0, 20]])
        elif LEFT in glb_pos_map0[x0] and RIGHT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[0, 0, 0, 0],
                                               [0, 8, 0, 12],
                                               [0, 0, 0, 0],
                                               [0, 24, 0, 28]])
        elif RIGHT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[0, 0, 0, 0],
                                               [0, 32, 0, 36],
                                               [0, 0, 0, 0],
                                               [0, 48, 0, 52]])
        else:
            assert np.all(np.array(f.data) == [[0, 0, 0, 0],
                                               [0, 40, 0, 44],
                                               [0, 0, 0, 0],
                                               [0, 56, 0, 60]])

        f.data[:] = 0
        f.data[6::-2, 6::-2] = t.data[:, :, 0]
        if LEFT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[60, 0, 56, 0],
                                               [0, 0, 0, 0],
                                               [44, 0, 40, 0],
                                               [0, 0, 0, 0]])
        elif LEFT in glb_pos_map0[x0] and RIGHT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[52, 0, 48, 0],
                                               [0, 0, 0, 0],
                                               [36, 0, 32, 0],
                                               [0, 0, 0, 0]])
        elif RIGHT in glb_pos_map0[x0] and LEFT in glb_pos_map0[y0]:
            assert np.all(np.array(f.data) == [[28, 0, 24, 0],
                                               [0, 0, 0, 0],
                                               [12, 0, 8, 0],
                                               [0, 0, 0, 0]])
        else:
            assert np.all(np.array(f.data) == [[20, 0, 16, 0],
                                               [0, 0, 0, 0],
                                               [4, 0, 0, 0],
                                               [0, 0, 0, 0]])

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('shape, slice0, slice1, slice2', [
        ((31, 31, 31), (slice(None, None, 1), 2, slice(None, None, 1)),
         (slice(None, None, 1), 0, slice(None, None, 1)),
         (slice(None, None, 1), 1, slice(None, None, 1))),
        ((17, 17, 17), (slice(None, None, 1), slice(None, None, 1), 2),
         (slice(None, None, 1), slice(None, None, 1), 0),
         (slice(None, None, 1), slice(None, None, 1), 1)),
        ((8, 8, 8), (slice(None, None, 1), 5, slice(None, None, 1)),
         (slice(None, None, 1), 1, slice(None, None, 1)),
         (slice(None, None, 1), 7, slice(None, None, 1)))])
    def test_niche_slicing2(self, shape, slice0, slice1, slice2, mode):
        grid = Grid(shape=shape)
        f = Function(name='f', grid=grid)
        f.data[:] = 1

        f.data[slice0] = f.data[slice1]
        f.data[slice0] += f.data[slice2]

        result0 = np.array(f.data[slice0])
        expected0 = np.full(result0.shape, 2)
        assert(np.all(result0 == expected0))
        result1 = np.array(f.data[slice1])
        expected1 = np.full(result1.shape, 1)
        assert(np.all(result1 == expected1))
        result2 = np.array(f.data[slice2])
        expected2 = np.full(result2.shape, 1)
        assert(np.all(result2 == expected2))

    def test_empty_slicing(self):
        grid = Grid(shape=(2, 2), extent=(1, 1))
        f = Function(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)
        assert(f.data[0:0].shape == (0, 2))
        assert(f.data[0:0, 0:0].shape == (0, 0))
        assert(g.data[0:0].shape == (0, 2, 2))
        assert(g.data[0:0, 0:0].shape == (0, 0, 2))
        assert(g.data[0:0, 0:0, 0:0].shape == (0, 0, 0))
        assert(f.data[1:1].shape == (0, 2))
        assert(f.data[0:0, 1:1].shape == (0, 0))
        assert(g.data[1:1].shape == (0, 2, 2))
        assert(g.data[1:1, 0:0].shape == (0, 0, 2))
        assert(g.data[1:1, 0:0, 1:1].shape == (0, 0, 0))

    @pytest.mark.parallel(mode=4)
    def test_neg_start_stop(self, mode):
        grid0 = Grid(shape=(8, 8))
        f = Function(name='f', grid=grid0, space_order=0, dtype=np.int32)
        dat = np.arange(64, dtype=np.int32)
        a = dat.reshape(grid0.shape)
        f.data[:] = a

        grid1 = Grid(shape=(12, 12))
        x, y = grid1.dimensions
        glb_pos_map = grid1.distributor.glb_pos_map
        h = Function(name='h', grid=grid1, space_order=0, dtype=np.int32)

        slices = (slice(-3, -1, 1), slice(-1, -5, -1))

        h.data[8:10, 0:4] = f.data[slices]

        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y] \
                or LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.count_nonzero(h.data[:]) == 0
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(np.array(h.data) == [[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [47, 46, 45, 44, 0, 0],
                                               [55, 54, 53, 52, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0]])
        else:
            assert np.count_nonzero(h.data[:]) == 0

    @pytest.mark.parallel(mode=4)
    def test_indexing_in_views(self, mode):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        myrank = grid.distributor.myrank
        u = Function(name='u', grid=grid, space_order=0)

        u.data[:] = myrank

        # Note that the `1`s are global indices
        view = u.data[1:, 1:]
        assert np.all(view[:] == myrank)
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert view.shape == (1, 1)
            assert np.all(view == 0.)
            assert view[0, 0] == 0.
            assert view[1, 1] is None
            assert view[1].shape == (0, 1)
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert view.shape == (1, 2)
            assert np.all(view == 1.)
            assert view[0, 0] is None
            assert view[1, 1] is None
            assert view[1].shape == (0, 2)
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert view.shape == (2, 1)
            assert np.all(view == 2.)
            assert view[0, 0] is None
            assert view[1, 1] is None
            assert view[1].shape == (1,)
            assert np.all(view[1] == 2.)
        else:
            assert view.shape == (2, 2)
            assert np.all(view == 3.)
            assert view[0, 0] is None
            assert view[1, 1] == 3.
            assert view[1].shape == (2,)
            assert np.all(view[1] == 3.)

        # Now we further slice into `view`
        view2 = view[1:, 1:]
        assert np.all(view2[:] == myrank)
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert view2.shape == (0, 0)
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert view2.shape == (0, 2)
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert view2.shape == (2, 0)
        else:
            assert view2.shape == (2, 2)

        # Now a change in `view2` by the only rank that "sees" it should affect
        # both `view` and `u.data`
        view2[:] += 1
        if RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data[:] == myrank + 1)
            assert np.all(view[:] == myrank + 1)
            assert np.all(view2[:] == myrank + 1)
        else:
            assert np.all(view[:] == myrank)
            assert np.all(view2[:] == myrank)
            assert view2.size == 0

    @pytest.mark.parallel(mode=4)
    def test_from_replicated_to_distributed(self, mode):
        shape = (4, 4)
        grid = Grid(shape=shape)
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        u = Function(name='u', grid=grid, space_order=0)  # distributed
        v = Function(name='v', grid=grid, space_order=0)  # distributed
        a = np.arange(16).reshape(shape)  # replicated

        # Full array
        u.data[:] = a
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data == [[0, 1], [4, 5]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data == [[2, 3], [6, 7]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data == [[8, 9], [12, 13]])
        else:
            assert np.all(u.data == [[10, 11], [14, 15]])

        # Subsection (all ranks touched)
        u.data[:] = 0
        u.data[1:3, 1:3] = a[1:3, 1:3]
        # Same as above but with negative indices
        v.data[:] = 0
        v.data[1:-1, 1:-1] = a[1:-1, 1:-1]
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data == [[0, 0], [0, 5]])
            assert np.all(v.data == [[0, 0], [0, 5]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data == [[0, 0], [6, 0]])
            assert np.all(v.data == [[0, 0], [6, 0]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data == [[0, 9], [0, 0]])
            assert np.all(v.data == [[0, 9], [0, 0]])
        else:
            assert np.all(u.data == [[10, 0], [0, 0]])
            assert np.all(v.data == [[10, 0], [0, 0]])

        # The assigned data must have same shape as the one of the distributed array,
        # otherwise an exception is expected
        try:
            u.data[1:3, 1:3] = a[1:2, 1:2]
        except ValueError:
            assert True
        except Exception as e:
            raise AssertionError('Assert False') from e

    @pytest.mark.parallel(mode=4)
    def test_misc_setup(self, mode):
        """Test setup of Functions with mixed distributed/replicated Dimensions."""
        grid = Grid(shape=(4, 4))
        _, y = grid.dimensions
        dy = Dimension(name='dy')

        # Note: `grid` must be passed to `c` since `x` is a distributed dimension,
        # and `grid` carries the `x` decomposition
        c = Function(name='c', grid=grid, dimensions=(y, dy), shape=(4, 5))

        # The following should be identical to `c` in everything but the name
        c2 = Function(name='c2', grid=grid, dimensions=(y, dy), shape=(None, 5))
        assert c.shape == c2.shape == (2, 5)
        assert c.shape_with_halo == c2.shape_with_halo
        assert c._decomposition == c2._decomposition

        # The following should all raise an exception as illegal
        try:
            Function(name='c3', grid=grid, dimensions=(y, dy))
            raise AssertionError('Assert False')
        except TypeError:
            # Missing `shape`
            assert True

        # The following should all raise an exception as illegal
        try:
            Function(name='c4', grid=grid, dimensions=(y, dy), shape=(3, 5))
            raise AssertionError('Assert False')
        except ValueError:
            # The provided y-size, 3, doesn't match the y-size in grid (4)
            assert True

        # The following should all raise an exception as illegal
        try:
            Function(name='c4', grid=grid, dimensions=(y, dy), shape=(4,))
            raise AssertionError('Assert False')
        except ValueError:
            # Too few entries for `shape` (two expected, for `y` and `dy`)
            assert True

    @pytest.mark.parallel(mode=4)
    def test_misc_data(self, mode):
        """
        Test data insertion/indexing for Functions with mixed
        distributed/replicated Dimensions.
        """
        dx = Dimension(name='dx')
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map

        # Note: `grid` must be passed to `c` since `x` is a distributed dimension,
        # and `grid` carries the `x` decomposition
        c = Function(name='c', grid=grid, dimensions=(x, dx), shape=(4, 5))

        # Data insertion
        for i in range(4):
            c.data[i, 0] = 1.0+i
            c.data[i, 1] = 1.0+i
            c.data[i, 2] = 3.0+i
            c.data[i, 3] = 6.0+i
            c.data[i, 4] = 5.0+i

        # Data indexing
        if LEFT in glb_pos_map[x]:
            assert(np.all(c.data[0] == [1., 1., 3., 6., 5.]))
            assert(np.all(c.data[1] == [2., 2., 4., 7., 6.]))
        else:
            assert(np.all(c.data[2] == [3., 3., 5., 8., 7.]))
            assert(np.all(c.data[3] == [4., 4., 6., 9., 8.]))

        # Same as before, but with negative indices and non-trivial slices
        if LEFT in glb_pos_map[x]:
            assert(np.all(c.data[0:-3] == [1., 1., 3., 6., 5.]))
            assert(np.all(c.data[-3:-2] == [2., 2., 4., 7., 6.]))
        else:
            assert(np.all(c.data[-2:-1] == [3., 3., 5., 8., 7.]))
            assert(np.all(c.data[-1] == [4., 4., 6., 9., 8.]))

    @pytest.mark.parallel(mode=[4, 7])
    @pytest.mark.parametrize('gslice', [
        (slice(None, None, -1), slice(None, None, -1), 0),
        (slice(None, None, -1), slice(None, None, -1), slice(0, 1, 1)),
        (slice(None, None, -1), 0, slice(None, None, -1)),
        (slice(None, None, -1), slice(0, 1, 1), slice(None, None, -1)),
        (0, slice(None, None, -1), slice(None, None, -1)),
        (slice(0, 1, 1), slice(None, None, -1), slice(None, None, -1))])
    def test_inversions(self, gslice, mode):
        """Index flipping along different axes is local (induced decomposition)."""
        grid = Grid(shape=(8, 8, 8))
        f = Function(name='f', grid=grid)
        vdat = np.zeros((8, 8, 8))
        vdat[:, :, 0] = np.arange(64).reshape(8, 8)

        f.data[:, :, 0] = vdat[:, :, 0]
        self._assert_induced(f.data[gslice], vdat[gslice])

    @pytest.mark.parallel(mode=4)
    def test_setitem_shorthands(self, mode):
        # Test setitem with various slicing shorthands
        nx = 8
        ny = 8
        nz = 8
        shape1 = (nx, ny)
        shape2 = (nx, ny, nz)
        grid1 = Grid(shape=shape1, dtype=np.float32)
        x, y = grid1.dimensions
        glb_pos_map = grid1.distributor.glb_pos_map
        grid2 = Grid(shape=shape2, dtype=np.float32)
        f1 = Function(name='f1', grid=grid1)
        f2 = Function(name='f2', grid=grid1)
        g = Function(name='g', grid=grid2)

        dat1 = np.arange(64, dtype=np.int32).reshape(grid1.shape)

        f1.data[:] = dat1
        f2.data[:] = f1.data[:]
        assert np.all(f2.data[:] == f1.data[:])

        g.data[0, :, :] = dat1
        f1.data[:] = g.data[0, ::-1, ::-1]
        result = np.array(f1.data[:])
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(result[0] == [[63, 62, 61, 60]])
            assert np.all(result[1] == [[55, 54, 53, 52]])
            assert np.all(result[2] == [[47, 46, 45, 44]])
            assert np.all(result[3] == [[39, 38, 37, 36]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(result[0] == [[59, 58, 57, 56]])
            assert np.all(result[1] == [[51, 50, 49, 48]])
            assert np.all(result[2] == [[43, 42, 41, 40]])
            assert np.all(result[3] == [[35, 34, 33, 32]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(result[0] == [[31, 30, 29, 28]])
            assert np.all(result[1] == [[23, 22, 21, 20]])
            assert np.all(result[2] == [[15, 14, 13, 12]])
            assert np.all(result[3] == [[7, 6, 5, 4]])
        else:
            assert np.all(result[0] == [[27, 26, 25, 24]])
            assert np.all(result[1] == [[19, 18, 17, 16]])
            assert np.all(result[2] == [[11, 10, 9, 8]])
            assert np.all(result[3] == [[3, 2, 1, 0]])


class TestDataGather:

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('rank', [0, 1, 2, 3])
    def test_simple_gather(self, rank, mode):
        """ Test a simple gather on various ranks."""
        grid = Grid(shape=(10, 10), extent=(9, 9))
        f = Function(name='f', grid=grid, dtype=np.int32)
        res = np.arange(100).reshape(grid.shape)
        f.data[:] = res
        myrank = grid._distributor.comm.Get_rank()
        ans = f.data_gather(rank=rank)
        if myrank == rank:
            assert np.all(ans == res)
        else:
            assert ans == np.array(None)

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('start, stop, step', [
        (None, None, None),
        (None, None, 2),
        (None, None, -1),
        (None, None, -2),
        (1, 8, 3),
        # Strided slices whose start falls inside a subdomain (see
        # TestDecomposition.test_glb_to_loc_strided_start_in_subdomain)
        (2, 8, 2),
        (7, 0, -2),
        ((0, 4), None, (2, 1))])
    def test_sliced_gather_2D(self, start, stop, step, mode):
        """ Test gather for various 2D slices."""
        grid = Grid(shape=(10, 10), extent=(9, 9))
        f = Function(name='f', grid=grid, dtype=np.int32)
        dat = np.arange(100).reshape(grid.shape)

        if isinstance(step, int) or step is None:
            step = [step for _ in grid.shape]
        if isinstance(start, int) or start is None:
            start = [start for _ in grid.shape]
        if isinstance(stop, int) or stop is None:
            stop = [stop for _ in grid.shape]
        idx = []
        for i, j, k in zip(start, stop, step, strict=True):
            idx.append(slice(i, j, k))
        idx = tuple(idx)

        res = dat[idx]
        f.data[:] = dat
        myrank = grid._distributor.comm.Get_rank()
        ans = f.data_gather(start=start, stop=stop, step=step)
        if myrank == 0:
            assert np.all(ans == res)
        else:
            assert ans == np.array(None)

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('start, stop, step', [
        (None, None, None),
        (None, None, 2),
        (None, None, -1),
        (None, None, -2),
        (1, 8, 3),
        (2, 8, 2),
        (7, 0, -2),
        ((0, 4, 4), None, (2, 1, 1))])
    def test_sliced_gather_3D(self, start, stop, step, mode):
        """ Test gather for various 3D slices."""
        grid = Grid(shape=(10, 10, 10), extent=(9, 9, 9))
        f = Function(name='f', grid=grid, dtype=np.int32)
        dat = np.arange(1000).reshape(grid.shape)

        if isinstance(step, int) or step is None:
            step = [step for _ in grid.shape]
        if isinstance(start, int) or start is None:
            start = [start for _ in grid.shape]
        if isinstance(stop, int) or stop is None:
            stop = [stop for _ in grid.shape]
        idx = []
        for i, j, k in zip(start, stop, step, strict=True):
            idx.append(slice(i, j, k))
        idx = tuple(idx)

        res = dat[idx]
        f.data[:] = dat
        myrank = grid._distributor.comm.Get_rank()
        ans = f.data_gather(start=start, stop=stop, step=step)
        if myrank == 0:
            assert np.all(ans == res)
        else:
            assert ans == np.array(None)

    @pytest.mark.parallel(mode=[4, 6])
    def test_gather_time_function(self, mode):
        """ Test gathering of TimeFunction objects. """
        grid = Grid(shape=(11, 11))
        f = TimeFunction(name='f', grid=grid, save=11)
        op = Operator([Eq(f.forward, f+1)])
        op.apply(time_m=0, time_M=9)
        ans = f.data_gather(rank=0)
        tdata = np.zeros((11, 11, 11))
        for i in range(11):
            tdata[i, :] = np.float32(i)
        myrank = grid._distributor.comm.Get_rank()
        if myrank == 0:
            assert np.all(ans == tdata)
        else:
            assert ans == np.array(None)

    @pytest.mark.parallel(mode=[4, 6])
    @pytest.mark.parametrize('sfunc', [SparseFunction,
                                       SparseTimeFunction,
                                       PrecomputedSparseFunction,
                                       PrecomputedSparseTimeFunction])
    @pytest.mark.parametrize('target_rank', [0, 2])
    def test_gather_sparse(self, mode, sfunc, target_rank):
        grid = Grid((11, 11))
        myrank = grid._distributor.comm.Get_rank()
        nt = 10
        coords = [[0, 0], [0, .25], [0, .75], [0, 1]]
        s = sfunc(name='s', grid=grid, npoint=4, r=4, nt=nt, coordinates=coords)

        np.random.seed(1234)
        try:
            a = np.random.rand(s.nt, s.npoint_global)
        except AttributeError:
            a = np.random.rand(s.npoint_global,)

        s.data[:] = a
        out = s.data_gather(rank=target_rank)
        if myrank == target_rank:
            assert np.allclose(out, a)
        else:
            assert not out


def test_scalar_arg_substitution():
    """
    Tests the relaxed (compared to other devito sympy subclasses)
    substitution semantics for scalars, which is used for argument
    substitution into symbolic expressions.
    """
    t0 = Scalar(name='t0').indexify()
    t1 = Scalar(name='t1').indexify()
    assert t0 != 0
    assert t0.subs('t0', 2) == 2
    assert t0.subs('t0', t1) == t1


@pytest.mark.skip(reason="will corrupt memory and risk crash")
def test_oob_noguard():
    """
    Tests the guard page allocator.  This writes to memory it shouldn't,
    and typically gets away with it.
    """
    # A tiny grid
    grid = Grid(shape=(4, 4))
    u = Function(name='u', grid=grid, space_order=0, allocator=ALLOC_ALIGNED)
    Operator(Eq(u[2000, 0], 1.0)).apply()


@pytest.mark.skip(reason="will crash entire test suite")
def test_oob_guard():
    """
    Tests the guard page allocator.  This causes a segfault in the
    test suite, deliberately.
    """
    # A tiny grid
    grid = Grid(shape=(4, 4))
    u = Function(name='u', grid=grid, space_order=0, allocator=ALLOC_GUARD)
    Operator(Eq(u[2000, 0], 1.0)).apply()


def test_numpy_c_contiguous():
    """
    Test that devito.Data is correctly reported by NumPy as being C-contiguous
    """
    grid = Grid(shape=(4, 4))
    u = Function(name='u', grid=grid, space_order=2)
    assert(u._data_allocated.flags.c_contiguous)


def test_boolean_masking_array():
    """
    Test truth value of array, raised in Python 3.9 (MFE for issue #1788)
    """
    shape = (5,)
    extent = (6)

    grid = Grid(shape=shape, extent=extent)
    f = Function(name='f', grid=grid, dtype=np.int32)

    bool_arr = np.array([True, True, False, False, True])

    f.data[bool_arr] = 1

    assert all(f.data == [1, 1, 0, 0, 1])


class TestDataReference:
    """
    Tests for passing data to a Function using a reference to a
    preexisting array-like.
    """

    def test_w_array(self):
        """Test using a preexisting NumPy array as Function data"""
        grid = Grid(shape=(3, 3))
        a = np.reshape(np.arange(25, dtype=np.float32), (5, 5))
        b = a.copy()
        c = a.copy()

        b[1:-1, 1:-1] += 1

        f = Function(name='f', grid=grid, space_order=1,
                     allocator=DataReference(a))

        # Check that the array hasn't been zeroed
        assert np.any(a != 0)

        # Check that running operator updates the original array
        Operator(Eq(f, f+1))()
        assert np.all(a == b)

        # Check that updating the array updates the function data
        a[1:-1, 1:-1] -= 1
        assert np.all(f.data_with_halo == c)

    def _w_data(self):
        shape = (5, 5)
        grid = Grid(shape=shape)
        f = Function(name='f', grid=grid, space_order=1)
        f.data_with_halo[:] = np.reshape(np.arange(49, dtype=np.float32), (7, 7))

        g = Function(name='g', grid=grid, space_order=1,
                     allocator=DataReference(f._data))

        # Check that the array hasn't been zeroed
        assert np.any(f.data_with_halo != 0)

        assert np.all(f.data_with_halo == g.data_with_halo)

        # Update f
        Operator(Eq(f, f+1))()
        assert np.all(f.data_with_halo == g.data_with_halo)

        # Update g
        Operator(Eq(g, g+1))()
        assert np.all(f.data_with_halo == g.data_with_halo)

        check = np.array(f.data_with_halo[1:-1, 1:-1])

        # Update both
        Operator([Eq(f, f+1), Eq(g, g+1)])()
        assert np.all(f.data_with_halo == g.data_with_halo)
        # Check that it was incremented by two
        check += 2
        assert np.all(f.data == check)

    def test_w_data(self):
        """Test passing preexisting Function data to another Function"""
        self._w_data()

    @pytest.mark.parallel(mode=[2, 4])
    def test_w_data_mpi(self, mode):
        """
        Test passing preexisting Function data to another Function with MPI.
        """
        self._w_data()


if __name__ == "__main__":
    configuration['mpi'] = True
    TestDataDistributed().test_misc_data()

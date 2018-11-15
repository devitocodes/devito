from conftest import skipif_backend
import pytest
import numpy as np

from devito import (Grid, Function, TimeFunction, Dimension, Eq, Operator, # noqa
                    configuration, ALLOC_GUARD, ALLOC_FLAT)
from devito.data import Decomposition
from devito.types import LEFT, RIGHT

pytestmark = pytest.mark.skipif(configuration['backend'] == 'ops',
                                reason="testing is currently restricted")


class TestDataBasic(object):

    def test_simple_indexing(self):
        """
        Tests packing/unpacking data in :class:`Function` objects.
        """
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
        """
        Tests packing/unpacking data in :class:`Function` objects with more advanced
        access functions.
        """
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

    def test_halo_indexing(self):
        """
        Tests packing/unpacking data in :class:`Function` objects when some halo
        region is present.
        """
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

    def test_arithmetic(self):
        """
        Tests arithmetic operations between :class:`Data` objects and values.
        """
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

    @skipif_backend(['yask'])  # YASK and OPS backends do not support MPI yet
    def test_illegal_indexing(self):
        """
        Tests that indexing into illegal entries throws an exception.
        """
        nt = 5
        grid = Grid(shape=(4, 4, 4))
        u = Function(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid, save=nt)

        try:
            u.data[5]
            assert False
        except IndexError:
            pass
        try:
            v.data[nt]
            assert False
        except IndexError:
            pass

    def test_logic_indexing(self):
        """
        Tests logic indexing for stepping dimensions.
        """
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

    def test_domain_vs_halo(self):
        """
        Tests access to domain and halo data.
        """
        grid = Grid(shape=(4, 4, 4))

        # Without padding
        u0 = Function(name='u0', grid=grid, space_order=0)
        u2 = Function(name='u2', grid=grid, space_order=2)

        assert u0.shape == u0._shape_with_inhalo == u0.shape_allocated
        assert u0.shape_with_halo == u0._shape_with_inhalo  # W/o MPI, these two coincide
        assert len(u2.shape) == len(u2._extent_halo.left)
        assert tuple(i + j*2 for i, j in zip(u2.shape, u2._extent_halo.left)) ==\
            u2.shape_with_halo

        assert all(i == (0, 0) for i in u0._offset_domain)
        assert all(i == 0 for i in u0._offset_domain.left)
        assert all(i == 0 for i in u0._offset_domain.right)

        assert all(i == (2, 2) for i in u2._offset_domain)
        assert all(i == 2 for i in u2._offset_domain.left)
        assert all(i == 2 for i in u2._offset_domain.right)

        # With some random padding
        v = Function(name='v', grid=grid, space_order=2, padding=(1, 3, 4))
        assert len(v.shape_allocated) == len(u2._extent_padding.left)
        assert tuple(i + j + k for i, (j, k) in zip(v.shape_with_halo, v._padding)) ==\
            v.shape_allocated

        assert all(i == (2, 2) for i in v._halo)
        assert v._offset_domain == ((3, 3), (5, 5), (6, 6))
        assert v._offset_domain.left == v._offset_domain.right == (3, 5, 6)
        assert v._extent_padding == ((1, 1), (3, 3), (4, 4))
        assert v._extent_padding.left == v._extent_padding.right == (1, 3, 4)


@skipif_backend(['yask'])
class TestDecomposition(object):

    """
    .. note::

        If these tests don't work, definitely TestDataDistributed won't behave
    """

    def test_convert_index(self):
        d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)

        # A global index as single argument
        assert d.convert_index(5) == 0
        assert d.convert_index(6) == 1
        assert d.convert_index(7) == 2
        assert d.convert_index(3) is None

        # Retrieve relative local min/man given global min/max
        assert d.convert_index((5, 7)) == (0, 2)
        assert d.convert_index((5, 9)) == (0, 2)
        assert d.convert_index((1, 3)) == (-1, -3)
        assert d.convert_index((1, 6)) == (0, 1)
        assert d.convert_index((None, None)) == (0, 2)

        # Retrieve absolute local min/man given global min/max
        assert d.convert_index((5, 7), rel=False) == (5, 7)
        assert d.convert_index((5, 9), rel=False) == (5, 7)
        assert d.convert_index((1, 3), rel=False) == (-1, -3)
        assert d.convert_index((1, 6), rel=False) == (5, 6)
        assert d.convert_index((None, None), rel=False) == (5, 7)

    def test_reshape_identity(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Identity decomposition
        assert len(d.reshape(0, 0)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(0, 0), [[0, 1], [2, 3]]))

    def test_reshape_right_only(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Extension at right only
        assert len(d.reshape(0, 2)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(0, 2), [[0, 1], [2, 3, 4, 5]]))
        # Reduction at right affecting one sub-domain only, but not the whole subdomain
        assert len(d.reshape(0, -1)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(0, -1), [[0, 1], [2]]))
        # Reduction at right over one whole sub-domain
        assert len(d.reshape(0, -2)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(0, -2), [[0, 1], []]))
        # Reduction at right over multiple sub-domains
        assert len(d.reshape(0, -3)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(0, -3), [[0], []]))

    def test_reshape_left_only(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Extension at left only
        assert len(d.reshape(2, 0)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(2, 0), [[0, 1, 2, 3], [4, 5]]))
        # Reduction at left affecting one sub-domain only, but not the whole subdomain
        assert len(d.reshape(-1, 0)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-1, 0), [[0], [1, 2]]))
        # Reduction at left over one whole sub-domain
        assert len(d.reshape(-2, 0)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-2, 0), [[], [0, 1]]))
        # Reduction at right over multiple sub-domains
        assert len(d.reshape(-3, 0)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-3, 0), [[], [0]]))

    def test_reshape_left_right(self):
        d = Decomposition([[0, 1], [2, 3]], 2)

        # Extension at both left and right
        assert len(d.reshape(1, 1)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(1, 1), [[0, 1, 2], [3, 4, 5]]))
        # Reduction at both left and right
        assert len(d.reshape(-1, -1)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-1, -1), [[0], [1]]))
        # Reduction at both left and right, with the right one obliterating one subdomain
        assert len(d.reshape(-1, -2)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-1, -2), [[0], []]))
        # Reduction at both left and right obliterating all subdomains
        # triggering an exception
        assert len(d.reshape(-1, -3)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-1, -3), [[], []]))
        assert len(d.reshape(-2, -2)) == 2
        assert all(list(i) == j for i, j in zip(d.reshape(-1, -3), [[], []]))

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


@skipif_backend(['yask'])  # YASK and OPS backends do not support MPI yet
class TestDataDistributed(object):

    """
    Test Data indexing and manipulation when distributed over a set of MPI processes.
    """

    @pytest.mark.parallel(nprocs=4)
    def test_localviews(self):
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

    @pytest.mark.parallel(nprocs=4)
    def test_trivial_insertion(self):
        grid = Grid(shape=(4, 4))
        u = Function(name='u', grid=grid, space_order=0)
        v = Function(name='v', grid=grid, space_order=1)

        u.data[:] = 1.
        assert np.all(u.data == 1.)
        assert np.all(u.data._local == 1.)

        v.data_with_halo[:] = 1.
        assert v.data_with_halo[:].shape == (3, 3)
        assert np.all(v.data_with_halo == 1.)
        assert np.all(v.data_with_halo[:] == 1.)
        assert np.all(v.data_with_halo._local == 1.)

    @pytest.mark.parallel(nprocs=4)
    def test_indexing(self):
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

    @pytest.mark.parallel(nprocs=4)
    def test_slicing(self):
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

    @pytest.mark.parallel(nprocs=4)
    def test_indexing_in_views(self):
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

    @pytest.mark.parallel(nprocs=4)
    def test_from_replicated_to_distributed(self):
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
        except:
            assert False

    @pytest.mark.parallel(nprocs=4)
    def test_misc_setup(self):
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
            assert False
        except TypeError:
            # Missing `shape`
            assert True

        # The following should all raise an exception as illegal
        try:
            Function(name='c4', grid=grid, dimensions=(y, dy), shape=(3, 5))
            assert False
        except ValueError:
            # The provided y-size, 3, doesn't match the y-size in grid (4)
            assert True

        # The following should all raise an exception as illegal
        try:
            Function(name='c4', grid=grid, dimensions=(y, dy), shape=(4,))
            assert False
        except ValueError:
            # Too few entries for `shape` (two expected, for `y` and `dy`)
            assert True

    @pytest.mark.parallel(nprocs=4)
    def test_misc_data(self):
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


def test_scalar_arg_substitution(t0, t1):
    """
    Tests the relaxed (compared to other devito sympy subclasses)
    substitution semantics for scalars, which is used for argument
    substitution into symbolic expressions.
    """
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
    u = Function(name='u', grid=grid, space_order=0, allocator=ALLOC_FLAT)
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


if __name__ == "__main__":
    from devito import configuration
    configuration['mpi'] = True
    TestDataDistributed().test_misc_data()

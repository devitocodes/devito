"""Shared Data behavior, tested against both numerical storage backends."""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from devito import (  # noqa: E402
    Eq, Function, Grid, Operator, TimeFunction, TorchAllocator
)
from devito.data import Data, Decomposition, NumpyData, TorchData  # noqa: E402


@pytest.fixture(params=['numpy', 'torch'])
def allocator(request):
    if request.param == 'torch':
        if torch is None:
            pytest.skip('Requires PyTorch')
        return TorchAllocator()
    return None


def native(data):
    if isinstance(data, Data):
        data = data._raw
    return data.numpy() if torch is not None and isinstance(data, torch.Tensor) \
        else np.asarray(data)


class TestDataBackends:

    def test_common_base(self, allocator):
        data = Data((4,), np.float32, modulo=(False,),
                    **({'allocator': allocator} if allocator else {}))
        assert isinstance(data, Data)
        assert isinstance(data, TorchData if allocator else NumpyData)
        assert not issubclass(Data, np.ndarray)

    def test_host_view_shares_storage(self, allocator):
        f = Function(name='host_view', grid=Grid(shape=(8, 8)), space_order=0,
                     allocator=allocator)
        host = f.data._host_array()
        assert isinstance(host, np.ndarray)
        host[2, 3] = 7
        assert f.data[2, 3] == 7
        f.data[4, 5] = 9
        assert host[4, 5] == 9

    def test_modulo_and_views(self, allocator):
        grid = Grid(shape=(8, 8))
        u = TimeFunction(name='u', grid=grid, allocator=allocator)
        u.data[0] = 2
        u.data[1] = 3
        np.testing.assert_equal(native(u.data[4]), np.full((8, 8), 2))
        np.testing.assert_equal(native(u.data[-3]), np.full((8, 8), 3))
        u.data[5, 2, 3] = 7
        assert u.data[1, 2, 3] == 7
        view = u.data[:, 1:5, 2:6]
        assert isinstance(view, Data)
        assert view._modulo == (True, False, False)
        assert view._distributor is u.grid.distributor
        view[4, 0, 0] = 9
        assert u.data[0, 1, 2] == 9
        projected = view[1]
        assert projected._modulo == (False, False)
        np.testing.assert_equal(native((view+1)[2]), native(view[0])+1)
        if torch is not None and isinstance(view, torch.Tensor):
            result = torch.add(input=view, other=1)
            np.testing.assert_equal(native(result[2]), native(view[0])+1)

    def test_decomposition_and_nonlocal_access(self, allocator):
        decomposition = Decomposition([range(4), range(4, 8)], 1)
        data = Data((4,), np.float32, decomposition=(decomposition,),
                    modulo=(False,), **({'allocator': allocator} if allocator else {}))
        data._local[:] = np.arange(4, 8, dtype=np.float32)
        assert data[2] is None
        assert data[5] == 5
        np.testing.assert_equal(native(data._local), np.arange(4, 8))
        view = data[5:7]
        assert isinstance(view, Data)
        assert view._decomposition[0].loc_abs_min == 0
        assert view[0] == 5
        view[1] = 12
        assert data[6] == 12
        assert native(data[:2]).size == 0

    def test_buffered_reverse_assignment_and_gather(self, allocator):
        u = TimeFunction(name='u', grid=Grid(shape=(8, 8)), allocator=allocator)
        u.data[0, ::-1, 0] = np.arange(8, dtype=np.float32)
        np.testing.assert_equal(native(u.data[0, :, 0]), np.arange(7, -1, -1))
        gathered = u.data_gather()
        np.testing.assert_equal(native(gathered), native(u.data))

    def test_private_data_accessors_use_native_indexing(self, allocator):
        u = TimeFunction(name='native_access', grid=Grid(shape=(8, 8)),
                         space_order=0, padding=0, allocator=allocator)
        # Public time buffers wrap, while raw storage follows native array bounds.
        u.data[2] = 3
        for data in (u._data_allocated, u._data_with_inhalo,
                     u._data_ro_allocated, u._data_ro_with_inhalo):
            assert not isinstance(data, Data)
            with pytest.raises(IndexError):
                data[2]
            np.testing.assert_equal(native(data[0]),
                                    np.full((8, 8), 3))


class TestDataIndexing:

    def test_indexing_assignment_and_broadcasting(self, allocator):
        f = Function(name='f', grid=Grid(shape=(8, 8)), space_order=0,
                     allocator=allocator)
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        f.data[:] = expected
        for index in [(slice(None, None, -1), slice(None, None, -2)),
                      (Ellipsis, 3), ([1, 4, 6], slice(2, 5)),
                      (np.array([True, False]*4), slice(None))]:
            np.testing.assert_equal(native(f.data[index]), expected[index])
        f.data[6:1:-2, 7:2:-2] = np.arange(9, dtype=np.float32).reshape(3, 3)
        expected[6:1:-2, 7:2:-2] = np.arange(9).reshape(3, 3)
        np.testing.assert_equal(native(f.data), expected)
        f.data[:] = np.arange(8, dtype=np.float32)
        np.testing.assert_equal(native(f.data), np.broadcast_to(np.arange(8), (8, 8)))

    @pytest.mark.parametrize('case', [
        'spatial', 'all_time', 'leading', 'ellipsis', 'trailing_ellipsis', 'modulo'
    ])
    def test_multidimensional_boolean_masks(self, allocator, case):
        u = TimeFunction(name='masked', grid=Grid(shape=(4, 5)), space_order=0,
                         padding=0, allocator=allocator)
        expected = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
        u.data[:] = expected
        spatial = np.arange(20).reshape(4, 5) % 3 == 0
        leading = np.arange(8).reshape(2, 4) % 2 == 0
        if case == 'spatial':
            index = (0, spatial)
        elif case == 'all_time':
            index = (slice(None), spatial)
        elif case == 'leading':
            index = (leading, slice(None))
        elif case == 'ellipsis':
            index = (Ellipsis, spatial)
        elif case == 'trailing_ellipsis':
            index = (leading, Ellipsis)
        else:
            index = (4, spatial)
        reference = (0, spatial) if case == 'modulo' else index
        if allocator is not None:
            index = tuple(torch.from_numpy(i) if isinstance(i, np.ndarray) else i
                          for i in index)
        np.testing.assert_equal(native(u.data[index]), expected[reference])
        u.data[index] = 9
        expected[reference] = 9
        np.testing.assert_equal(native(u.data), expected)

    @pytest.mark.parametrize('index', [
        (slice(0, 1, -1), slice(None)),
        (slice(3, 3, -1), slice(None)),
        (slice(2, 1), slice(None, None, -1)),
        (slice(None, None, -1), slice(0, 1, -1))
    ])
    def test_empty_reverse_slices(self, allocator, index):
        f = Function(name='empty_reverse', grid=Grid(shape=(8, 8)), space_order=0,
                     allocator=allocator)
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        f.data[:] = expected
        np.testing.assert_equal(native(f.data[index]), expected[index])
        f.data[index] = 9
        np.testing.assert_equal(native(f.data), expected)


class TestDataMPI:

    @pytest.mark.parallel(mode=4)
    def test_mpi_indexing_halo_and_gather(self, allocator, mode):
        grid = Grid(shape=(8, 8), extent=(7., 7.))
        f = Function(name='f', grid=grid, space_order=1, allocator=allocator)
        values = np.arange(64, dtype=np.float32).reshape(8, 8)
        f.data[:] = values
        owned = grid.distributor.glb_slices
        selection = tuple(owned[d] for d in grid.dimensions)
        np.testing.assert_equal(native(f.data._local), values[selection])
        for x, y in [(0, 0), (4, 4), (7, 7)]:
            actual = f.data[x, y]
            if selection[0].start <= x < selection[0].stop and \
               selection[1].start <= y < selection[1].stop:
                assert actual == values[x, y]
            else:
                assert actual is None
        # Existing Function halo orchestration, using backend-specific MPI buffers.
        halo = native(f._data_with_inhalo)
        padded = np.pad(values, 1)
        expected_halo = padded[selection[0].start:selection[0].stop+2,
                               selection[1].start:selection[1].stop+2]
        np.testing.assert_equal(halo, expected_halo)
        gathered = f.data_gather(rank=2)
        if grid.distributor.myrank == 2:
            np.testing.assert_equal(native(gathered), values)
        else:
            assert gathered is None
        reversed_data = f.data[::-1, ::-1]
        np.testing.assert_equal(native(reversed_data._local),
                                values[selection][::-1, ::-1])
        f.data[:] = reversed_data
        gathered = f.data_gather()
        if grid.distributor.myrank == 0:
            np.testing.assert_equal(native(gathered), values[::-1, ::-1])
        # Replicated values follow the same global negative-slice assignment semantics.
        f.data[::-1, ::-1] = values
        gathered = f.data_gather()
        if grid.distributor.myrank == 0:
            np.testing.assert_equal(native(gathered), values[::-1, ::-1])

    @pytest.mark.parallel(mode=2)
    def test_mpi_tensor_kernel(self, allocator, mode):
        grid = Grid(shape=(16, 8), extent=(15., 7.))
        u = TimeFunction(name='u', grid=grid, space_order=2, allocator=allocator)
        u.data[0] = 1
        language = 'torch' if allocator else 'C'
        op = Operator(Eq(u.forward, u + 0.1*u.laplace), language=language)
        op.apply(time_M=0)
        # MPI-generated stencil execution must agree with the same global reference.
        gathered = u.data_gather()
        if grid.distributor.myrank == 0:
            expected = np.ones((16, 8), dtype=np.float32)
            expected[0, :] -= 0.1
            expected[-1, :] -= 0.1
            expected[:, 0] -= 0.1
            expected[:, -1] -= 0.1
            np.testing.assert_allclose(native(gathered)[1], expected, atol=1e-6)

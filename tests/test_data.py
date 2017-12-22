from conftest import skipif_yask

import numpy as np
import pytest

from devito import Grid, Function, TimeFunction


@pytest.fixture
def u():
    grid = Grid(shape=(16, 16, 16))
    u = Function(name='yu3D', grid=grid, space_order=0)
    return u


def test_basic_indexing(u):
    """
    Tests packing/unpacking :class:`Data` objects.
    """
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


def test_basic_arithmetic(u):
    """
    Tests arithmetic operations between :class:`Data` objects and values.
    """
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


@skipif_yask  # YASK not throwing excpetions yet
def test_illegal_indexing():
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


def test_logic_indexing():
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


def test_domain_vs_halo():
    """
    Tests access to domain and halo data.
    """
    grid = Grid(shape=(4, 4, 4))
    u0 = Function(name='u0', grid=grid, space_order=0)
    u2 = Function(name='u2', grid=grid, space_order=2)

    assert u0.shape == u0.shape_with_halo == u0.shape_allocated
    assert len(u2.shape) == len(u2._extent_halo_left)
    assert tuple(i + j*2 for i, j in zip(u2.shape, u2._extent_halo_left)) ==\
        u2.shape_with_halo

    v = Function(name='v', grid=grid, space_order=2, padding=(1, 3, 4))
    assert len(v.shape_allocated) == len(u2._extent_padding_left)
    assert tuple(i + j + k for i, (j, k) in zip(v.shape_with_halo, v._padding)) ==\
        v.shape_allocated

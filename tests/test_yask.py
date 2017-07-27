import numpy as np
from sympy import Eq  # noqa

import pytest  # noqa

pexpect = pytest.importorskip('yask_compiler')  # Run only if YASK is available

from devito import Operator, DenseData, t, x, y, z, configuration  # noqa
from devito.yask.interfaces import YaskGrid  # noqa

pytestmark = pytest.mark.skipif(configuration['backend'] != 'yask',
                                reason="'yask' wasn't selected as backend on startup")


def test_data_type():
    u = DenseData(name='yu3D', shape=(16, 16, 16), dimensions=(x, y, z), space_order=0)
    u.data  # Trigger initialization
    assert type(u._data_object) == YaskGrid


@pytest.mark.xfail(reason="YASK always seems to use 3D grids")
def test_data_movement_1D():
    u = DenseData(name='yu1D', shape=(16,), dimensions=(x,), space_order=0)
    u.data
    assert type(u._data_object) == YaskGrid

    u.data[1] = 1.
    assert u.data[0] == 0.
    assert u.data[1] == 1.
    assert all(i == 0 for i in u.data[2:])


def test_data_movement_nD():
    u = DenseData(name='yu3D', shape=(16, 16, 16), dimensions=(x, y, z), space_order=0)
    u.data
    assert type(u._data_object) == YaskGrid

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
    u.data[4, :, 4] = block
    assert np.all(u.data[4, :, 4] == block)


def test_data_arithmetic_nD():
    u = DenseData(name='yu3D', shape=(16, 16, 16), dimensions=(x, y, z), space_order=0)

    # Simple arithmetic
    u.data[:] = 1
    assert np.all(u.data == 1)
    assert np.all(u.data + 2. == 3.)
    assert np.all(u.data - 2. == -1.)
    assert np.all(u.data * 2. == 2.)
    assert np.all(u.data / 2. == 0.5)
    assert np.all(u.data % 2 == 1.)

    # Increments and parital increments
    u.data[:] += 2.
    assert np.all(u.data == 3.)
    u.data[9, :, :] += 1.
    assert all(np.all(u.data[i, :, :] == 3.) for i in range(9))
    assert np.all(u.data[9, :, :] == 4.)

    # Right operations __rOP__
    u.data[:] = 1.
    arr = np.ndarray(shape=(16, 16, 16), dtype=np.float32)
    arr.fill(2.)
    assert np.all(arr - u.data == -1.)


@pytest.mark.parametrize("space_order", [0])
def test_trivial_operator(space_order):
    u = DenseData(name='yu4D', shape=(2, 16, 16, 16), dimensions=(t, x, y, z),
                  space_order=space_order)
    op = Operator(Eq(u.indexed[t + 1, x, y, z], u.indexed[t, x, y, z] + 1.))
    op(u, t=1)
    assert np.all(u.data == 1.)

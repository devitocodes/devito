import numpy as np
from sympy import Eq  # noqa

import pytest  # noqa

pexpect = pytest.importorskip('yask_compiler')  # Run only if YASK is available

from devito import Operator, DenseData, x, y, z, configuration  # noqa
from devito.yask.interfaces import YaskGrid  # noqa

pytestmark = pytest.mark.skipif(configuration['backend'] != 'yask',
                                reason="'yask' wasn't selected as backend on startup")


def test_data_type():
    u = DenseData(name='yu', shape=(10, 10, 10), dimensions=(x, y, z))
    u.data  # Trigger initialization
    assert type(u._data_object) == YaskGrid


@pytest.mark.xfail(reason="YASK always seems to use 3D grids")
def test_data_movement_1D():
    u = DenseData(name='yu1D', shape=(10,), dimensions=(x,))
    assert type(u._data_object) == YaskGrid

    u.data[1] = 1.
    assert u.data[0] == 0.
    assert u.data[1] == 1.
    assert all(i == 0 for i in u.data[2:])


def test_data_movement_nD():
    u = DenseData(name='yu3D', shape=(10, 10, 10), dimensions=(x, y, z))

    # Test simple insertion and extraction
    u.data[0, 1, 1] = 1.
    assert u.data[0, 0, 0] == 0.
    assert u.data[0, 1, 1] == 1.
    assert np.all(u.data == u.data[:, :, :])
    assert 1. in u.data[0]
    assert 1. in u.data[0, 1]

    # Test negative indices
    assert u.data[0, -9, -9] == 1.
    u.data[6, 0, 0] = 1.
    assert u.data[-4, :, :].sum() == 1.

    # Test setting whole array to given value
    u.data[:] = 3.
    assert np.all(u.data == 3.)

    # Test insertion of single value into block
    u.data[5, :, 5] = 5.
    assert np.all(u.data[5, :, 5] == 5.)

    # Test insertion of block into block
    block = np.ndarray(shape=(1, 10, 1), dtype=np.float32)
    block.fill(4.)
    u.data[4, :, 4] = block
    assert np.all(u.data[4, :, 4] == block)


def test_data_arithmetic_nD():
    u = DenseData(name='yu3D', shape=(10, 10, 10), dimensions=(x, y, z))

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
    arr = np.ndarray(shape=(10, 10, 10), dtype=np.float32)
    arr.fill(2.)
    assert np.all(arr - u.data == -1.)


def test_simple_operator():
    u = DenseData(name='yu', shape=(10, 10, 10), dimensions=(x, y, z))
    u.data
    # FIXME: Yask crashes because the support to run Devito operators through
    # YASK is still mostly missing
    # op = Operator(Eq(u, 1.))
    # op.apply(u)
    # assert np.allclose(u.data, 1)

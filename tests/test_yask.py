import numpy as np
from sympy import Eq

import pytest  # noqa

from devito import Operator, DenseData, x, y, z
from devito.parameters import configuration, defaults
from devito.dle.backends.yask import YaskGrid


def setup_module(module):
    configuration['dle'] = 'yask'


def teardown_module(module):
    configuration['dle'] = defaults['dle']


def test_data_type():
    u = DenseData(name='yu', shape=(10, 10), dimensions=(x, y))
    assert type(u.data) == YaskGrid


@pytest.mark.xfail(reason="YASK always seems to use 3D grids")
def test_data_movement_1D():
    u = DenseData(name='yu1D', shape=(10,), dimensions=(x,))
    u.data[1] = 1.
    assert u.data[0] == 0.
    assert u.data[1] == 1.
    assert all(i == 0 for i in u.data[2:])


@pytest.mark.xfail(reason="FAIL on block insertion")
def test_data_movement_nD():
    u = DenseData(name='yu3D', shape=(10, 10, 10), dimensions=(x, y, z))
    u.data[0, 1, 1] = 1.
    assert u.data[0, 0, 0] == 0.
    assert u.data[0, 1, 1] == 1.
    assert np.all(u.data[:] == u.data[:,:,:])

    # Test block insertion
    block = np.ndarray(shape=(1, 10, 1))
    block.fill(5.)
    u.data[5,:,5] = block
    assert np.all(u.data[5,:,5] == block)


def test_storage_layout():
    u = DenseData(name='yu', shape=(10, 10), dimensions=(x, y))
    op = Operator(Eq(u, 1.), dse='noop', dle='noop')
    op.apply(u)
    assert np.allclose(u.data, 1)

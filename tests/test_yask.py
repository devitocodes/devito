import numpy as np
from sympy import Eq

import pytest  # noqa

from devito import Operator, DenseData, x, y, z
from devito.parameters import configuration, defaults
from devito.dle.backends import YaskGrid


def setup_module(module):
    configuration['dle'] = 'yask'


def teardown_module(module):
    configuration['dle'] = defaults['dle']


def test_data_type():
    u = DenseData(name='yu', shape=(10, 10), dimensions=(x, y))
    assert type(u.data) == YaskGrid


def test_data_swap():
    u = DenseData(name='yu1D', shape=(10,), dimensions=(x,))
    u.data[1] = 1.
    assert u.data[1] == 1.
    u = DenseData(name='yu2D', shape=(10, 10), dimensions=(x, y))
    u.data[0, 1] = 1.
    assert u.data[1, 0] == 1.
    u = DenseData(name='yu3D', shape=(10, 10, 10), dimensions=(x, y, z))
    u.data[0, 1, 1] = 1.
    assert u.data[1, 1, 0] == 1.


def test_storage_layout():
    u = DenseData(name='yu', shape=(10, 10), dimensions=(x, y))
    op = Operator(Eq(u, 1.), dse='noop', dle='noop')
    op.apply(u)
    assert np.allclose(u.data, 1)

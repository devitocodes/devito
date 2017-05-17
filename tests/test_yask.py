from sympy import Eq

import pytest

from devito import DLE_DEFAULT, Operator, DenseData, parameters, x, y
from devito.dle.backends import yaskarray


def setup_module(module):
    parameters['dle']['mode'] = 'yask'


def teardown_module(module):
    parameters['dle']['mode'] = DLE_DEFAULT


def test_data_type():
    u = DenseData(name='yu', shape=(10, 10), dimensions=(x, y))
    assert type(u.data) == yaskarray

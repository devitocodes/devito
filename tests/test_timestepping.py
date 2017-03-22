import numpy as np
from sympy import Eq

import pytest

from devito.interfaces import Backward, Forward, TimeData
from devito.stencilkernel import StencilKernel


@pytest.fixture
def a(shape=(11, 11)):
    """Forward time data object, unrolled (save=True)"""
    return TimeData(name='a', shape=shape, time_order=1,
                    time_dim=6, save=True)


@pytest.fixture
def b(shape=(11, 11)):
    """Backward time data object, unrolled (save=True)"""
    return TimeData(name='b', shape=shape, time_order=1,
                    time_dim=6, save=True)


@pytest.fixture
def c(shape=(11, 11)):
    """Forward time data object, buffered (save=False)"""
    return TimeData(name='c', shape=shape, time_order=1,
                    save=False, time_axis=Forward)


def test_forward(a, nt=5):
    a.data[0, :] = 1.
    eqn = Eq(a.forward, a + 1.)
    StencilKernel(eqn, dle=None, dse=None)()
    for i in range(nt):
        assert np.allclose(a.data[i, :], 1. + i, rtol=1.e-12)


def test_backward(b, nt=5):
    b.data[nt, :] = 6.
    eqn = Eq(b.backward, b - 1.)
    StencilKernel(eqn, dle=None, dse=None, time_axis=Backward)(time=nt)
    for i in range(nt + 1):
        assert np.allclose(b.data[i, :], 1. + i, rtol=1.e-12)


def test_forward_unroll(a, c, nt=5):
    """Test forward time marching with a buffered and an unrolled t"""
    a.data[0, :] = 1.
    c.data[0, :] = 1.
    eqn_c = Eq(c.forward, c + 1.)
    eqn_a = Eq(a.forward, c.forward)
    StencilKernel([eqn_c, eqn_a], dle=None, dse=None)(time=nt)
    for i in range(nt):
        assert np.allclose(a.data[i, :], 1. + i, rtol=1.e-12)

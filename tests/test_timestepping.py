import numpy as np
from sympy import Eq

import pytest

from devito.interfaces import TimeData
from devito.stencilkernel import StencilKernel


@pytest.fixture
def a(shape=(11, 11)):
    """Forward time data object, unrolled (save=True)"""
    return TimeData(name='a', shape=shape, time_order=1,
                    time_dim=6, save=True)


def test_forward(a, nt=5):
    a.data[0, :] = 1.
    eqn = Eq(a.forward, a + 1.)
    StencilKernel(eqn, dle=None, dse=None)()
    for i in range(nt):
        assert np.allclose(a.data[i, :], 1. + i, rtol=1.e-12)

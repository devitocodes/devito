import weakref

import numpy as np
import pytest

from conftest import skipif
from devito import Grid, Function, TimeFunction

pytestmark = skipif(['yask', 'ops'])


@pytest.mark.parametrize('FunctionToBeNamed', [Function, TimeFunction])
def test_cache_function_new(FunctionToBeNamed):
    """Test that ..."""
    grid = Grid(shape=(2,2))
    u0 = TimeFunction(name='u', grid=grid)
    u1 = TimeFunction(name='u', grid=grid, coefficients='symbolic')
    eq1 = Eq(u1.dt-u1.dx)
    eq0 = Eq(-u0.dx+u0.dt)
    assert(eq1.__repr__() == eq0.__repr__())

import weakref

import numpy as np
import pytest

from conftest import skipif
from devito import Grid, Function, TimeFunction

pytestmark = skipif(['yask', 'ops'])


@pytest.mark.parametrize('FunctionToBeNamed', [Function, TimeFunction])
def test_cache_function_new(FunctionToBeNamed):
    """Test that ..."""
    grid = Grid(shape=(1,1))
    u1 = TimeFunction(name='u', grid=grid)

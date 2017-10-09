import devito
import pytest
from conftest import skipif_yask
from devito import Grid, FixedDimension, Function, y


@skipif_yask
@pytest.mark.xfail
def test_incorrect_usage():
    grid = Grid(shape=(10, 10))
    m = Function(name="m", grid=grid)
    devito.x = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    assert(devito.x in m.indices)


@skipif_yask
def test_correct_usage():
    myx = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    grid = Grid(shape=(10, 10), dimensions=(myx, y))
    m = Function(name="m", grid=grid)
    assert(myx in m.indices)

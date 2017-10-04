import devito
import pytest
from devito import Grid, FixedDimension, DenseData, y


@pytest.mark.xfail
def test_incorrect_usage():
    grid = Grid(shape=(10, 10))
    m = DenseData(name="m", grid=grid)
    devito.x = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    assert(devito.x in m.indices)


def test_correct_usage():
    myx = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    grid = Grid(shape=(10, 10), dimensions=(myx, y))
    m = DenseData(name="m", grid=grid)
    assert(myx in m.indices)

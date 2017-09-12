import devito
from devito.dimension import FixedDimension
from devito import DenseData, y


def test1():
    m = DenseData(name="m", shape=(10, 10))
    devito.x = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    assert(devito.x in m.indices)


def test2():
    myx = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    m = DenseData(name="m", dimensions=(myx, y))
    assert(myx in m.indices)

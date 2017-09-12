import devito
from devito.dimension import FixedDimension
from devito import DenseData, TimeData, x, y, z

def test1():
    m = DenseData(name="m", shape=(10,10))
    devito.x = FixedDimension(name='x', size=10, spacing=devito.x.spacing)
    assert(devito.x in m.indices)



def test2():
    u=TimeData(name="u", dimensions=(x,y,z))

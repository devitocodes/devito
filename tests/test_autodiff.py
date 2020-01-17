from devito import Grid, Function, Eq
from devito.autodiff import Adjoint


def test_simple_stencil_ad():
    g = Grid(shape=(10, 10))

    i = Function(name="i", grid=g)
    r = Function(name="r", grid=g)

    x, y = g.dimensions
    
    stencil = Eq(r.indexed[x, y], 2*i.indexed[x-1, y] + 3*i.indexed[x, y] + 4*i.indexed[x+1, y])

    print(stencil)

    adj = Adjoint([stencil], {})

    print(adj)

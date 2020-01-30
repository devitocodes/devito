import pytest

from devito import Grid, Function, Eq, TimeFunction, solve, Operator
from devito.autodiff import Adjoint


def test_substitutions_reqd():
    g = Grid(shape=(10, 10))

    i = Function(name="i", grid=g)
    r = Function(name="r", grid=g)

    x, y = g.dimensions
    
    stencil = Eq(r.indexed[x, y], 2*i.indexed[x-1, y] + 3*i.indexed[x, y] + 4*i.indexed[x+1, y])

    ib = Function(name="ib", grid=g)
    rb = Function(name="rb", grid=g)
    
    with pytest.raises(ValueError):
        adj = Adjoint([stencil], {i: ib}).collection


def test_1d_stencil_ad():
    g = Grid(shape=(10, 10))

    i = Function(name="i", grid=g)
    r = Function(name="r", grid=g)

    x, y = g.dimensions
    
    stencil = Eq(r[x, y], 2*i[x-1, y] + 3*i[x, y] + 4*i[x+1, y])

    ib = Function(name="ib", grid=g)
    rb = Function(name="rb", grid=g)
    
    adj = Adjoint([stencil], {i: ib, r: rb}).collection

    expected = [Eq(ib[x - 1, y], 2*rb[x, y]), Eq(ib[x, y], 3*rb[x, y]),
                Eq(ib[x + 1, y], 4*rb[x, y])]
    assert(adj == expected)


def test_2d_stencil_ad():
    g = Grid(shape=(10, 10))

    i = Function(name="i", grid=g)
    r = Function(name="r", grid=g)

    x, y = g.dimensions
    
    stencil = Eq(r[x, y], 2*i[x, y] + 3*i[x-1, y] + 4*i[x+1, y] +
                 5*i[x, y-1] + 6*i[x, y+1])

    ib = Function(name="ib", grid=g)
    rb = Function(name="rb", grid=g)
    
    adj = Adjoint([stencil], {i: ib, r: rb}).collection

    expected = [Eq(ib[x, y], 2*rb[x, y]), Eq(ib[x-1, y], 3*rb[x, y]),
                Eq(ib[x + 1, y], 4*rb[x, y]), Eq(ib[x, y-1], 5*rb[x, y]),
                Eq(ib[x, y+1], 6*rb[x, y])]
    assert(adj == expected)

def test_time_stencil_ad():
    g = Grid(shape=(10, 10))

    f = TimeFunction(name="f", grid=g)

    fb = TimeFunction(name="fb", grid=g)

    x, y = g.dimensions

    time  = g.time_dim

    stencil = Eq(f[time, x, y], 3*f[time-1, x, y] + 1)

    adj = Adjoint([stencil], {f: fb}).collection

    expected = [Eq(fb[time - 1, x, y], 3*fb[time, x, y])]

    assert(adj == expected)

def test_ignored():
    g = Grid(shape=(10, 10))

    f = TimeFunction(name="f", grid=g)

    fb = TimeFunction(name="fb", grid=g)

    m = Function(name="m", grid=g)

    x, y = g.dimensions

    time  = g.time_dim

    stencil = Eq(f[time, x, y], m[x, y]*f[time-1, x, y] + 1)

    adj = Adjoint([stencil], {f: fb}, [m]).collection

    expected = [Eq(fb[time - 1, x, y], m[x, y]*fb[time, x, y])]

    assert(adj == expected)

def test_pde_stencil():
    g = Grid(shape=(10, 10))

    u = TimeFunction(name="u", grid=g, space_order=2, time_order=2)
    ub = TimeFunction(name="ub", grid=g, space_order=2, time_order=2)

    m = Function(name="m", grid=g, space_order=2)

    damp = Function(name="damp", grid=g, space_order=2)

    pde = m * u.dt2 - u.laplace + damp * u.dt

    stencil = Eq(u.forward, solve(pde, u.forward))

    op = Operator([stencil], dse='basic')

    print(op)

    adj = Adjoint([stencil], {u: ub}, [damp, m]).collection
    print(stencil)
    print("$$$")
    for e  in adj:
        print(e)

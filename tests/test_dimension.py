import numpy as np

from conftest import skipif_yask

from devito import ConditionalDimension, Grid, TimeFunction, Eq, Operator, Constant


@skipif_yask
def test_conditional_basic():
    nt = 19
    grid = Grid(shape=(11, 11))
    time = grid.time_dim

    u = TimeFunction(name='u', grid=grid)
    assert(grid.stepping_dim in u.indices)

    u2 = TimeFunction(name='u2', grid=grid, save=nt)
    assert(time in u2.indices)

    factor = 4
    time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
    usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                         time_dim=time_subsampled)
    assert(time_subsampled in usave.indices)

    eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.), Eq(usave, u)]
    op = Operator(eqns)
    op.apply(t=nt)
    assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
    assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
    assert np.all([np.allclose(usave.data[i], i*factor)
                  for i in range((nt+factor-1)//factor)])


@skipif_yask
def test_conditional_as_expr():
    nt = 19
    grid = Grid(shape=(11, 11))
    time = grid.time_dim

    u = TimeFunction(name='u', grid=grid)
    assert(grid.stepping_dim in u.indices)

    u2 = TimeFunction(name='u2', grid=grid, save=nt)
    assert(time in u2.indices)

    factor = 4
    time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
    usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                         time_dim=time_subsampled)
    assert(time_subsampled in usave.indices)

    eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
            Eq(usave, time_subsampled * u)]
    op = Operator(eqns)
    op.apply(t=nt)
    assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
    assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
    assert np.all([np.allclose(usave.data[i], i*factor*i)
                  for i in range((nt+factor-1)//factor)])


@skipif_yask
def test_conditional_shifted():
    nt = 19
    grid = Grid(shape=(11, 11))
    time = grid.time_dim

    u = TimeFunction(name='u', grid=grid)
    assert(grid.stepping_dim in u.indices)

    u2 = TimeFunction(name='u2', grid=grid, save=nt)
    assert(time in u2.indices)

    factor = 4
    time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
    usave = TimeFunction(name='usave', grid=grid, save=2, time_dim=time_subsampled)
    assert(time_subsampled in usave.indices)

    t_sub_shift = Constant(name='t_sub_shift', dtype=np.int32)

    eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
            Eq(usave.subs(time_subsampled, time_subsampled - t_sub_shift), u)]
    op = Operator(eqns)

    # Starting at time_m=10, so time_subsampled - t_sub_shift is in range
    op.apply(time_m=10, time_M=nt, t_sub_shift=3)
    assert np.all(np.allclose(u.data[0], 8))
    assert np.all([np.allclose(u2.data[i], i - 10) for i in range(10, nt)])
    assert np.all([np.allclose(usave.data[i], 2+i*factor) for i in range(2)])

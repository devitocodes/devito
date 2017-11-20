from devito import SubsampledDimension, Grid, TimeFunction, Eq, Operator
from devito.tools import pprint


def test_subsampled_dimension():
    nt = 10
    grid = Grid(shape=(11, 11))
    x, y = grid.dimensions
    time = grid.time_dim
    t = grid.stepping_dim
    time_subsampled = SubsampledDimension('t_sub', parent=time, factor=4)
    u = TimeFunction(name='u', grid=grid)
    u2 = TimeFunction(name='u2', grid=grid, save=nt)
    assert(t in u.indices)
    u_s = TimeFunction(name='u_s', grid=grid, time_dim=time_subsampled)
    assert(time_subsampled in u_s.indices)
    fwd_eqn = Eq(u.indexed[t+1, x, y], u.indexed[t, x, y] + 1.)
    fwd_eqn_2 = Eq(u2.indexed[time+1, x, y], u2.indexed[time, x, y] + 1.)
    save_eqn = Eq(u_s, u)
    #fwd_op = Operator([fwd_eqn])
    fwd_op = Operator([fwd_eqn, fwd_eqn_2, save_eqn])
    pprint(fwd_op)
    print(fwd_op)

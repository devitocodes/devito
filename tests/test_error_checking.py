import pytest

from devito import Grid, Function, TimeFunction, Eq, Operator, switchconfig
from devito.exceptions import ExecutionError


@switchconfig(safe_math=True)
def test_stability():
    grid = Grid(shape=(10, 10))

    f = Function(name='f', grid=grid, space_order=2)
    u = TimeFunction(name='u', grid=grid, space_order=2)

    eq = Eq(u.forward, u/f)

    op = Operator(eq, opt=('advanced', {'errctl': 'max'}))

    u.data[:] = 1.

    with pytest.raises(ExecutionError):
        op.apply(time_M=200, dt=.1)

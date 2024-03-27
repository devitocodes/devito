import pytest
import re

from devito import Grid, Function, TimeFunction, Eq, Operator, switchconfig
from devito.exceptions import ExecutionError
from devito.arch.compiler import sniff_compiler_version

@switchconfig(safe_math=True)
@pytest.mark.parametrize("expr", [
    'u/f',
    '(u + v)/f',
])
def test_stability(expr):
    grid = Grid(shape=(10, 10))

    f = Function(name='f', grid=grid, space_order=2)  # noqa
    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid, space_order=2)

    eq = Eq(u.forward, eval(expr))

    op = Operator(eq, opt=('advanced', {'errctl': 'max'}))

    u.data[:] = 1.
    v.data[:] = 2.

    with pytest.raises(ExecutionError):
        op.apply(time_M=200, dt=.1)

@pytest.mark.parametrize("cc", [
    "doesn'texist",
    "/root/doesn'texist",
])
def test_sniff_compiler_version(cc):
    with pytest.raises(SystemExit) as e:
        sniff_compiler_version(cc)

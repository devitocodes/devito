from functools import reduce
from operator import mul

import pytest
import numpy as np
from devito import (Grid, Function, TimeFunction, Eq, Operator, configuration,
                    silencio, ruido)

pytestmark = pytest.mark.skipif(configuration['backend'] == 'yask' or
                                configuration['backend'] == 'ops',
                                reason="testing is currently restricted")


@silencio(log_level='DEBUG')
@pytest.mark.parametrize("shape,expected", [
    ((30, 30), 13),
    ((30, 30, 30), 17)
])
def test_at_is_actually_working(shape, expected):
    """
    Check that autotuning is actually running when switched on,
    in both 2D and 3D operators.
    """
    grid = Grid(shape=shape)
    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)

    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockinner': True, 'blockalways': True}))

    # Run with whatever `configuration` says (by default, basic+preemptive)
    op(infield=infield, outfield=outfield, autotune=True)
    assert op._state['autotuning'][-1]['runs'] == 4
    assert op._state['autotuning'][-1]['tpr'] == 1

    # Now try `aggressive` autotuning
    configuration['autotuning'] = 'aggressive'
    op(infield=infield, outfield=outfield, autotune=True)
    assert op._state['autotuning'][-1]['runs'] == expected
    assert op._state['autotuning'][-1]['tpr'] == 1
    configuration['autotuning'] = configuration._defaults['autotuning']

    # Try again, but using the Operator API directly
    op(infield=infield, outfield=outfield, autotune='aggressive')
    assert op._state['autotuning'][-1]['runs'] == expected
    assert op._state['autotuning'][-1]['tpr'] == 1

    # Similar to above
    op(infield=infield, outfield=outfield, autotune=('aggressive', 'preemptive'))
    assert op._state['autotuning'][-1]['runs'] == expected
    assert op._state['autotuning'][-1]['tpr'] == 1


@silencio(log_level='DEBUG')
def test_timesteps_per_at_run():
    """
    Check that each autotuning run (ie with a given block shape) takes
    ``autotuning.core.options['at_squeezer']`` timesteps, for an operator
    performing the increment ``a[t + timeorder, ...] = f(a[t, ...], ...)``.
    """
    from devito.core.autotuning import options

    shape = (30, 30, 30)
    grid = Grid(shape=shape)
    x, y, z = grid.dimensions
    t = grid.stepping_dim

    # Function
    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockalways': True}))
    op(infield=infield, outfield=outfield, autotune=True)
    assert op._state['autotuning'][-1]['runs'] == 4
    assert op._state['autotuning'][-1]['tpr'] == 1

    # TimeFunction with increasing time order; increasing the time order
    # shouldn't affect how many iterations the autotuner is gonna run
    for to in [1, 2, 4]:
        infield = TimeFunction(name='infield', grid=grid, time_order=to)
        infield.data[:] = np.arange(reduce(mul, infield.shape),
                                    dtype=np.int32).reshape(infield.shape)
        outfield = TimeFunction(name='outfield', grid=grid, time_order=to)
        stencil = Eq(outfield[t + to, x, y, z],
                     outfield.indexify() + infield.indexify()*3.0)
        op = Operator(stencil, dle=('blocking', {'blockalways': True}))
        op(infield=infield, outfield=outfield, time=20, autotune=True)
        assert op._state['autotuning'][-1]['runs'] == 4
        assert op._state['autotuning'][-1]['tpr'] == options['at_squeezer']+1


@ruido(profiling='advanced')
def test_nondestructive_forward():
    """Test autotuning in non-destructive mode."""
    grid = Grid(shape=(64, 64, 64))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.))
    summary = op.apply(time=100, autotune=('basic', 'runtime'))

    # At is expected to have attempted 6 block shapes
    assert op._state['autotuning'][0]['runs'] == 6

    # AT is expected to have executed 30 timesteps
    assert summary['section0'].itershapes[0][0] == 101-30
    assert np.all(f.data[0] == 100)
    assert np.all(f.data[1] == 101)


@ruido(profiling='advanced')
def test_nondestructive_backward():
    """Test autotuning in non-destructive mode."""
    grid = Grid(shape=(64, 64, 64))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.backward, f + 1.))
    summary = op.apply(time=101, autotune=('basic', 'runtime'))

    # At is expected to have attempted 6 block shapes
    assert op._state['autotuning'][0]['runs'] == 6

    # AT is expected to have executed 30 timesteps
    assert summary['section0'].itershapes[0][0] == 101-30
    assert np.all(f.data[0] == 101)
    assert np.all(f.data[1] == 100)


def test_tti_aggressive():
    from test_dse import tti_operator
    op = tti_operator(dse='aggressive').op_fwd(kernel='centered', save=False)
    op.apply(time=0, autotune='aggressive')
    assert op._state['autotuning'][0]['runs'] == 33

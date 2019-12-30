from functools import reduce
from operator import mul

import pytest
import numpy as np
from unittest.mock import patch

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, Operator, configuration,
                    switchconfig)
from devito.data import LEFT

pytestmark = skipif(['yask', 'ops'], whole_module=True)

# All core-specific imports *must* be avoided if `backend != core`, otherwise
# a backend reinitialization would be triggered via `devito/core/.__init__.py`,
# thus invalidating all of the future tests. This is guaranteed by the
# `pytestmark` above
from devito.core.autotuning import options  # noqa


@switchconfig(log_level='DEBUG')
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
    op = Operator(stencil, dle=('blocking', {'openmp': False,
                                             'blockinner': True,
                                             'blockalways': True}))

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


@switchconfig(log_level='DEBUG')
def test_timesteps_per_at_run():
    """
    Check that each autotuning run (ie with a given block shape) takes
    ``autotuning.core.options['squeezer']`` timesteps, for an operator
    performing the increment ``a[t + timeorder, ...] = f(a[t, ...], ...)``.
    """
    shape = (30, 30, 30)
    grid = Grid(shape=shape)
    x, y, z = grid.dimensions
    t = grid.stepping_dim

    # Function
    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'openmp': False, 'blockalways': True}))
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
        op = Operator(stencil, dle=('blocking', {'openmp': False, 'blockalways': True}))
        op(infield=infield, outfield=outfield, time=20, autotune=True)
        assert op._state['autotuning'][-1]['runs'] == 4
        assert op._state['autotuning'][-1]['tpr'] == options['squeezer'] + 1


@switchconfig(profiling='advanced')
def test_mode_runtime_forward():
    """Test autotuning in runtime mode."""
    grid = Grid(shape=(96, 96, 96))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.), dle=('advanced', {'openmp': False}))
    summary = op.apply(time=100, autotune=('basic', 'runtime'))

    # AT is expected to have attempted 6 block shapes
    assert op._state['autotuning'][0]['runs'] == 6

    # AT is expected to have executed 30 timesteps
    assert summary[('section0', None)].itershapes[0][0] == 101-30
    assert np.all(f.data[0] == 100)
    assert np.all(f.data[1] == 101)


@switchconfig(profiling='advanced')
def test_mode_runtime_backward():
    """Test autotuning in runtime mode."""
    grid = Grid(shape=(96, 96, 96))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.backward, f + 1.), dle=('advanced', {'openmp': False}))
    summary = op.apply(time=101, autotune=('basic', 'runtime'))

    # AT is expected to have attempted 6 block shapes
    assert op._state['autotuning'][0]['runs'] == 6

    # AT is expected to have executed 30 timesteps
    assert summary[('section0', None)].itershapes[0][0] == 101-30
    assert np.all(f.data[0] == 101)
    assert np.all(f.data[1] == 100)


@switchconfig(profiling='advanced')
def test_mode_destructive():
    """Test autotuning in destructive mode."""
    grid = Grid(shape=(96, 96, 96))
    f = TimeFunction(name='f', grid=grid, time_order=0)

    op = Operator(Eq(f, f + 1.), dle=('advanced', {'openmp': False}))
    op.apply(time=100, autotune=('basic', 'destructive'))

    # AT is expected to have executed 30 timesteps (6 block shapes, 5 timesteps each)
    # The operator runs for 101 timesteps
    # So, overall, f.data[0] is incremented 131 times
    assert np.all(f.data == 131)


def test_blocking_only():
    grid = Grid(shape=(96, 96, 96))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.), dle=('advanced', {'openmp': False}))
    op.apply(time=0, autotune=True)

    assert op._state['autotuning'][0]['runs'] == 6
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 2
    assert 'nthreads' not in op._state['autotuning'][0]['tuned']


def test_mixed_blocking_nthreads():
    grid = Grid(shape=(96, 96, 96))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.), dle=('advanced', {'openmp': True}))
    op.apply(time=100, autotune=True)

    assert op._state['autotuning'][0]['runs'] == 6
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 3
    assert 'nthreads' in op._state['autotuning'][0]['tuned']


def test_tti_aggressive():
    from test_dse import TestTTI
    wave_solver = TestTTI().tti_operator(dse='aggressive')
    op = wave_solver.op_fwd(kernel='centered')
    op.apply(time=0, autotune='aggressive')
    assert op._state['autotuning'][0]['runs'] == 28


@switchconfig(develop_mode=False)
@patch("devito.dle.parallelizer.Ompizer.COLLAPSE_NCORES", 1)
def test_discarding_runs():
    grid = Grid(shape=(64, 64, 64))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.), dle=('advanced', {'openmp': True}))
    op.apply(time=100, nthreads=4, autotune='aggressive')

    assert op._state['autotuning'][0]['runs'] == 18
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 3
    assert op._state['autotuning'][0]['tuned']['nthreads'] == 4

    # With 1 < 4 threads, the AT eventually tries many more combinations
    op.apply(time=100, nthreads=1, autotune='aggressive')

    assert op._state['autotuning'][1]['runs'] == 25
    assert op._state['autotuning'][1]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][1]['tuned']) == 3
    assert op._state['autotuning'][1]['tuned']['nthreads'] == 1


@skipif('nompi')
@pytest.mark.parallel(mode=[(2, 'diag'), (2, 'full')])
def test_at_w_mpi():
    """Make sure autotuning works in presence of MPI. MPI ranks work
    in isolation to determine the best block size, locally."""
    grid = Grid(shape=(8, 8))
    t = grid.stepping_dim
    x, y = grid.dimensions

    f = TimeFunction(name='f', grid=grid, time_order=1)
    f.data_with_halo[:] = 1.

    eq = Eq(f.forward, f[t, x-1, y] + f[t, x+1, y])
    op = Operator(eq, dle=('advanced', {'openmp': False, 'blockinner': True}))

    op.apply(time=-1, autotune=('basic', 'runtime'))
    # Nothing really happened, as not enough timesteps
    assert np.all(f.data_ro_domain[0] == 1.)
    assert np.all(f.data_ro_domain[1] == 1.)

    # The 'destructive' mode writes directly to `f` for whatever timesteps required
    # to perform the autotuning. Eventually, the result is complete garbage; note
    # also that this autotuning mode disables the halo exchanges
    op.apply(time=-1, autotune=('basic', 'destructive'))
    assert np.all(f._data_ro_with_inhalo.sum() == 904)

    # Check the halo hasn't been touched during AT
    glb_pos_map = grid.distributor.glb_pos_map
    if LEFT in glb_pos_map[x]:
        assert np.all(f._data_ro_with_inhalo[:, -1] == 1)
    else:
        assert np.all(f._data_ro_with_inhalo[:, 0] == 1)

    # Finally, try running w/o AT, just to be sure nothing was broken
    f.data_with_halo[:] = 1.
    op.apply(time=2)
    if LEFT in glb_pos_map[x]:
        assert np.all(f.data_ro_domain[1, 0] == 5.)
        assert np.all(f.data_ro_domain[1, 1] == 7.)
        assert np.all(f.data_ro_domain[1, 2:4] == 8.)
    else:
        assert np.all(f.data_ro_domain[1, 4:6] == 8)
        assert np.all(f.data_ro_domain[1, 6] == 7)
        assert np.all(f.data_ro_domain[1, 7] == 5)


def test_multiple_blocking():
    """
    Test that if there are more than one blocked Iteration nests, then
    the autotuner works "incrementally" -- it starts determining the best block
    shape for the first Iteration nest, then it moves on to the second one,
    then the third, etc. IOW, the autotuner must not be attempting the
    cartesian product of all possible block shapes across the various
    blocked nests.
    """
    grid = Grid(shape=(96, 96, 96))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid)

    op = Operator([Eq(u.forward, u + 1), Eq(v.forward, u.forward.dx2 + v + 1)],
                  dle=('blocking', {'openmp': False}))

    # First of all, make sure there are indeed two different loop nests
    assert 'bf0' in op._func_table
    assert 'bf1' in op._func_table

    # 'basic' mode
    op.apply(time_M=0, autotune='basic')
    assert op._state['autotuning'][0]['runs'] == 12  # 6 for each Iteration nest
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 4

    # 'aggressive' mode
    op.apply(time_M=0, autotune='aggressive')
    assert op._state['autotuning'][1]['runs'] == 60
    assert op._state['autotuning'][1]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][1]['tuned']) == 4

    # With OpenMP, we tune over one more argument (`nthreads`), though the AT
    # will only attempt one value
    op = Operator([Eq(u.forward, u + 1), Eq(v.forward, u.forward.dx2 + v + 1)],
                  dle=('blocking', {'openmp': True}))
    op.apply(time_M=0, autotune='basic')
    assert op._state['autotuning'][0]['runs'] == 12
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 5


def test_hierarchical_blocking():
    grid = Grid(shape=(64, 64, 64))

    u = TimeFunction(name='u', grid=grid, space_order=2)

    op = Operator(Eq(u.forward, u + 1), dle=('blocking', {'openmp': False,
                                                          'blocklevels': 2}))

    # 'basic' mode
    op.apply(time_M=0, autotune='basic')
    assert op._state['autotuning'][0]['runs'] == 10
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 4

    # 'aggressive' mode
    op.apply(time_M=0, autotune='aggressive')
    assert op._state['autotuning'][1]['runs'] == 38
    assert op._state['autotuning'][1]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][1]['tuned']) == 4


@switchconfig(platform='cpu64-dummy')  # To fix the core count
def test_multiple_threads():
    """
    Test autotuning when different ``num_threads`` for a given OpenMP parallel
    region are attempted.
    """
    grid = Grid(shape=(96, 96, 96))

    v = TimeFunction(name='v', grid=grid)

    op = Operator(Eq(v.forward, v + 1), dle=('blocking', {'openmp': True}))
    op.apply(time_M=0, autotune='max')
    assert op._state['autotuning'][0]['runs'] == 60  # Would be 30 with `aggressive`
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 3


@switchconfig(platform='knl7210')  # To trigger nested parallelism
def test_nested_nthreads():
    grid = Grid(shape=(96, 96, 96))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.), dle=('advanced', {'openmp': True}))
    op.apply(time=10, autotune=True)

    assert op._state['autotuning'][0]['runs'] == 6
    assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
    assert len(op._state['autotuning'][0]['tuned']) == 3
    assert 'nthreads' in op._state['autotuning'][0]['tuned']
    # No tuning for the nested level
    assert 'nthreads_nested' not in op._state['autotuning'][0]['tuned']


def test_few_timesteps():
    grid = Grid(shape=(16, 16, 16))

    save = 3
    assert save < options['squeezer']
    v = TimeFunction(name='v', grid=grid, save=save)

    # Try forward propagation first
    op = Operator(Eq(v.forward, v + 1))
    op.apply(autotune=True)
    assert op._state['autotuning'][0]['runs'] == 2
    assert op._state['autotuning'][0]['tpr'] == 2  # Induced by `save`

    # Now try backward propagation
    op = Operator(Eq(v.backward, v + 1))
    op.apply(autotune=True)
    assert op._state['autotuning'][0]['runs'] == 2
    assert op._state['autotuning'][0]['tpr'] == 2  # Induced by `save`

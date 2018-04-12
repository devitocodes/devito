from __future__ import absolute_import

from functools import reduce
from operator import mul
try:
    from StringIO import StringIO
except ImportError:
    # Python3 compatibility
    from io import StringIO

import pytest
from conftest import skipif_yask

import numpy as np

from devito import Grid, Function, TimeFunction, Eq, Operator, configuration, silencio
from devito.logger import logger, logging


@silencio(log_level='DEBUG')
@skipif_yask
@pytest.mark.parametrize("shape,expected", [
    ((30, 30), 17),
    ((30, 30, 30), 21)
])
def test_at_is_actually_working(shape, expected):
    """
    Check that autotuning is actually running when switched on,
    in both 2D and 3D operators.
    """
    grid = Grid(shape=shape)

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)

    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockinner': True, 'blockalways': True}))

    # Expected 3 AT attempts for the given shape
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 4

    # Now try the same with aggressive autotuning, which tries 9 more cases
    configuration.core['autotuning'] = 'aggressive'
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == expected
    configuration.core['autotuning'] = configuration.core._defaults['autotuning']

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()


@silencio(log_level='DEBUG')
@skipif_yask
def test_timesteps_per_at_run():
    """
    Check that each autotuning run (ie with a given block shape) takes
    ``autotuning.core.options['at_squeezer']`` timesteps, for an operator
    performing the increment ``a[t + timeorder, ...] = f(a[t, ...], ...)``.
    """
    from devito.core.autotuning import options

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)

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
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 4
    assert all('in 1 time steps' in i for i in out)
    buffer.truncate(0)

    # TimeFunction with increasing time order; increasing the time order
    # shouldn't affect how many iterations the autotuner is gonna run
    for to in [1, 2, 4]:
        infield = TimeFunction(name='infield', grid=grid, time_order=to)
        infield.data[:] = np.arange(reduce(mul, infield.shape),
                                    dtype=np.int32).reshape(infield.shape)
        outfield = TimeFunction(name='outfield', grid=grid, time_order=to)
        stencil = Eq(outfield.indexed[t + to, x, y, z],
                     outfield.indexify() + infield.indexify()*3.0)
        op = Operator(stencil, dle=('blocking', {'blockalways': True}))
        op(infield=infield, outfield=outfield, t=2, autotune=True)
        out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
        assert len(out) == 4
        assert all('in %d time steps' % options['at_squeezer'] in i for i in out)
        buffer.truncate(0)

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()

from __future__ import absolute_import

from functools import reduce
from operator import mul
try:
    from StringIO import StringIO
except ImportError:
    # Python3 compatibility
    from io import StringIO

import pytest

import numpy as np

from devito import Eq, DenseData, TimeData, Operator, t, x, y, z, configuration
from devito.logger import logger, logging, set_log_level
from devito.core.autotuning import options


@pytest.mark.parametrize("shape,expected", [
    ((30, 30), 12),
    ((30, 30, 30), 16)
])
def test_at_is_actually_working(shape, expected):
    """
    Check that autotuning is actually running when switched on,
    in both 2D and 3D operators.
    """

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)
    set_log_level('DEBUG')

    infield = DenseData(name='infield', shape=shape, dtype=np.int32)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = DenseData(name='outfield', shape=shape, dtype=np.int32)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockinner': True, 'blockalways': True}))

    # Expected 3 AT attempts for the given shape
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 3

    # Now try the same with aggressive autotuning, which tries 9 more cases
    configuration['autotuning'] = 'aggressive'
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == expected
    configuration['autotuning'] = configuration._defaults['autotuning']

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()
    set_log_level('INFO')


def test_timesteps_per_at_run():
    """
    Check that each autotuning run (ie with a given block shape) takes
    ``autotuning.options['at_squeezer'] - data.time_order`` timesteps.
    in an operator performing an increment such as
    ``a[t + timeorder, ...] = f(a[t, ...], ...)``.
    """

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)
    set_log_level('DEBUG')

    shape = (30, 30, 30)

    # DenseData
    infield = DenseData(name='infield', shape=shape, dtype=np.int32)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = DenseData(name='outfield', shape=shape, dtype=np.int32)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockalways': True}))
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 3
    assert all('in 1 time steps' in i for i in out)
    buffer.truncate(0)

    # TimeData with increasing time order
    for to in [1, 2, 4]:
        infield = TimeData(name='infield', shape=shape, dtype=np.int32, time_order=to)
        infield.data[:] = np.arange(reduce(mul, infield.shape),
                                    dtype=np.int32).reshape(infield.shape)
        outfield = TimeData(name='outfield', shape=shape, dtype=np.int32, time_order=to)
        stencil = Eq(outfield.indexed[t + to, x, y, z],
                     outfield.indexify() + infield.indexify()*3.0)
        op = Operator(stencil, dle=('blocking', {'blockalways': True}))
        op(infield=infield, outfield=outfield, autotune=True)
        out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
        expected = options['at_squeezer'] - to
        assert len(out) == 3
        assert all('in %d time steps' % expected in i for i in out)
        buffer.truncate(0)

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()
    set_log_level('INFO')

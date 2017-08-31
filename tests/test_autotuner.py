from __future__ import absolute_import

from functools import reduce
from operator import mul
try:
    from StringIO import StringIO
except ImportError:
    # Python3 compatibility
    from io import StringIO

import numpy as np
from sympy import Eq

from devito import DenseData, Operator, configuration
from devito.logger import logger, logging, set_log_level


def test_at_is_actually_working():
    """
    Check that autotuning is actually running when switched on.
    """

    buffer = StringIO()
    temporary_handler = logging.StreamHandler(buffer)
    logger.addHandler(temporary_handler)
    set_log_level('DEBUG')

    shape = (30, 30, 30)

    infield = DenseData(name='infield', shape=shape, dtype=np.int32)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = DenseData(name='outfield', shape=shape, dtype=np.int32)
    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=('blocking', {'blockalways': True}))

    # Expected 3 AT attempts for the given shape
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 3

    # Now try the same with aggressive autotuning, which tries 9 more cases
    configuration['autotuning'] = 'aggressive'
    op(infield=infield, outfield=outfield, autotune=True)
    out = [i for i in buffer.getvalue().split('\n') if 'AutoTuner:' in i]
    assert len(out) == 12
    configuration['autotuning'] = configuration._defaults['autotuning']

    logger.removeHandler(temporary_handler)

    temporary_handler.flush()
    temporary_handler.close()
    buffer.flush()
    buffer.close()
    set_log_level('INFO')

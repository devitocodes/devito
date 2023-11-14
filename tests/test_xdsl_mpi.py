"""
DEVITO_MPI=0 DEVITO_LOGGING=DEBUG pytest -m "not parallel" tests/test_xdsl_*
DEVITO_MPI=1 DEVITO_LOGGING=DEBUG pytest -m parallel tests/test_xdsl_mpi.py
"""

import pytest

from devito import (Grid, TimeFunction, Eq, norm, inner,
                    switchconfig, XDSLOperator)
from devito.data import LEFT, RIGHT


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32, 32))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = XDSLOperator(Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + 1))
        op.apply(time=1)

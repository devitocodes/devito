"""
DEVITO_MPI=0 DEVITO_LOGGING=DEBUG pytest -m "not parallel" tests/test_xdsl_*
DEVITO_MPI=1 DEVITO_LOGGING=DEBUG pytest -m parallel tests/test_xdsl_mpi.py
"""

from conftest import skipif
import pytest
import numpy as np

from devito import (Grid, TimeFunction, Eq, norm, XDSLOperator)


pytestmark = skipif(['nompi'], whole_module=True)


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32, 32))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = XDSLOperator(Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + 1))
        op.apply(time=2)
        assert np.isclose(norm(f), 515.9845, rtol=1e-4)

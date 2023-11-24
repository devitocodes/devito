"""
DEVITO_MPI=0 DEVITO_LOGGING=DEBUG pytest -m "not parallel" tests/test_xdsl_*
DEVITO_MPI=1 DEVITO_LOGGING=DEBUG pytest -m parallel tests/test_xdsl_mpi.py
"""

from conftest import skipif
import pytest
import numpy as np

from devito import (Grid, TimeFunction, Eq, norm, XDSLOperator, solve, Operator)


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

    @pytest.mark.parallel(mode=[2])
    @pytest.mark.parametrize('shape', [(101, 101, 101), (202, 10, 45)])
    @pytest.mark.parametrize('so', [2, 4, 8])
    @pytest.mark.parametrize('to', [2])
    @pytest.mark.parametrize('nt', [10, 20, 100])
    def test_acoustic_3D(self, shape, so, to, nt):

        grid = Grid(shape=shape)
        dt = 0.0001

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

        pde = u.dt2 - u.laplace
        eq0 = solve(pde, u.forward)

        stencil = Eq(u.forward, eq0)
        u.data[:, :, :] = 0
        u.data[:, 40:50, 40:50] = 1

        # Devito Operator
        op = Operator([stencil])
        op.apply(time=nt, dt=dt)
        devito_norm = norm(u)

        u.data[:, :, :] = 0
        u.data[:, 40:50, 40:50] = 1

        # XDSL Operator
        xdslop = XDSLOperator([stencil])
        xdslop.apply(time=nt, dt=dt)
        xdsl_norm = norm(u)

        assert np.isclose(devito_norm, xdsl_norm, rtol=1e-04).all()

from math import floor

import pytest
import numpy as np
import scipy.sparse

from devito import Grid, TimeFunction, Eq, Operator, MatrixSparseTimeFunction


class TestMatrixSparseTimeFunction(object):

    def _precompute_linear_interpolation(self, points, grid, origin):
        """ Sample precompute function that, given point and grid information
            precomputes gridpoints and coefficients according to a linear
            scheme to be used in PrecomputedSparseFunction.
        """
        gridpoints = [
            tuple(
                floor((point[i] - origin[i]) / grid.spacing[i]) for i in range(len(point))
            )
            for point in points
        ]

        coefficients = np.zeros((len(points), 2, 2))
        for i, point in enumerate(points):
            for d in range(grid.dim):
                coefficients[i, d, 0] = (
                    (gridpoints[i][d] + 1) * grid.spacing[d] - point[d]
                ) / grid.spacing[d]
                coefficients[i, d, 1] = (
                    point[d] - gridpoints[i][d] * grid.spacing[d]
                ) / grid.spacing[d]
        return gridpoints, coefficients

    def test_precomputed_interpolation(self):
        shape = (101, 101)
        points = [(0.05, 0.9), (0.01, 0.8), (0.07, 0.84)]
        origin = (0, 0)

        grid = Grid(shape=shape, origin=origin)
        x, y = grid.dimensions
        r = 2

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=nt, time_order=0)
        for it in range(nt):
            m.data[it, :] = it

        gridpoints, coefficients = self._precompute_linear_interpolation(
            points, grid, origin
        )

        mat = scipy.sparse.eye(len(points), dtype=np.float32)

        sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=mat, nt=nt)

        sf.gridpoints.data[:] = gridpoints
        sf.interpolation_coefficients[x].data[:] = coefficients[:, 0, :]
        sf.interpolation_coefficients[y].data[:] = coefficients[:, 1, :]

        eqn = sf.interpolate(m)
        op = Operator(eqn)

        sf.manual_scatter()

        # args = op.arguments(time_m=0, time_M=9)
        op(time_m=0, time_M=9)

        sf.manual_gather()

        for it in range(nt):
            assert np.all(sf.data[it, :] == pytest.approx(it))

    def test_precomputed_interpolation_empty(self):
        shape = (101, 101)
        origin = (0, 0)

        grid = Grid(shape=shape, origin=origin)
        x, y = grid.dimensions
        #  because we interpolate across 2 neighbouring points in each dimension
        r = 2

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=nt, time_order=0)
        for it in range(nt):
            m.data[it, :] = it

        mat = scipy.sparse.coo_matrix((0, 0), dtype=np.float32)
        sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=mat, nt=nt)

        eqn = sf.interpolate(m)
        op = Operator(eqn)

        sf.manual_scatter()
        op(time_m=0, time_M=9)
        sf.manual_gather()
        # There are no receivers, so nothing to assert here

    def test_precomputed2(self):
        shape = (101, 101)
        grid = Grid(shape=shape)
        x, y = grid.dimensions
        r = 2  # Constant for linear interpolation
        #  because we interpolate across 2 neighbouring points in each dimension

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=None, time_order=1)

        m.data[:] = 0.0
        m.data[:, 40, 40] = 1.0

        matrix = scipy.sparse.eye(1, dtype=np.float32)
        sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=matrix, nt=nt)

        # Lookup the exact point
        sf.gridpoints.data[0, 0] = 40
        sf.gridpoints.data[0, 1] = 40
        sf.interpolation_coefficients[x].data[0, 0] = 1.0
        sf.interpolation_coefficients[x].data[0, 1] = 2.0
        sf.interpolation_coefficients[y].data[0, 0] = 1.0
        sf.interpolation_coefficients[y].data[0, 1] = 2.0
        sf.data[:] = 0.0

        step = [Eq(m.forward, m)]
        interp = sf.interpolate(m)
        op = Operator(step + interp)

        sf.manual_scatter()
        op(time_m=0, time_M=0)
        sf.manual_gather()

        assert sf.data[0, 0] == 1.0

    def test_precomputed_subpoints(self):
        shape = (101, 101)
        grid = Grid(shape=shape)
        x, y = grid.dimensions
        r = 2  # Constant for linear interpolation
        #  because we interpolate across 2 neighbouring points in each dimension

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=None, time_order=1)

        m.data[:] = 0.0
        m.data[:, 40, 40] = 1.0

        # Two-location source with 2 coefficients the same
        matrix = scipy.sparse.coo_matrix(np.array([[1], [1]], dtype=np.float32))

        sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=matrix, nt=nt)

        # Lookup the exact point
        sf.gridpoints.data[0, 0] = 40
        sf.gridpoints.data[0, 1] = 40
        sf.interpolation_coefficients[x].data[0, 0] = 1.0
        sf.interpolation_coefficients[x].data[0, 1] = 2.0
        sf.interpolation_coefficients[y].data[0, 0] = 1.0
        sf.interpolation_coefficients[y].data[0, 1] = 2.0
        sf.gridpoints.data[1, 0] = 39
        sf.gridpoints.data[1, 1] = 39
        sf.interpolation_coefficients[x].data[1, 0] = 1.0
        sf.interpolation_coefficients[x].data[1, 1] = 2.0
        sf.interpolation_coefficients[y].data[1, 0] = 1.0
        sf.interpolation_coefficients[y].data[1, 1] = 2.0
        sf.data[:] = 0.0

        step = [Eq(m.forward, m)]
        interp = sf.interpolate(m)
        op = Operator(step + interp)

        sf.manual_scatter()
        op(time_m=0, time_M=0)
        sf.manual_gather()

        assert sf.data[0, 0] == 5.0

    def test_precomputed_subpoints_inject(self):
        shape = (101, 101)
        grid = Grid(shape=shape)
        x, y = grid.dimensions
        r = 2  # Constant for linear interpolation
        #  because we interpolate across 2 neighbouring points in each dimension

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=None, time_order=1)

        m.data[:] = 0.0
        m.data[:, 40, 40] = 1.0

        # Single two-component source with coefficients both +1
        matrix = scipy.sparse.coo_matrix(np.array([[1], [1]], dtype=np.float32))

        sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=matrix, nt=nt)

        # Lookup the exact point
        sf.gridpoints.data[0, 0] = 40
        sf.gridpoints.data[0, 1] = 40
        sf.interpolation_coefficients[x].data[0, 0] = 1.0
        sf.interpolation_coefficients[x].data[0, 1] = 2.0
        sf.interpolation_coefficients[y].data[0, 0] = 1.0
        sf.interpolation_coefficients[y].data[0, 1] = 2.0
        sf.gridpoints.data[1, 0] = 39
        sf.gridpoints.data[1, 1] = 39
        sf.interpolation_coefficients[x].data[1, 0] = 1.0
        sf.interpolation_coefficients[x].data[1, 1] = 2.0
        sf.interpolation_coefficients[y].data[1, 0] = 1.0
        sf.interpolation_coefficients[y].data[1, 1] = 2.0
        sf.data[0, 0] = 1.0

        step = [Eq(m.forward, m)]
        inject = sf.inject(field=m.forward, expr=sf)
        op = Operator(step + inject)

        sf.manual_scatter()
        op(time_m=0, time_M=0)
        sf.manual_gather()

        assert m.data[1, 40, 40] == 6.0  # 1 + 1 + 4
        assert m.data[1, 40, 41] == 2.0
        assert m.data[1, 41, 40] == 2.0
        assert m.data[1, 41, 41] == 4.0
        assert m.data[1, 39, 39] == 1.0
        assert m.data[1, 39, 40] == 2.0
        assert m.data[1, 40, 39] == 2.0

    def test_precomputed_subpoints_inject_dt2(self):
        shape = (101, 101)
        grid = Grid(shape=shape)
        x, y = grid.dimensions
        r = 2  # Constant for linear interpolation
        #  because we interpolate across 2 neighbouring points in each dimension

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=None, time_order=1)

        m.data[:] = 0.0
        m.data[:, 40, 40] = 1.0

        matrix = scipy.sparse.coo_matrix(np.array([[1], [1]], dtype=np.float32))

        sf = MatrixSparseTimeFunction(
            name="s", grid=grid, r=r, matrix=matrix, nt=nt, time_order=2
        )

        # Lookup the exact point
        sf.gridpoints.data[0, 0] = 40
        sf.gridpoints.data[0, 1] = 40
        sf.interpolation_coefficients[x].data[0, 0] = 1.0
        sf.interpolation_coefficients[x].data[0, 1] = 2.0
        sf.interpolation_coefficients[y].data[0, 0] = 1.0
        sf.interpolation_coefficients[y].data[0, 1] = 2.0
        sf.gridpoints.data[1, 0] = 39
        sf.gridpoints.data[1, 1] = 39
        sf.interpolation_coefficients[x].data[1, 0] = 1.0
        sf.interpolation_coefficients[x].data[1, 1] = 2.0
        sf.interpolation_coefficients[y].data[1, 0] = 1.0
        sf.interpolation_coefficients[y].data[1, 1] = 2.0

        # Single timestep, -0.5*1e-6, so that with dt=0.001, the .dt2 == 1 at t=1
        sf.data[1, 0] = -5e-7

        step = [Eq(m.forward, m)]
        inject = sf.inject(field=m.forward, expr=sf.dt2)
        op = Operator(step + inject)

        sf.manual_scatter()
        op(time_m=1, time_M=1, dt=0.001)
        sf.manual_gather()

        assert m.data[0, 40, 40] == pytest.approx(6.0)  # 1 + 1 + 4
        assert m.data[0, 40, 41] == pytest.approx(2.0)
        assert m.data[0, 41, 40] == pytest.approx(2.0)
        assert m.data[0, 41, 41] == pytest.approx(4.0)
        assert m.data[0, 39, 39] == pytest.approx(1.0)
        assert m.data[0, 39, 40] == pytest.approx(2.0)
        assert m.data[0, 40, 39] == pytest.approx(2.0)

    @pytest.mark.parallel(mode=4)
    def test_mpi(self):
        # Shape chosen to get a source in multiple ranks
        shape = (91, 91)
        grid = Grid(shape=shape)
        x, y = grid.dimensions
        #  because we interpolate across 2 neighbouring points in each dimension
        r = 2

        nt = 10

        # NOTE: halo on function (space_order//2?) must be at least >= r
        m = TimeFunction(name="m", grid=grid, space_order=4, save=None, time_order=1)

        m.data[:] = 0.0
        m.data[:, 40, 40] = 1.0
        m.data[:, 50, 50] = 1.0

        # only rank 0 is allowed to have points
        if grid.distributor.myrank == 0:
            # A single dipole source - so two rows, one column
            matrix = scipy.sparse.coo_matrix(np.array([[1], [-1]], dtype=np.float32))
        else:
            matrix = scipy.sparse.coo_matrix((0, 0), dtype=np.float32)

        sf = MatrixSparseTimeFunction(name="s", grid=grid, r=r, matrix=matrix, nt=nt)

        if grid.distributor.myrank == 0:
            # First component of the dipole at 40, 40
            sf.gridpoints.data[0, 0] = 40
            sf.gridpoints.data[0, 1] = 40
            sf.interpolation_coefficients[x].data[0, 0] = 1.0
            sf.interpolation_coefficients[x].data[0, 1] = 2.0
            sf.interpolation_coefficients[y].data[0, 0] = 1.0
            sf.interpolation_coefficients[y].data[0, 1] = 2.0
            sf.gridpoints.data[1, 0] = 50
            sf.gridpoints.data[1, 1] = 50
            sf.interpolation_coefficients[x].data[1, 0] = 2.0
            sf.interpolation_coefficients[x].data[1, 1] = 2.0
            sf.interpolation_coefficients[y].data[1, 0] = 2.0
            sf.interpolation_coefficients[y].data[1, 1] = 2.0

        op = Operator(sf.interpolate(m))
        sf.manual_scatter()
        args = op.arguments(time_m=0, time_M=9)
        print("rank %d: %s" % (grid.distributor.myrank, str(args)))
        op.apply(time_m=0, time_M=0)
        sf.manual_gather()

        for i in range(grid.distributor.nprocs):
            print("==== from rank %d" % i)
            if i == grid.distributor.myrank:
                print(repr(sf.data))
            grid.distributor.comm.Barrier()

        if grid.distributor.myrank == 0:
            assert sf.data[0, 0] == -3.0  # 1 * (1 * 1) * 1 + (-1) * (2 * 2) * 1

from math import floor

import pytest
import numpy as np
import scipy.sparse

from devito import (Grid, TimeFunction, Eq, Operator, Dimension, Function,
                    SparseFunction, SparseTimeFunction, PrecomputedSparseFunction,
                    PrecomputedSparseTimeFunction, MatrixSparseTimeFunction,
                    switchconfig)


_sptypes = [SparseFunction, SparseTimeFunction,
            PrecomputedSparseFunction, PrecomputedSparseTimeFunction]


class TestMatrixSparseTimeFunction:

    def _precompute_linear_interpolation(self, points, grid, origin):
        """ Sample precompute function that, given point and grid information
            precomputes gridpoints and coefficients according to a linear
            scheme to be used in PrecomputedSparseFunction.
        """
        gridpoints = np.array([
            tuple(
                floor((point[i] - origin[i]) / grid.spacing[i]) for i in range(len(point))
            )
            for point in points
        ])

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
        points = np.array([(0.05, 0.9), (0.01, 0.8), (0.07, 0.84)])
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

    def _pure_python_coeffs(self, mstf):
        # Return a numpy array with the (nsrc, *grid.shape) coefficients
        # represented by the MatrixSparseTimeFunction mstf
        nloc, npoint = mstf.matrix.shape
        out = np.zeros((npoint, *mstf.grid.shape_local), dtype=np.float32)

        m_coo = mstf.matrix.tocoo()

        for row, col, val in zip(m_coo.row, m_coo.col, m_coo.data):
            base_gridpoint = mstf.gridpoints.data[row, :]

            # construct the stencil and the slices to which it will be applied
            stencil = np.array([1], dtype=np.float32)
            slices = [slice(col, col+1)]
            for i, d in enumerate(mstf.grid.dimensions):
                stencil = np.multiply.outer(
                    stencil, np.array(mstf.interpolation_coefficients[d].data[row, :])
                )
                if mstf.r[d] is None:
                    # applies to whole slice
                    slices.append(slice(None, None))
                else:
                    # applies to radius based at gridpoint
                    assert base_gridpoint[i] >= 0
                    assert base_gridpoint[i] + mstf.r[d] < mstf.grid.shape_local[i]
                    slices.append(
                        slice(base_gridpoint[i], base_gridpoint[i] + mstf.r[d])
                    )

            out[tuple(slices)] += val * stencil

        return out

    @pytest.mark.parametrize("rxy,par_dim_index", [
        # single-point injection
        (1, 0),
        # 2x2 stencil, parallel over x
        ((2, 2), 0),
        # 2x3 stencil, parallel over x
        ((2, 3), 0),
        # allx2 stencil, parallel over x
        ((None, 2), 0),
        # allx2 stencil, parallel over y
        ((None, 2), 1),
    ])
    def test_precomputed_subpoints_inject(self, rxy, par_dim_index):
        shape = (101, 101)
        grid = Grid(shape=shape)
        x, y = grid.dimensions

        if isinstance(rxy, tuple):
            r = {grid.dimensions[0]: rxy[0], grid.dimensions[1]: rxy[1]}
        else:
            r = rxy

        par_dim = grid.dimensions[par_dim_index]

        nt = 10

        m = TimeFunction(name="m", grid=grid, space_order=0, save=None, time_order=1)

        # Put some data in there to ensure it acts additively
        m.data[:] = 0.0
        m.data[:, 40, 40] = 1.0

        # Single two-component source with coefficients both +1
        matrix = scipy.sparse.coo_matrix(np.array([[1], [1]], dtype=np.float32))
        sf = MatrixSparseTimeFunction(
            name="s", grid=grid, r=r, par_dim=par_dim, matrix=matrix, nt=nt
        )

        coeff_size_x = sf.interpolation_coefficients[x].data.shape[1]
        coeff_size_y = sf.interpolation_coefficients[y].data.shape[1]

        sf.gridpoints.data[0, 0] = 40
        sf.gridpoints.data[0, 1] = 40
        sf.gridpoints.data[1, 0] = 39
        sf.gridpoints.data[1, 1] = 39
        sf.interpolation_coefficients[x].data[0, :] = 1.0 + np.arange(coeff_size_x)
        sf.interpolation_coefficients[y].data[0, :] = 1.0 + np.arange(coeff_size_y)
        sf.interpolation_coefficients[x].data[1, :] = 1.0 + np.arange(coeff_size_x)
        sf.interpolation_coefficients[y].data[1, :] = 1.0 + np.arange(coeff_size_y)
        sf.data[0, 0] = 1.0

        step = [Eq(m.forward, m)]
        inject = sf.inject(field=m.forward, expr=sf)
        op = Operator(step + inject)

        sf.manual_scatter()
        op(time_m=0, time_M=0)
        sf.manual_gather()

        check_coeffs = self._pure_python_coeffs(sf)
        expected_data1 = (
            m.data[0]
            + np.tensordot(
                np.array(sf.data[0, :]),
                check_coeffs,
                axes=1
            )
        )

        assert np.all(m.data[1] == expected_data1)

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
    def test_mpi(self, mode):
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


class TestSparseFunction:

    @pytest.mark.parametrize('sptype', _sptypes)
    def test_rebuild(self, sptype):
        grid = Grid((3, 3, 3))
        # Base object
        sp = sptype(name="s", grid=grid, npoint=1, nt=11, r=2,
                    interpolation_coeffs=np.random.randn(1, 3, 2),
                    coordinates=np.random.randn(1, 3))

        # Check subfunction setup
        for subf in sp._sub_functions:
            if getattr(sp, subf) is not None:
                assert getattr(sp, subf).name.startswith("s_")

        # Rebuild with different name, this should drop the function
        # and create new data, while the coordinates and more generally all
        # SubFunctions remain the same
        sp2 = sp._rebuild(name="sr")
        for subf in sp2._sub_functions:
            if getattr(sp2, subf) is not None:
                assert getattr(sp2, subf) == getattr(sp, subf)

        # Rebuild with different name as an alias
        sp2 = sp._rebuild(name="sr2", alias=True)
        assert sp2.name == "sr2"
        assert sp2.dimensions == sp.dimensions
        for subf in sp2._sub_functions:
            if getattr(sp2, subf) is not None:
                assert getattr(sp2, subf).name.startswith("sr2_")
                assert getattr(sp2, subf).data is None

        # Rebuild with different name and dimensions. This is expected to recreate
        # the SubFunctions as well
        sp2 = sp._rebuild(name="sr3", dimensions=None)
        assert sp2.name == "sr3"
        assert sp2.dimensions == sp.dimensions
        for subf in sp2._sub_functions:
            if getattr(sp2, subf) is not None:
                assert getattr(sp2, subf) == getattr(sp, subf)

    @pytest.mark.parametrize('sptype', _sptypes)
    def test_subs(self, sptype):
        grid = Grid((3, 3, 3))
        # Base object
        sp = sptype(name="s", grid=grid, npoint=1, nt=11, r=2,
                    interpolation_coeffs=np.random.randn(1, 3, 2),
                    coordinates=np.random.randn(1, 3))

        # Check subfunction setup
        for subf in sp._sub_functions:
            if getattr(sp, subf) is not None:
                assert getattr(sp, subf).dimensions[0] == sp._sparse_dim

        # Do substitution on sparse dimension
        new_spdim = Dimension(name="newsp")

        sps = sp._subs(sp._sparse_dim, new_spdim)
        assert sps.indices[sp._sparse_position] == new_spdim
        for subf in sps._sub_functions:
            if getattr(sps, subf) is not None:
                assert getattr(sps, subf).indices[0] == new_spdim
                assert np.all(getattr(sps, subf).data == getattr(sp, subf).data)

    @switchconfig(safe_math=True)
    @pytest.mark.parallel(mode=[1, 4])
    def test_mpi_no_data(self, mode):
        grid = Grid((11, 11), extent=(10, 10))
        time = grid.time_dim
        # Base object
        sp = SparseTimeFunction(name="s", grid=grid, npoint=1, nt=1,
                                coordinates=[[5., 5.]])

        m = TimeFunction(name="m", grid=grid, space_order=2, time_order=1)
        eq = [Eq(m.forward, m + m.laplace)]

        op = Operator(eq + sp.inject(field=m.forward, expr=time))
        # Not using the source data so can run with any time_M
        op(time_M=5)

        expected = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 4., -10., 4., 0., 0., 0., 0.],
                            [0., 0., 0., 6., -30., 55., -30., 6., 0., 0., 0.],
                            [0., 0., 4., -30., 102., -158., 102., -30., 4., 0., 0.],
                            [0., 1., -10., 55., -158., 239., -158., 55., -10., 1., 0.],
                            [0., 0., 4., -30., 102., -158., 102., -30., 4., 0., 0.],
                            [0., 0., 0., 6., -30., 55., -30., 6., 0., 0., 0.],
                            [0., 0., 0., 0., 4., -10., 4., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        ftest = Function(name='ftest', grid=grid, space_order=2)
        ftest.data[:] = expected
        assert np.all(m.data[0, :, :] == ftest.data[:])

    @pytest.mark.parametrize('dtype, expected', [(np.complex64, np.float32),
                                                 (np.complex128, np.float64),
                                                 (np.float16, np.float16)])
    def test_coordinate_type(self, dtype, expected):
        """
        Test that coordinates are always real and SparseFunction dtype is
        otherwise preserved.
        """
        grid = Grid(shape=(11,))
        s = SparseFunction(name='src', npoint=1,
                           grid=grid, dtype=dtype)

        assert s.coordinates.dtype is expected


if __name__ == "__main__":
    TestMatrixSparseTimeFunction().test_mpi_no_data()

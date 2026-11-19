import numpy as np
import pytest

from devito import Function, Grid, Inc, Operator, SparseTimeFunction, TimeFunction
from devito.ir.iet import FindSymbols


class TestLowerReductions:

    def test_no_temp_upon_reduce_expansion(self):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid)
        sf = SparseTimeFunction(name='sf', grid=grid, npoint=1, nt=5)

        rec_term = sf.interpolate(expr=u)

        op = Operator(rec_term, opt=('advanced', {'mapify-reduce': True}))

        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len([i for i in arrays if i.ndim == grid.dim]) == 0

    @pytest.mark.parametrize('shape', [(7,), (7, 9)])
    def test_pointwise_increment(self, shape):
        """Keep grid-valued time accumulations out of scalar map reductions."""
        grid = Grid(shape=shape)
        nt = 5
        u = TimeFunction(name='u', grid=grid, save=nt, space_order=0)
        error = Function(name='error', grid=grid, space_order=0)
        u.data[:] = np.indices(u.shape).sum(axis=0) % 4 + 1

        op = Operator(Inc(error, (u - 1)**2),
                      opt=('advanced', {'mapify-reduce': True}),
                      name='PointwiseIncrement')

        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert all(grid.time_dim not in i.dimensions for i in arrays)

        op.apply(time_m=0, time_M=nt - 1)
        np.testing.assert_array_equal(error.data, ((u.data - 1)**2).sum(axis=0))

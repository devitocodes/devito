from devito import Grid, SparseTimeFunction, TimeFunction, Operator
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

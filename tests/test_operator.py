from devito.interfaces import DenseData
import numpy as np
from devito.operators import SimpleOperator
from sympy import Eq
from sympy.abc import t, x


class Test_Operator(object):
    def test_codegen(self):
        input_grid = DenseData("input_grid", (3, 2), np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        output_grid = DenseData("output_grid", input_grid.shape, input_grid.dtype)
        eq = Eq(output_grid[t, x], input_grid[t, x] + 3)
        op = SimpleOperator(input_grid, output_grid, [eq])
        op.apply()
        assert(output_grid.data[2][1] == 8)

    def test_python(self):
        input_grid = DenseData("input_grid", (3, 2), np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        output_grid = DenseData("output_grid", input_grid.shape, input_grid.dtype)
        eq = Eq(output_grid[t, x], input_grid[t, x] + 3)
        op = SimpleOperator(input_grid, output_grid, [eq])
        op.apply(debug=True)
        assert(output_grid.data[2][1] == 8)

    def test_compare(self):
        input_grid = DenseData("input_grid", (3, 2), np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        output_grid_py = DenseData("output_grid_py", input_grid.shape, input_grid.dtype)
        output_grid_cg = DenseData("output_grid_cg", input_grid.shape, input_grid.dtype)
        eq_py = Eq(output_grid_py[t, x], input_grid[t, x] + 3)
        eq_cg = Eq(output_grid_cg[t, x], input_grid[t, x] + 3)
        op_cg = SimpleOperator(input_grid, output_grid_cg, [eq_cg])
        op_py = SimpleOperator(input_grid, output_grid_py, [eq_py])
        op_cg.apply()
        op_py.apply(debug=True)
        assert(np.equal(output_grid_cg.data, output_grid_py.data).all())

from devito.interfaces import DenseData
import numpy as np
from devito.operators import Operator
from sympy import Eq, symbols


class SimpleOperator(Operator):
    def __init__(self, input_grid):
        nt = input_grid.shape[0]
        shape = input_grid.shape[1:]
        output_grid = DenseData("output_grid", input_grid.shape, input_grid.dtype)
        self.input_params = [input_grid]
        self.output_params = [output_grid]
        x, t = symbols("x t")
        kernel = Eq(output_grid[t, x], input_grid[t, x] + 3)
        self.stencils = [(kernel, [])]
        super(SimpleOperator, self).__init__([], nt, shape, 0, 0, True, input_grid.dtype)


class Test_Operator(object):
    def test_codegen(self):
        input_grid = DenseData("input_grid", (3, 2), np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        op = SimpleOperator(input_grid)
        output_grid = op.apply()[0]
        assert(output_grid[2][1] == 8)

    def test_python(self):
        input_grid = DenseData("input_grid", (3, 2), np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        op = SimpleOperator(input_grid)
        output_grid = op.apply(debug=True)[0]
        assert(output_grid[2][1] == 8)

    def test_compare(self):
        input_grid = DenseData("input_grid", (3, 2), np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        op_cg = SimpleOperator(input_grid)
        op_py = SimpleOperator(input_grid)
        output_grid_cg = op_cg.apply()[0]
        output_grid_py = op_py.apply(debug=True)[0]
        assert(np.linalg.norm(output_grid_cg) == np.linalg.norm(output_grid_py))

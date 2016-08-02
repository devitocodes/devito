import numpy as np
from sympy import Eq
from sympy.abc import t, x

from devito.interfaces import DenseData
from devito.operator import SimpleOperator


class Test_Operator(object):
    def test_codegen(self):
        input_grid = DenseData(name="input_grid", shape=(3, 2), dtype=np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        output_grid = DenseData(name="output_grid", shape=input_grid.shape,
                                dtype=input_grid.dtype)
        eq = Eq(output_grid.indexed[t, x], input_grid.indexed[t, x] + 3)
        op = SimpleOperator(input_grid, output_grid, [eq])
        op.apply()
        assert(output_grid.data[2][1] == 8)

    def test_python(self):
        input_grid = DenseData(name="input_grid", shape=(3, 2), dtype=np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        output_grid = DenseData(name="output_grid", shape=input_grid.shape,
                                dtype=input_grid.dtype)
        eq = Eq(output_grid.indexed[t, x], input_grid.indexed[t, x] + 3)
        op = SimpleOperator(input_grid, output_grid, [eq])
        op.apply(debug=True)
        assert(output_grid.data[2][1] == 8)

    def test_compare(self):
        input_grid = DenseData(name="input_grid", shape=(3, 2), dtype=np.float64)
        input_grid.data[:] = np.arange(6, dtype=np.float64).reshape((3, 2))
        output_grid_py = DenseData(name="output_grid_py", shape=input_grid.shape,
                                   dtype=input_grid.dtype)
        output_grid_cg = DenseData(name="output_grid_cg", shape=input_grid.shape,
                                   dtype=input_grid.dtype)
        eq_py = Eq(output_grid_py.indexed[t, x], input_grid.indexed[t, x] + 3)
        eq_cg = Eq(output_grid_cg.indexed[t, x], input_grid.indexed[t, x] + 3)
        op_cg = SimpleOperator(input_grid, output_grid_cg, [eq_cg])
        op_py = SimpleOperator(input_grid, output_grid_py, [eq_py])
        op_cg.apply()
        op_py.apply(debug=True)
        assert(np.equal(output_grid_cg.data, output_grid_py.data).all())

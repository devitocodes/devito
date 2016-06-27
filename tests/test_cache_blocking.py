from devito.interfaces import DenseData
import numpy as np
from sympy import symbols, Eq
from devito.operators import SimpleOperator


class Test_Cache_Blocking(object):
    def test_cache_blocking_no_remainder(self):
        input_grid = DenseData("input_grid", (300, 300), np.float64)
        input_grid.data[:] = np.arange(90000, dtype=np.float64).reshape((300, 300))
        output_grid_noblock = DenseData("output_grid", (300, 300), np.float64)
        x, t = symbols("x t")
        eq = Eq(output_grid_noblock[t, x], input_grid[t, x] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq])
        op_noblock.apply()
        output_grid_block = DenseData("output_grid", (300, 300), np.float64)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq], cache_blocking=True)
        op_block.apply()
        assert(np.linalg.norm(output_grid_block.data) == np.linalg.norm(output_grid_noblock.data))

    def test_cache_blocking_remainder(self):
        input_grid = DenseData("input_grid", (302, 302), np.float64)
        input_grid.data[:] = np.arange(91204, dtype=np.float64).reshape((302, 302))
        output_grid_noblock = DenseData("output_grid", (302, 302), np.float64)
        x, t = symbols("x t")
        eq = Eq(output_grid_noblock[t, x], input_grid[t, x] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq])
        op_noblock.apply()
        output_grid_block = DenseData("output_grid", (302, 302), np.float64)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq], cache_blocking=True)
        op_block.apply()
        assert(np.linalg.norm(output_grid_block.data) == np.linalg.norm(output_grid_noblock.data))

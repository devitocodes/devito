import numpy as np
from sympy import Eq, symbols

from devito.interfaces import DenseData
from devito.operator import SimpleOperator


class Test_Cache_Blocking(object):
    def test_cache_blocking_no_remainder(self):
        input_grid = DenseData(name="input_grid", shape=(300, 300), dtype=np.float64)
        input_grid.data[:] = np.arange(90000, dtype=np.float64).reshape((300, 300))
        output_grid_noblock = DenseData(name="output_grid", shape=(300, 300), dtype=np.float64)
        x, t = symbols("x t")
        eq = Eq(output_grid_noblock.indexed[t, x], input_grid.indexed[t, x] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq])
        op_noblock.apply()
        output_grid_block = DenseData(name="output_grid", shape=(300, 300), dtype=np.float64)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq], cache_blocking=True)
        op_block.apply()
        assert(np.equal(output_grid_block.data, output_grid_noblock.data).all())

    def test_cache_blocking_remainder(self):
        input_grid = DenseData(name="input_grid", shape=(302, 302), dtype=np.float64)
        input_grid.data[:] = np.arange(91204, dtype=np.float64).reshape((302, 302))
        output_grid_noblock = DenseData(name="output_grid", shape=(302, 302), dtype=np.float64)
        x, t = symbols("x t")
        eq = Eq(output_grid_noblock.indexed[t, x], input_grid.indexed[t, x] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq])
        op_noblock.apply()
        output_grid_block = DenseData(name="output_grid", shape=(302, 302), dtype=np.float64)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq], cache_blocking=True)
        op_block.apply()
        assert(np.equal(output_grid_block.data, output_grid_noblock.data).all())

    def test_cache_blocking_cb_inner_dim(self):
        input_grid = DenseData(name="input_grid", shape=(302, 302), dtype=np.float64)
        input_grid.data[:] = np.arange(91204, dtype=np.float64).reshape((302, 302))
        output_grid_noblock = DenseData(name="output_grid", shape=(302, 302), dtype=np.float64)
        x, t = symbols("x t")
        eq = Eq(output_grid_noblock.indexed[t, x], input_grid.indexed[t, x] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq])
        op_noblock.apply()
        output_grid_block = DenseData(name="output_grid", shape=(302, 302), dtype=np.float64)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq], cache_blocking=True, cb_inner_dim=True)
        op_block.apply()
        assert (np.equal(output_grid_block.data, output_grid_noblock.data).all())

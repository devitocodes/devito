from devito.interfaces import DenseData
import numpy as np
import pytest
from sympy import symbols, Eq
from devito.operator import SimpleOperator


class Test_Cache_Blocking(object):

    # Full range testing
    @pytest.mark.parametrize("time_order", [2])
    @pytest.mark.parametrize("spc_border", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("block_size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    @pytest.mark.parametrize("cb_inner_dim", [False, True])
    def test_cache_blocking_full_range(self, time_order, spc_border, block_size, cb_inner_dim):
        input_grid = DenseData(name="input_grid", shape=(302, 302), dtype=np.float64)
        input_grid.data[:] = np.arange(91204, dtype=np.float64).reshape((302, 302))
        x, t = symbols("x t")

        output_grid_noblock = DenseData(name="output_grid", shape=(302, 302), dtype=np.float64)
        eq = Eq(output_grid_noblock.indexed[t, x], input_grid.indexed[t, x] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq], time_order=time_order, spc_border=spc_border)
        op_noblock.apply()

        output_grid_block = DenseData(name="output_grid", shape=(302, 302), dtype=np.float64)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq], cache_blocking=True, cb_inner_dim=cb_inner_dim,
                                  block_size=block_size, time_order=time_order, spc_border=spc_border)
        op_block.apply()
        assert np.equal(output_grid_block.data, output_grid_noblock.data).all()

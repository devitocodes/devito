import numpy as np
import pytest
from sympy import Eq, symbols

from devito.interfaces import DenseData
from devito.operator import SimpleOperator


class Test_Cache_Blocking(object):

    # Full range testing.            This syntax tests all possible permutations of parameters
    @pytest.mark.parametrize("shape", [(10, 45), (10, 31, 45), (10, 45, 31, 45)])
    @pytest.mark.parametrize("time_order", [2])
    @pytest.mark.parametrize("spc_border", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("block_size", [2, 3, 4, 5, 6, 7, 8])
    def test_cache_blocking_full_range(self, shape, time_order, spc_border, block_size):
        self.cache_blocking_test(shape, time_order, spc_border, block_size)

    # Edge cases. Different block sizes, etc
    @pytest.mark.parametrize("shape,time_order,spc_border,block_size", [
        ((10, 25, 25, 46), 2, 3, [None, None, None]),
        ((10, 25, 25, 46), 2, 3, [7, None, None]),
        ((10, 25, 25, 46), 2, 3, [None, None, 7]),
        ((10, 25, 25, 46), 2, 3, [None, 7, None]),
        ((10, 25, 25, 46), 2, 3, [5, None, 7]),
        ((10, 25, 25, 46), 2, 3, [10, 3, None]),
        ((10, 25, 25, 46), 2, 3, [None, 7, 11]),
        ((10, 25, 25, 46), 2, 3, [8, 2, 4]),
        ((10, 25, 25, 46), 2, 3, [2, 4, 8]),
        ((10, 25, 25, 46), 2, 3, [4, 8, 2]),
        ((10, 25, 46), 2, 3, [None, 7]),
        ((10, 25, 46), 2, 3, [7, None]),
        ((10, 25, 46), 2, 3, None)
    ])
    def test_cache_blocking_edge_cases(self, shape, time_order, spc_border, block_size):
        self.cache_blocking_test(shape, time_order, spc_border, block_size)

    def cache_blocking_test(self, shape, time_order, spc_border, block_size):
        symbols_combinations = ['t', 't x', 't x z', 't x y z']
        indexes = symbols(symbols_combinations[len(shape) - 1])

        size = 1
        for element in shape:
            size *= element

        input_grid = DenseData(name="input_grid", shape=shape, dtype=np.float64)
        input_grid.data[:] = np.arange(size, dtype=np.float64).reshape(shape)

        output_grid_noblock = DenseData(name="output_grid", shape=shape, dtype=np.float64)
        eq_noblock = Eq(output_grid_noblock.indexed[indexes],
                        output_grid_noblock.indexed[indexes] + input_grid.indexed[indexes] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq_noblock],
                                    time_order=time_order, spc_border=spc_border)
        op_noblock.apply()

        output_grid_block = DenseData(name="output_grid", shape=shape, dtype=np.float64)
        eq_block = Eq(output_grid_block.indexed[indexes],
                      output_grid_block.indexed[indexes] + input_grid.indexed[indexes] + 3)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq_block], cache_blocking=True,
                                  block_size=block_size, time_order=time_order, spc_border=spc_border)
        op_block.apply()
        assert np.equal(output_grid_block.data, output_grid_noblock.data).all()

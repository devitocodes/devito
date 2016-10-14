import numpy as np
import pytest
from sympy import Eq

from devito.dimension import t, x, y, z
from devito.interfaces import TimeData
from devito.operator import SimpleOperator


class Test_Cache_Blocking(object):

    # Full range testing.   This syntax tests all possible permutations of parameters
    @pytest.mark.parametrize("shape", [(45, ), (31, 45), (45, 31, 45)])
    @pytest.mark.parametrize("time_order", [2])
    @pytest.mark.parametrize("spc_border", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("cache_blocking", [2, 3, 4, 5, 6, 7, 8])
    def test_cache_blocking_full_range(self, shape, time_order,
                                       spc_border, cache_blocking):
        self.cache_blocking_test(shape, time_order, spc_border, cache_blocking)

    # Edge cases. Different block sizes, etc
    @pytest.mark.parametrize("shape,time_order,spc_border,cache_blocking", [
        ((25, 25, 46), 2, 3, [None, None, None]),
        ((25, 25, 46), 2, 3, [7, None, None]),
        ((25, 25, 46), 2, 3, [None, None, 7]),
        ((25, 25, 46), 2, 3, [None, 7, None]),
        ((25, 25, 46), 2, 3, [5, None, 7]),
        ((25, 25, 46), 2, 3, [10, 3, None]),
        ((25, 25, 46), 2, 3, [None, 7, 11]),
        ((25, 25, 46), 2, 3, [8, 2, 4]),
        ((25, 25, 46), 2, 3, [2, 4, 8]),
        ((25, 25, 46), 2, 3, [4, 8, 2]),
        ((25, 46), 2, 3, [None, 7]),
        ((25, 46), 2, 3, [7, None])
    ])
    def test_cache_blocking_edge_cases(self, shape, time_order,
                                       spc_border, cache_blocking):
        self.cache_blocking_test(shape, time_order, spc_border, cache_blocking)

    def cache_blocking_test(self, shape, time_order, spc_border, cache_blocking):
        symbols_combinations = [(t, x), (t, x, z), (t, x, y, z)]
        indexes = symbols_combinations[len(shape) - 1]

        size = 1
        for element in shape:
            size *= element

        input_grid = TimeData(name="input_grid", shape=shape, dtype=np.float64,
                              save=True, time_dim=9)
        input_grid.data[:] = np.arange(size, dtype=np.float64).reshape(shape)

        output_grid_noblock = TimeData(name="output_grid", shape=shape,
                                       dtype=np.float64, save=True, time_dim=9)
        eq_noblock = Eq(output_grid_noblock.indexed[indexes],
                        output_grid_noblock.indexed[indexes] +
                        input_grid.indexed[indexes] + 3)
        op_noblock = SimpleOperator(input_grid, output_grid_noblock, [eq_noblock],
                                    time_order=time_order, spc_border=spc_border)
        op_noblock.apply()

        output_grid_block = TimeData(name="output_grid", shape=shape,
                                     dtype=np.float64, save=True, time_dim=9)
        eq_block = Eq(output_grid_block.indexed[indexes],
                      output_grid_block.indexed[indexes] +
                      input_grid.indexed[indexes] + 3)
        op_block = SimpleOperator(input_grid, output_grid_block, [eq_block],
                                  cache_blocking=cache_blocking,
                                  time_order=time_order, spc_border=spc_border)
        op_block.apply()
        assert np.equal(output_grid_block.data, output_grid_noblock.data).all()

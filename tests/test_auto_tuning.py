from os import path
from shutil import rmtree

import numpy as np
import pytest
from sympy import Eq

from devito.dimension import t, x, y, z
from devito.interfaces import DenseData
from devito.operator import SimpleOperator


class Test_Auto_Tuning(object):

    @classmethod
    def setup_class(self):  # Sets env var for report dir
        self.test_dir = path.join(path.dirname(path.realpath(__file__)), "At_test")
        self.report_file = path.join(self.test_dir, "final_report.txt")

    def teardown_method(self, method):
        if path.isdir(self.test_dir):
            rmtree(self.test_dir)

    @pytest.mark.parametrize("block_dims,tune_range,expected_result", [
        ([True, True, False], (5, 6), [5, 5, None]),
        ([True, False, True], (5, 6), [5, None, 5]),
        ([False, True, True], (5, 6), [None, 5, 5]),
        ([True, False, False], (5, 6), [5, None, None]),
        ([False, True, False], (5, 6), [None, 5, None])
    ])
    def test_auto_tuning_blocks(self, block_dims, tune_range, expected_result):
        self.auto_tuning_test_general(block_dims, tune_range, expected_result)

    def auto_tuning_test_general(self, block_dims, tune_range, expected_result):
        shape = (50, 50, 50, 50)
        input_grid = DenseData(name="input_grid", shape=shape, dtype=np.float64)
        input_grid.data[:] = np.arange(6250000, dtype=np.float64).reshape(shape)
        output_grid = DenseData(name="output_grid", shape=shape, dtype=np.float64)
        indexes = (t, x, y, z)
        eq = Eq(output_grid.indexed[indexes],
                output_grid.indexed[indexes] + input_grid.indexed[indexes] + 3)

        op = SimpleOperator(
            input_grid, output_grid, [eq], time_order=2, spc_border=2,
            auto_tuning=True, blocked_dims=block_dims, at_range=tune_range
        )

        assert op.propagator.cache_blocking == expected_result

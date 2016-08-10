from os import path
from shutil import rmtree

import numpy as np
import pytest
from sympy import Eq, symbols

from devito.at_controller import AutoTuner, get_at_block_size
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

    @pytest.mark.parametrize("cache_blocking,tune_range,expected_result", [
        (5, (5, 6), [5, 5, None]),
        ([5, 5, None], (5, 6), [5, 5, None]),
        ([5, None, 5], (5, 6), [5, None, 5]),
        ([None, 5, 5], (5, 6), [None, 5, 5]),
        ([5, None, None], (5, 6), [5, None, None]),
        ([None, 5, None], (5, 6), [None, 5, None])
    ])
    def test_auto_tuning_blocks(self, cache_blocking, tune_range, expected_result):
        self.auto_tuning_test_general(cache_blocking, tune_range, expected_result)

    @pytest.mark.parametrize("cache_blocking,tune_range,expected_result", [
        pytest.mark.xfail((5, (5, 6), [5, 5, 5]), strict=True),
        pytest.mark.xfail(([5, 5, None], (5, 6), [5, 5, 5]), strict=True)
    ])
    def test_auto_tuning_b_negative(self, cache_blocking, tune_range, expected_result):
        self.auto_tuning_test_general(cache_blocking, tune_range, expected_result)

    def test_auto_tuning_correctness(self):

        shape = (50, 50, 50, 50)
        indexes = symbols("t x y z")

        size = 1
        for element in shape:
            size *= element

        input_grid = DenseData(name="input_grid", shape=shape, dtype=np.float64)
        input_grid.data[:] = np.arange(size, dtype=np.float64).reshape(shape)

        output_grid_noat = DenseData(name="output_grid", shape=shape, dtype=np.float64)
        eq_noat = Eq(output_grid_noat.indexed[indexes],
                     output_grid_noat.indexed[indexes] + input_grid.indexed[indexes] + 3)
        op_noat = SimpleOperator(input_grid, output_grid_noat, [eq_noat], time_order=2, spc_border=2)
        op_noat.apply()

        output_grid_at = DenseData(name="output_grid", shape=shape, dtype=np.float64)
        eq_block = Eq(output_grid_at.indexed[indexes],
                      output_grid_at.indexed[indexes] + input_grid.indexed[indexes] + 3)
        op_at = SimpleOperator(input_grid, output_grid_at, [eq_block],
                               cache_blocking=[5, 5, 5], time_order=2, spc_border=2)
        op_at.propagator.auto_tune = True

        f, args = op_at.apply(auto_tune=True)
        args += [5, 5, 5]
        f(*args)

        assert np.equal(output_grid_noat.data, output_grid_at.data).all()

    def auto_tuning_test_general(self, cache_blocking, tune_range, expected_result):
        shape = (50, 50, 50, 50)
        input_grid = DenseData(name="input_grid", shape=shape, dtype=np.float64)
        input_grid.data[:] = np.arange(6250000, dtype=np.float64).reshape(shape)
        output_grid = DenseData(name="output_grid", shape=shape, dtype=np.float64)
        indexes = symbols("t x y z")
        eq = Eq(output_grid.indexed[indexes],
                output_grid.indexed[indexes] + input_grid.indexed[indexes] + 3)
        op = SimpleOperator(input_grid, output_grid, [eq], time_order=2, spc_border=2,
                            cache_blocking=cache_blocking)

        auto_tuner = AutoTuner(op, self.test_dir)
        auto_tuner.auto_tune_blocks(tune_range[0], tune_range[1])

        at_block_sizes = get_at_block_size(op.getName(), op.time_order, op.spc_border, op.shape,
                                           op.propagator.cache_blocking, at_report_dir=self.test_dir)

        assert at_block_sizes == expected_result

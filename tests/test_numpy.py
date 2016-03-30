from generator import Generator
import numpy as np
import cgen
import os
import shutil


# Note: The name of the generated file needs to be different for
# each test to bypass ctype caching
class Test_Numpy_Array_Transfer(object):
    _tmp_dir_name = "tmp"

    def setup_method(self, test_method):
        os.mkdir(self._tmp_dir_name)

    def teardown_method(self, test_method):
        shutil.rmtree(self._tmp_dir_name)

    def test_2d(self):
        kernel = cgen.Assign("output_grid[i2][i1]", "input_grid[i2][i1] + 3")
        g = Generator(np.arange(6, dtype=np.float64).reshape((3, 2)), kernel)
        arr = g.execute(self._tmp_dir_name+"/basic_2d.cpp")
        assert(arr[2][1] == 8)

    def test_3d(self):
        kernel = cgen.Assign("output_grid[i3][i2][i1]",
                             "input_grid[i3][i2][i1] + 3")
        g = Generator(np.arange(24, dtype=np.float64).reshape((4, 3, 2)),
                      kernel)
        arr = g.execute(self._tmp_dir_name+"/basic_3d.cpp")
        assert(arr[3][2][1] == 26)

    def test_4d(self):
        kernel = cgen.Assign("output_grid[i4][i3][i2][i1]",
                             "input_grid[i4][i3][i2][i1] + 3")
        g = Generator(np.arange(120, dtype=np.float64).reshape((5, 4, 3, 2)),
                      kernel)
        arr = g.execute(self._tmp_dir_name+"/basic_4d.cpp")
        assert(arr[4][3][2][1] == 122)

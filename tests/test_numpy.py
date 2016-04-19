from generator import Generator
import numpy as np
import cgen
from function_descriptor import FunctionDescriptor


class Test_Numpy_Array_Transfer(object):

    def test_2d(self):
        data = np.arange(6, dtype=np.float64).reshape((3, 2))
        kernel = cgen.Assign("output_grid[i2][i1]", "input_grid[i2][i1] + 3")
        fd = FunctionDescriptor("process", kernel)
        fd.add_matrix_param("input_grid", data.shape, True)
        fd.add_matrix_param("output_grid", data.shape)
        g = Generator(fd)
        f = g.get_wrapped_function()
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[2][1] == 8)

    def test_3d(self):
        kernel = cgen.Assign("output_grid[i3][i2][i1]",
                             "input_grid[i3][i2][i1] + 3")
        data = np.arange(24, dtype=np.float64).reshape((4, 3, 2))
        fd = FunctionDescriptor("process", kernel)
        fd.add_matrix_param("input_grid", data.shape, True)
        fd.add_matrix_param("output_grid", data.shape)
        g = Generator(fd)
        f = g.get_wrapped_function()
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[3][2][1] == 26)

    def test_4d(self):
        kernel = cgen.Assign("output_grid[i4][i3][i2][i1]",
                             "input_grid[i4][i3][i2][i1] + 3")
        data = np.arange(120, dtype=np.float64).reshape((5, 4, 3, 2))
        fd = FunctionDescriptor("process", kernel)
        fd.add_matrix_param("input_grid", data.shape, True)
        fd.add_matrix_param("output_grid", data.shape)
        g = Generator(fd)
        f = g.get_wrapped_function()
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[4][3][2][1] == 122)

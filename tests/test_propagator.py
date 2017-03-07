import numpy as np

import cgen
from devito import x, y, z
from devito.propagator import Propagator


class Test_Propagator(object):

    def test_2d(self):
        data = np.arange(6, dtype=np.float64).reshape((3, 2))
        propagator = Propagator("process", 3, (2, ), [])
        propagator.add_param("input_grid", data.shape, data.dtype)
        propagator.add_param("output_grid", data.shape, data.dtype)
        kernel = cgen.Assign("output_grid[i2][i1]", "input_grid[i2][i1] + 3")
        propagator.loop_body = kernel
        f = propagator.cfunction
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[2][1] == 8)

    def test_3d(self):
        kernel = cgen.Assign("output_grid[i3][i1][i2]",
                             "input_grid[i3][i1][i2] + 3")
        data = np.arange(24, dtype=np.float64).reshape((4, 3, 2))
        propagator = Propagator("process", 4, (3, 2), [])
        propagator.add_param("input_grid", data.shape, data.dtype)
        propagator.add_param("output_grid", data.shape, data.dtype)
        propagator.loop_body = kernel
        f = propagator.cfunction
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[3][2][1] == 26)

    def test_4d(self):
        kernel = cgen.Assign("output_grid[i4][i1][i2][i3]",
                             "input_grid[i4][i1][i2][i3] + 3")
        data = np.arange(120, dtype=np.float64).reshape((5, 4, 3, 2))
        propagator = Propagator("process", 5, (4, 3, 2), [])
        propagator.add_param("input_grid", data.shape, data.dtype)
        propagator.add_param("output_grid", data.shape, data.dtype)
        propagator.loop_body = kernel
        f = propagator.cfunction
        arr = np.empty_like(data)
        f(data, arr)
        assert(arr[4][3][2][1] == 122)

    def test_space_dims_2d(self):
        space_dims = (z, x)
        propagator = Propagator("process", 1, (4, 3), [], space_dims=space_dims)
        assert(space_dims == propagator.space_dims)

    def test_space_dims_3d(self):
        space_dims = (z, y, x)
        propagator = Propagator("process", 1, (4, 3, 2), [], space_dims=space_dims)
        assert(space_dims == propagator.space_dims)

    def test_space_dims_2d_default(self):
        space_dims = (x, z)
        propagator = Propagator("process", 1, (4, 3), [])
        assert(space_dims == propagator.space_dims)

    def test_space_dims_3d_default(self):
        space_dims = (x, y, z)
        propagator = Propagator("process", 1, (4, 3, 2), [])
        assert(space_dims == propagator.space_dims)


if __name__ == "__main__":
    t = Test_Propagator()
    t.test_2d()
    t.test_3d()
    t.test_4d()

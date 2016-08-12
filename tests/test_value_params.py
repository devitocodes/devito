import numpy as np

import devito.cgen_wrapper as cgen
from devito.propagator import Propagator


def test_value_param():
    data = np.arange(6, dtype=np.float64).reshape((3, 2))
    kernel = cgen.Assign("output_grid[i2][i1]", "input_grid[i2][i1] + offset")
    propagator = Propagator("process", 3, (2,), [])
    propagator.add_param("input_grid", data.shape, data.dtype)
    propagator.add_param("output_grid", data.shape, data.dtype)
    propagator.add_scalar_param("offset", np.int32)
    propagator.loop_body = kernel
    f = propagator.cfunction
    arr = np.empty_like(data)
    f(data, arr, np.int32(3))
    assert(arr[2][1] == 8)

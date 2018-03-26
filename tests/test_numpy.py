import numpy as np
from numpy.random import rand
from conftest import skipif_yask

import numpy as np

# Checking for bug in numpy.dot
# https://github.com/ContinuumIO/anaconda-issues/issues/7457
v = rand(1000).astype(np.float64)
assert np.isclose(np.dot(v, v), (v*v).sum())

v = rand(1000).astype(np.float32)
assert np.isclose(np.dot(v, v), (v*v).sum())


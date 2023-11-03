import numpy as np


def test_numpy_dot():
    # Checking for bug in numpy.dot
    # https://github.com/ContinuumIO/anaconda-issues/issues/7457

    # If you run into this bug then try:
    #  conda install numpy nomkl
    v = np.random.rand(1000).astype(np.float64)
    assert np.isclose(np.dot(v, v), (v*v).sum())

    v = np.random.rand(1000).astype(np.float32)
    assert np.isclose(np.dot(v, v), (v*v).sum())

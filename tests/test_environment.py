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


def test_without_pytorch():
    """Optional tensor classes remain importable when PyTorch is not installed."""
    import subprocess
    import sys

    result = subprocess.run([sys.executable, '-c', '''
import importlib.abc
import sys

class WithoutTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'")

sys.meta_path.insert(0, WithoutTorch())
import numpy as np
from devito import Function, Grid, TorchAllocator
from devito.data import Data, NumpyData, TorchData
assert issubclass(TorchData, Data)
for factory in (lambda: TorchData((4,), np.float32), TorchAllocator):
    try:
        factory()
    except ImportError:
        pass
    else:
        raise AssertionError('PyTorch dependency must be checked at construction')
f = Function(name='f', grid=Grid(shape=(4, 4)))
assert isinstance(f.data, NumpyData)
'''], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

import numpy as np

import devito
from devito.interfaces import DenseData


def init(data):
    data = np.zeros(data.shape)


def test_symbol_cache(nx=1000, ny=1000):

    for i in range(10):
        DenseData(name='u', shape=(nx, ny), dtype=np.float64, space_order=2, initializer=init)

        assert(len(devito.interfaces._SymbolCache) == 1)

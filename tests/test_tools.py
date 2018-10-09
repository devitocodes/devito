import pytest
from conftest import skipif_yask

from sympy.abc import a, b, c, d, e
import numpy as np

from devito.tools import toposort, numpy_view_offsets


@skipif_yask
@pytest.mark.parametrize('elements, expected', [
    ([[a, b, c], [c, d, e]], [a, b, c, d, e]),
    ([[e, d, c], [c, b, a]], [e, d, c, b, a]),
    ([[a, b, c], [b, d, e]], [a, b, d, c, e]),
    ([[a, b, c], [d, b, c]], [a, d, b, c]),
    ([[a, b, c], [c, d, b]], None),
])
def test_toposort(elements, expected):
    try:
        ordering = toposort(elements)
        assert ordering == expected
    except ValueError:
        assert expected is None


@skipif_yask
@pytest.mark.parametrize('arr,index,expected', [
    (np.zeros(shape=(8, 10)), [slice(2, 6), slice(5, 7)], ((2, 2), (5, 3))),
    (np.zeros(shape=(8, 10)), [slice(2), slice(2)], ((0, 6), (0, 8))),
    (np.zeros(shape=(8, 10, 2)), [slice(2, 5), slice(2, 5), slice(1, 2)],
     ((2, 3), (2, 5), (1, 0)))
])
def test_numpy_view_offsets(arr, index, expected):
    view = arr[index]
    assert numpy_view_offsets(view) == expected

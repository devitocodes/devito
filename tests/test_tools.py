import numpy as np
import pytest
from sympy.abc import a, b, c, d, e
import time

from conftest import skipif
from devito.tools import toposort, filter_ordered

pytestmark = skipif(['yask', 'ops'])


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


def test_sorting():
    key = lambda x: x
    array = np.random.rand(10000)
    t0 = time.time()
    sort_key = filter_ordered(array, key=key)
    t1 = time.time()
    sort_nokey = filter_ordered(array)
    t2 = time.time()
    # This one is slightly faster
    assert t2 - t1 < .5 * (t1 - t0)
    assert sort_key == sort_nokey

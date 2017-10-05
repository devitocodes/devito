import pytest
from conftest import skipif_yask

from sympy.abc import a, b, c, d, e

from devito.tools import partial_order


@skipif_yask
@pytest.mark.parametrize('elements, expected', [
    ([[a, b, c], [c, d, e]], [a, b, c, d, e]),
    ([[e, d, c], [c, b, a]], [e, d, c, b, a]),
    ([[a, b, c], [b, d, e]], [a, b, c, d, e]),
    ([[a, b, c], [d, b, c]], [a, d, b, c]),
])
def test_partial_order(elements, expected):
    ordering = partial_order(elements)
    assert ordering == expected

import numpy as np

from devito import DenseData, clear_cache
from devito.interfaces import _SymbolCache


@pytest.mark.xfail(reason="Known symbol caching bug due to false aliasing")
def test_symbol_cache_aliasing():
    """Test to assert that our aiasing cache isn't defeated by sympys
    non-aliasing symbol cache.

    For further explanation consider the symbol u[x, y] and it's first
    derivative in x, which includes the symbols u[x, y] and u[x + h, y].
    The two functions are aliased in devito's caching mechanism to allow
    multiple stencil indices pointing at the same data object u, but
    SymPy treats these two instances as separate functions and thus is
    allowed to delete one or the other when the cache is cleared.

    The test below asserts that if either of these instances is deleted,
    the data on u is still intact through our own caching mechanism."""

    # Ensure a clean cache to start with
    clear_cache()
    assert(len(_SymbolCache) == 0)
    # Create first instance of u and fill its data
    u = DenseData(name='u', shape=(3, 4))
    u.data[:] = 6.

    # Test 1: Create u[x + h, y] and delete it again
    dx = u.dx  # Contains two u symbols: u[x, y] and u[x + h, y]
    del dx
    clear_cache()
    assert len(_SymbolCache) == 1  # We still have a reference to u
    assert np.allclose(u.data, 6.)  # u.data is alive

    # Test 2: Create and keep u[x, y + h] and delete u[x, y]
    dy = u.dy
    u_h = dy.args[0].args[1]  # Store a copy of the second variant
    del dy
    del u
    clear_cache()
    assert len(_SymbolCache) == 1  # We still have a reference to u_h
    assert np.allclose(u_h.data, 6.)  # u_h.data is alive


def test_clear_cache(nx=1000, ny=1000):
    clear_cache()
    cache_size = len(_SymbolCache)

    for i in range(10):
        assert(len(_SymbolCache) == cache_size)

        DenseData(name='u', shape=(nx, ny), dtype=np.float64, space_order=2)

        assert(len(_SymbolCache) == cache_size + 1)

        clear_cache()

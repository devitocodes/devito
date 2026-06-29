import numpy as np
import pytest

from devito import Grid


# Unsigned ints are unreasonable but not necessarily invalid
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.longdouble,
                                   np.complex64, np.complex128, np.int8, np.int16,
                                   np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                                   np.uint64])
def test_extent_dtypes(dtype: np.dtype[np.number]) -> None:
    """
    Test that grid spacings are correctly computed for different dtypes.
    """

    # Construct a grid with the dtype and retrieve the spacing values
    extent = (1, 1j) if np.issubdtype(dtype, np.complexfloating) else (2, 4)
    grid = Grid(shape=(5, 5), extent=extent, dtype=dtype)
    dx, dy = grid.spacing_map.values()

    # Check that the spacings have the correct dtype
    assert dx.dtype == dy.dtype == dtype

    # Check that the spacings have the correct values
    assert dx == dtype(extent[0] / 4)
    assert dy == dtype(extent[1] / 4)

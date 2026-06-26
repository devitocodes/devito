import numpy as np
import pytest

from devito import Grid
from devito.types.grid import GridHierarchy, SubGrid


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


class TestGridHierarchy:

    def test_shapes_1d(self):
        grid = Grid(shape=(17,))
        hierarchy = GridHierarchy(grid, nlevels=3)
        assert hierarchy.levels[0] is grid
        assert hierarchy.levels[1].shape == (9,)
        assert hierarchy.levels[2].shape == (5,)

    def test_coarsening_depth(self):
        grid = Grid(shape=(17,))
        hierarchy = GridHierarchy(grid, nlevels=3)
        assert hierarchy.coarse_levels[0].coarsening_depth == 1
        assert hierarchy.coarse_levels[1].coarsening_depth == 2

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            GridHierarchy(Grid(shape=(10,)), nlevels=3)

    def test_shapes_2d(self):
        grid = Grid(shape=(17, 17))
        hierarchy = GridHierarchy(grid, nlevels=3)
        assert hierarchy.levels[0] is grid
        assert hierarchy.levels[1].shape == (9, 9)
        assert hierarchy.levels[2].shape == (5, 5)

    def test_shapes_3d(self):
        grid = Grid(shape=(17, 17, 17))
        hierarchy = GridHierarchy(grid, nlevels=3)
        assert hierarchy.levels[0] is grid
        assert hierarchy.levels[1].shape == (9, 9, 9)
        assert hierarchy.levels[2].shape == (5, 5, 5)

    def test_shared_properties(self):
        grid = Grid(shape=(17,))
        hierarchy = GridHierarchy(grid, nlevels=3)
        for sg in hierarchy.coarse_levels:
            assert sg.dimensions == grid.dimensions
            assert sg.extent == grid.extent
            assert sg.dtype == grid.dtype

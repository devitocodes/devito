import numpy as np
import pytest

from devito import Dimension, Function, Grid
from devito.types import StencilDimension, SparseFunction, PrecomputedSparseFunction
from devito.data.allocators import DataReference


class TestFunction:
    """Tests for rebuilding of Function types."""

    def test_w_new_dims(self):
        x = Dimension('x')
        y = Dimension('y')
        x0 = Dimension('x0')
        y0 = Dimension('y0')

        f = Function(name='f', dimensions=(x, y), shape=(11, 11))
        f.data[:] = 1

        dims0 = (x0, y0)
        dims1 = (x, y0)

        f0 = f._rebuild(dimensions=dims0)
        f1 = f._rebuild(dimensions=dims1)
        f2 = f._rebuild(dimensions=f.dimensions)
        f3 = f._rebuild(dimensions=dims0,
                        allocator=DataReference(f._data))

        assert f0.function is f0
        assert f0.dimensions == dims0
        assert np.all(f0.data[:] == 0)

        assert f1.function is f1
        assert f1.dimensions == dims1
        assert np.all(f1.data[:] == 0)

        assert f2.function is f
        assert f2.dimensions == f.dimensions
        assert np.all(f2.data[:] == 1)

        assert f3.function is f3
        assert f3.dimensions == dims0
        assert np.all(f3.data[:] == 1)

    def test_w_more_dims(self):
        grid = Grid(shape=(4, 4))
        d = Dimension('d')

        f = Function(name='f', grid=grid, space_order=2)

        with pytest.raises(ValueError):
            # Expected shape to be passed in too
            f._rebuild(dimensions=(d, *f.dimensions))

        f0 = f._rebuild(dimensions=(d, *f.dimensions), shape=(1, 4, 4))
        assert f0.dimensions == (d, *f.dimensions)
        assert f0.shape == (1, 4, 4)
        assert f0.halo == ((0, 0), (2, 2), (2, 2))

        f1 = f._rebuild(dimensions=(d, *f.dimensions), shape=(1, 4, 4),
                        halo=((1, 1), (0, 0), (0, 0)))
        assert f1.dimensions == (d, *f.dimensions)
        assert f1.shape == (1, 4, 4)
        assert f1.halo == ((1, 1), (0, 0), (0, 0))


class TestDimension:

    def test_stencil_dimension(self):
        sd0 = StencilDimension('i', 0, 1)
        sd1 = StencilDimension('i', 0, 1)

        # StencilDimensions are cached by devito so they are guaranteed to be
        # unique for a given set of args/kwargs
        assert sd0 is sd1

        # Same applies to reconstruction
        sd2 = sd0._rebuild()
        assert sd0 is sd2

    @pytest.mark.xfail(reason="Borked caching when supplying a kwarg for an arg")
    def test_stencil_dimension_borked(self):
        sd0 = StencilDimension('i', 0, _max=1)
        sd1 = sd0._rebuild()

        # TODO: Look into Symbol._cache_key and the way the key is generated
        assert sd0 is sd1


class TestSparseFunction:

    @pytest.mark.parametrize('sfunc', [SparseFunction, PrecomputedSparseFunction])
    def test_none_subfunc(self, sfunc):
        grid = Grid((4, 4))
        coords = np.zeros((5, 2))

        s = sfunc(name='s', grid=grid, npoint=5, coordinates=coords, r=1)

        assert s.coordinates is not None

        # Explicity set coordinates to None
        sr = s._rebuild(function=None, initializer=None, coordinates=None)
        assert sr.coordinates is None

import numpy as np
import pytest

from devito import Dimension, Function
from devito.types import StencilDimension
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

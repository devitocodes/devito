import pytest
import numpy as np

from devito import Dimension, Function, Grid, Eq, Operator
from devito.data.allocators import DataReference


class TestRebuild:
    """Tests for rebuilding of Function types."""

    def test_w_new_dims(self):
        x = Dimension('x')
        y = Dimension('y')
        x0 = Dimension('x0')
        y0 = Dimension('y0')

        f = Function(name='f', dimensions=(x, y), shape=(11, 11))

        dims0 = (x0, y0)
        dims1 = (x, y0)

        f0 = f._rebuild(dimensions=dims0)
        f1 = f._rebuild(dimensions=dims1)

        assert f0.function is f0
        assert f0.dimensions == dims0

        assert f1.function is f1
        assert f1.dimensions == dims1


class TestDataReference:
    """
    Tests for passing data to a Function using a reference to a
    preexisting array-like.
    """

    def test_w_array(self):
        """Test using a preexisting NumPy array as Function data"""
        grid = Grid(shape=(3, 3))
        a = np.reshape(np.arange(25, dtype=np.float32), (5, 5))
        b = a.copy()
        c = a.copy()

        b[1:-1, 1:-1] += 1

        f = Function(name='f', grid=grid, space_order=1,
                     allocator=DataReference(a))

        # Check that the array hasn't been zeroed
        assert np.any(a != 0)

        # Check that running operator updates the original array
        Operator(Eq(f, f+1))()
        assert np.all(a == b)

        # Check that updating the array updates the function data
        a[1:-1, 1:-1] -= 1
        assert np.all(f.data_with_halo == c)

    def _w_data(self):
        shape = (5, 5)
        grid = Grid(shape=shape)
        f = Function(name='f', grid=grid, space_order=1)
        f.data_with_halo[:] = np.reshape(np.arange(49, dtype=np.float32), (7, 7))

        g = Function(name='g', grid=grid, space_order=1,
                     allocator=DataReference(f._data),
                     initializer=lambda x: None)

        # Check that the array hasn't been zeroed
        assert np.any(f.data_with_halo != 0)

        assert np.all(f.data_with_halo == g.data_with_halo)

        # Update f
        Operator(Eq(f, f+1))()
        assert np.all(f.data_with_halo == g.data_with_halo)

        # Update g
        Operator(Eq(g, g+1))()
        assert np.all(f.data_with_halo == g.data_with_halo)

        check = np.array(f.data_with_halo[1:-1, 1:-1])

        # Update both
        Operator([Eq(f, f+1), Eq(g, g+1)])()
        assert np.all(f.data_with_halo == g.data_with_halo)
        # Check that it was incremented by two
        check += 2
        assert np.all(f.data == check)

    def test_w_data(self):
        """Test passing preexisting Function data to another Function"""
        self._w_data()

    @pytest.mark.parallel(mode=[2, 4])
    def test_w_data_mpi(self, mode):
        """
        Test passing preexisting Function data to another Function with MPI.
        """
        self._w_data()

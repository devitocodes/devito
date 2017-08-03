import numpy as np
from sympy import Eq  # noqa

import pytest  # noqa

pexpect = pytest.importorskip('yask_compiler')  # Run only if YASK is available

from devito import (Operator, DenseData, TimeData, t, x, y, z,
                    configuration, clear_cache)  # noqa
from devito.yask.interfaces import YaskGrid  # noqa

pytestmark = pytest.mark.skipif(configuration['backend'] != 'yask',
                                reason="'yask' wasn't selected as backend on startup")


@pytest.fixture(scope="module")
def u(dims):
    u = DenseData(name='yu3D', shape=(16, 16, 16), dimensions=(x, y, z), space_order=0)
    u.data  # Trigger initialization
    return u


class TestGrids(object):

    """
    Test YASK grid wrappers.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_data_type(self, u):
        assert type(u._data_object) == YaskGrid

    @pytest.mark.xfail(reason="YASK always seems to use 3D grids")
    def test_data_movement_1D(self):
        u = DenseData(name='yu1D', shape=(16,), dimensions=(x,), space_order=0)
        u.data
        assert type(u._data_object) == YaskGrid

        u.data[1] = 1.
        assert u.data[0] == 0.
        assert u.data[1] == 1.
        assert all(i == 0 for i in u.data[2:])

    def test_data_movement_nD(self, u):
        """
        Tests packing/unpacking data to/from YASK through Devito's YaskGrid.
        """
        # Test simple insertion and extraction
        u.data[0, 1, 1] = 1.
        assert u.data[0, 0, 0] == 0.
        assert u.data[0, 1, 1] == 1.
        assert np.all(u.data == u.data[:, :, :])
        assert 1. in u.data[0]
        assert 1. in u.data[0, 1]

        # Test negative indices
        assert u.data[0, -15, -15] == 1.
        u.data[6, 0, 0] = 1.
        assert u.data[-10, :, :].sum() == 1.

        # Test setting whole array to given value
        u.data[:] = 3.
        assert np.all(u.data == 3.)

        # Test insertion of single value into block
        u.data[5, :, 5] = 5.
        assert np.all(u.data[5, :, 5] == 5.)

        # Test extraction of block with negative indices
        sliced = u.data[-11, :, -11]
        assert sliced.shape == (16,)
        assert np.all(sliced == 5.)

        # Test insertion of block into block
        block = np.ndarray(shape=(1, 16, 1), dtype=np.float32)
        block.fill(4.)
        u.data[4, :, 4] = block
        assert np.all(u.data[4, :, 4] == block)

    def test_data_arithmetic_nD(self, u):
        """
        Tests arithmetic operations through YaskGrids.
        """
        u.data[:] = 1

        # Simple arithmetic
        assert np.all(u.data == 1)
        assert np.all(u.data + 2. == 3.)
        assert np.all(u.data - 2. == -1.)
        assert np.all(u.data * 2. == 2.)
        assert np.all(u.data / 2. == 0.5)
        assert np.all(u.data % 2 == 1.)

        # Increments and parital increments
        u.data[:] += 2.
        assert np.all(u.data == 3.)
        u.data[9, :, :] += 1.
        assert all(np.all(u.data[i, :, :] == 3.) for i in range(9))
        assert np.all(u.data[9, :, :] == 4.)

        # Right operations __rOP__
        u.data[:] = 1.
        arr = np.ndarray(shape=(16, 16, 16), dtype=np.float32)
        arr.fill(2.)
        assert np.all(arr - u.data == -1.)


class TestOperatorExecution(object):

    """
    Test execution of Operators through YASK.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.mark.parametrize("space_order", [0, 1, 2])
    def test_increasing_halo_wo_ofs(self, space_order):
        """
        Use the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1`` to check
        that increasing space orders lead to proportionately larger halo regions.
        For example, with ``space_order = 0``, produce (in 2D view):

            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1

        With ``space_order = 1``, produce:

            0 0 0 ... 0 0
            0 1 1 ... 1 0
            0 1 1 ... 1 0
            0 1 1 ... 1 0
            0 0 0 ... 0 0

        And so on and so forth.
        """
        u = TimeData(name='yu4D', shape=(16, 16, 16), dimensions=(x, y, z),
                     space_order=space_order)
        op = Operator(Eq(u.indexed[t + 1, x, y, z], u.indexed[t, x, y, z] + 1.))
        op(u, t=1)
        lbound, rbound = space_order, 16 - space_order
        written_region = u.data[1, lbound:rbound, lbound:rbound, lbound:rbound]
        assert np.all(written_region == 1.)

    def test_fixed_halo_w_ofs(self):
        """
        Compute an N-point stencil sum, where N is the number of points sorrounding
        an inner (i.e., non-border) grid point.
        For example (in 2D view):

            1 1 1 ... 1 1
            1 4 4 ... 4 1
            1 4 4 ... 4 1
            1 4 4 ... 4 1
            1 1 1 ... 1 1
        """
        space_order = 2
        v = TimeData(name='yv4D', shape=(16, 16, 16), dimensions=(x, y, z),
                     space_order=space_order)
        v.data[:] = 1.
        op = Operator(Eq(v.forward, v.laplace + 6*v),
                      subs={t.spacing: 1, x.spacing: 1, y.spacing: 1, z.spacing: 1})
        op(v, t=1)
        lbound, rbound = space_order, 16 - space_order
        written_region = v.data[1, lbound:rbound, lbound:rbound, lbound:rbound]
        assert np.all(written_region == 6.)

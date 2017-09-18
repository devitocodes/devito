import numpy as np
from sympy import Eq  # noqa

import pytest  # noqa

pexpect = pytest.importorskip('yask_compiler')  # Run only if YASK is available

from devito import (Operator, DenseData, TimeData, PointData,
                    t, x, y, z, configuration, clear_cache, info)  # noqa
from devito.dle import retrieve_iteration_tree  # noqa
from devito.yask.wrappers import YaskGrid, contexts  # noqa

# For the acoustic wave test
from examples.seismic.acoustic import AcousticWaveSolver  # noqa
from examples.seismic import demo_model, RickerSource, Receiver  # noqa

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

        # Increments and partial increments
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
    Test execution of "toy" Operators through YASK.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.mark.parametrize("space_order", [0, 1, 2])
    def test_increasing_halo_wo_ofs(self, space_order):
        """
        Apply the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1`` and check
        that increasing space orders lead to proportionately larger halo regions,
        which are *not* written by the Operator.
        For example, with ``space_order = 0``, produce (in 2D view):

            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1

        With ``space_order = 1``, produce:

            0 0 0 0 0 0 0 0 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 0 0 0 0 0 0 0 0

        And so on and so forth.
        """
        u = TimeData(name='yu4D', shape=(16, 16, 16), dimensions=(x, y, z),
                     space_order=space_order)
        u.data.with_halo[:] = 0.
        op = Operator(Eq(u.forward, u + 1.), subs={t.spacing: 1})
        op(yu4D=u, t=1)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(u.data[1] == 1.)
        # Check that the halo planes are still 0
        assert all(np.all(u.data.with_halo[1, i, :, :] == 0) for i in range(space_order))
        assert all(np.all(u.data.with_halo[1, :, i, :] == 0) for i in range(space_order))
        assert all(np.all(u.data.with_halo[1, :, :, i] == 0) for i in range(space_order))

    def test_increasing_multi_steps(self):
        """
        Apply the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1`` for 11
        timesteps and check that all grid domain values are equal to 11 within
        ``u[1]`` and equal to 10 within ``u[0]``.
        """
        u = TimeData(name='yu4D', shape=(8, 8, 8), dimensions=(x, y, z), space_order=0)
        u.data.with_halo[:] = 0.
        op = Operator(Eq(u.forward, u + 1.), subs={t.spacing: 1})
        op(yu4D=u, t=12)
        assert 'run_solution' in str(op)
        assert np.all(u.data[0] == 10.)
        assert np.all(u.data[1] == 11.)

    @pytest.mark.parametrize("space_order", [2])
    def test_fixed_halo_w_ofs(self, space_order):
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
        v = TimeData(name='yv4D', shape=(16, 16, 16), dimensions=(x, y, z),
                     space_order=space_order)
        v.data.with_halo[:] = 1.
        op = Operator(Eq(v.forward, v.laplace + 6*v),
                      subs={t.spacing: 1, x.spacing: 1, y.spacing: 1, z.spacing: 1})
        op(yv4D=v, t=1)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(v.data[1] == 6.)
        # Check that the halo planes are untouched
        assert all(np.all(v.data.with_halo[1, i, :, :] == 1) for i in range(space_order))
        assert all(np.all(v.data.with_halo[1, :, i, :] == 1) for i in range(space_order))
        assert all(np.all(v.data.with_halo[1, :, :, i] == 1) for i in range(space_order))

    def test_mixed_space_order(self):
        """
        Make sure that no matter whether data objects have different space order,
        as long as they have same domain, the Operator will be executed correctly.
        """
        u = TimeData(name='yu4D', shape=(8, 8, 8), dimensions=(x, y, z), space_order=0)
        v = TimeData(name='yv4D', shape=(8, 8, 8), dimensions=(x, y, z), space_order=1)
        u.data.with_halo[:] = 1.
        v.data.with_halo[:] = 2.
        op = Operator(Eq(v.forward, u + v), subs={t.spacing: 1})
        op(yu4D=u, yv4D=v, t=1)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(v.data[1] == 3.)
        # Check that the halo planes are untouched
        assert np.all(v.data.with_halo[1, 0, :, :] == 2)
        assert np.all(v.data.with_halo[1, :, 0, :] == 2)
        assert np.all(v.data.with_halo[1, :, :, 0] == 2)

    def test_multiple_loop_nests(self):
        """
        Compute a simple stencil S, preceded by an "initialization loop" I and
        followed by a "random loop" R.

            * S is the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1``;
            * I initializes ``u`` to 0;
            * R adds 2 to another field ``v`` along the ``z`` dimension but only
                over the planes ``[x=0, y=2]`` and ``[x=0, y=5]``.

        Out of these three loop nests, only S should be "offloaded" to YASK; indeed,
        I is outside the time loop, while R does not loop over space dimensions.
        This test checks that S is the only loop nest "offloaded" to YASK, and
        that the numerical output is correct.
        """
        u = TimeData(name='yu4D', shape=(12, 12, 12), dimensions=(x, y, z),
                     space_order=0)
        v = TimeData(name='yv4D', shape=(12, 12, 12), dimensions=(x, y, z),
                     space_order=0)
        v.data[:] = 0.
        eqs = [Eq(u.indexed[0, x, y, z], 0),
               Eq(u.indexed[1, x, y, z], 0),
               Eq(u.forward, u + 1.),
               Eq(v.indexed[t + 1, 0, 2, z], v.indexed[t + 1, 0, 2, z] + 2.),
               Eq(v.indexed[t + 1, 0, 5, z], v.indexed[t + 1, 0, 5, z] + 2.)]
        op = Operator(eqs, subs={t.spacing: 1})
        op(yu4D=u, yv4D=v, t=1)
        assert 'run_solution' in str(op)
        assert len(retrieve_iteration_tree(op)) == 3
        assert np.all(u.data[0] == 0.)
        assert np.all(u.data[1] == 1.)
        assert np.all(v.data[0] == 0.)
        assert np.all(v.data[1, 0, 2] == 2.)
        assert np.all(v.data[1, 0, 5] == 2.)

    def test_irregular_write(self):
        """
        Compute a simple stencil S w/o offloading it to YASK because of the presence
        of indirect write accesses (e.g. A[B[i]] = ...); YASK grid functions are however
        used in the generated code to access the data at the right location. This
        test checks that the numerical output is correct after this transformation.

        Initially, the input array (a YASK grid, under the hood), at t=0 is (2D view):

            0 1 2 3
            0 1 2 3
            0 1 2 3
            0 1 2 3

        Then, the Operator "flips" its content, and at timestep t=1 we get (2D view):

            3 2 1 0
            3 2 1 0
            3 2 1 0
            3 2 1 0
        """
        p = PointData(name='points', nt=1, npoint=4)
        u = TimeData(name='yu4D', shape=(4, 4, 4), dimensions=(x, y, z), space_order=0)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u.data[0, i, j, k] = k
        ind = lambda i: p.indexed[0, i]
        eqs = [Eq(p.indexed[0, 0], 3.), Eq(p.indexed[0, 1], 2.),
               Eq(p.indexed[0, 2], 1.), Eq(p.indexed[0, 3], 0.),
               Eq(u.indexed[t + 1, ind(x), ind(y), ind(z)], u.indexed[t, x, y, z])]
        op = Operator(eqs, subs={t.spacing: 1})
        op(yu4D=u, t=1)
        assert 'run_solution' not in str(op)
        assert all(np.all(u.data[1, :, :, i] == 3 - i) for i in range(4))


class TestOperatorRealAcoustic(object):

    """
    Test the acoustic wave model through YASK.

    This test is very similar to the one in test_adjointA.
    """

    presets = {
        'constant': {'preset': 'constant'},
        'layers': {'preset': 'layers', 'ratio': 3},
    }

    @pytest.mark.parametrize('mkey, dimensions, time_order, space_order, nbpml', [
        # 3D tests with varying space orders
        pytest.mark.xfail(('layers', (60, 70, 80), 2, 4, 10)),
        pytest.mark.xfail(('layers', (60, 70, 80), 2, 8, 10)),
    ])
    def test_acoustic(self, mkey, dimensions, time_order, space_order, nbpml):
        t0 = 0.0  # Start time
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create model from preset
        model = demo_model(spacing=[15. for _ in dimensions],
                           shape=dimensions, nbpml=nbpml, **(self.presets[mkey]))

        # Derive timestepping from model spacing
        dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
        nt = int(1 + (tn-t0) / dt)  # Number of timesteps
        time_values = np.linspace(t0, tn, nt)  # Discretized time axis

        # Define source geometry (center of domain, just below surface)
        src = RickerSource(name='src', ndim=model.dim, f0=0.01, time=time_values)
        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = 30.

        # Define receiver geometry (same as source, but spread across x)
        rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=model.dim)
        rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

        # Create solver object to provide relevant operators
        solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                    time_order=time_order,
                                    space_order=space_order)

        # Create adjoint receiver symbol
        srca = Receiver(name='srca', ntime=solver.source.nt,
                        coordinates=solver.source.coordinates.data)

        # Run forward and adjoint operators
        rec, _, _ = solver.forward(save=False)
        solver.adjoint(rec=rec, srca=srca)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = np.dot(srca.data.reshape(-1), solver.source.data)
        term2 = np.linalg.norm(rec.data) ** 2
        info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %12.12f, ratio: %f'
             % (term1, term2, term1 - term2, term1 / term2))
        assert np.isclose(term1, term2, rtol=1.e-5)

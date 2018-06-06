from sympy import cos
import numpy as np

import pytest  # noqa

pexpect = pytest.importorskip('yask')  # Run only if YASK is available

from devito import (Eq, Grid, Dimension, Operator, Constant, Function, TimeFunction,
                    SparseTimeFunction, configuration, clear_cache)  # noqa
from devito.ir.iet import retrieve_iteration_tree  # noqa
from devito.yask.wrappers import contexts  # noqa

# For the acoustic wave test
from examples.seismic.acoustic import AcousticWaveSolver, iso_stencil  # noqa
from examples.seismic import demo_model, TimeAxis, PointSource, RickerSource, Receiver  # noqa

pytestmark = pytest.mark.skipif(configuration['backend'] != 'yask',
                                reason="'yask' wasn't selected as backend on startup")


def setup_module(module):
    """Get rid of any YASK modules generated and JIT-compiled in previous runs.
    This is not strictly necessary for the tests, but it helps in keeping the
    lib directory clean, which may be helpful for offline analysis."""
    contexts.dump()


@pytest.fixture(autouse=True)
def reset_isa():
    """Force back to NO-SIMD after each test, as some tests may optionally
    switch on SIMD."""
    configuration['develop-mode'] = True


class TestOperatorSimple(object):

    """
    Test execution of "toy" Operators through YASK.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.mark.parametrize("space_order", [0, 1, 2])
    @pytest.mark.parametrize("nosimd", [True, False])
    def test_increasing_halo_wo_ofs(self, space_order, nosimd):
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
        # SIMD on/off
        configuration['develop-mode'] = nosimd

        grid = Grid(shape=(16, 16, 16))
        u = TimeFunction(name='yu4D', grid=grid, space_order=space_order)
        u.data_with_halo[:] = 0.
        op = Operator(Eq(u.forward, u + 1.))
        op(yu4D=u, time=0)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(u.data[1] == 1.)
        # Check that the halo planes are still 0
        assert all(np.all(u.data_with_halo[1, i, :, :] == 0)
                   for i in range(u._extent_halo.left[1]))
        assert all(np.all(u.data_with_halo[1, :, i, :] == 0)
                   for i in range(u._extent_halo.left[2]))
        assert all(np.all(u.data_with_halo[1, :, :, i] == 0)
                   for i in range(u._extent_halo.left[3]))

    def test_increasing_multi_steps(self):
        """
        Apply the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1`` for 11
        timesteps and check that all grid domain values are equal to 11 within
        ``u[1]`` and equal to 10 within ``u[0]``.
        """
        grid = Grid(shape=(8, 8, 8))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        u.data_with_halo[:] = 0.
        op = Operator(Eq(u.forward, u + 1.))
        op(yu4D=u, time=10)
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
        grid = Grid(shape=(16, 16, 16))
        v = TimeFunction(name='yv4D', grid=grid, space_order=space_order)
        v.data_with_halo[:] = 1.
        op = Operator(Eq(v.forward, v.laplace + 6*v), subs=grid.spacing_map)
        op(yv4D=v, time=0)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(v.data[1] == 6.)
        # Check that the halo planes are untouched
        assert all(np.all(v.data_with_halo[1, i, :, :] == 1)
                   for i in range(v._extent_halo.left[1]))
        assert all(np.all(v.data_with_halo[1, :, i, :] == 1)
                   for i in range(v._extent_halo.left[2]))
        assert all(np.all(v.data_with_halo[1, :, :, i] == 1)
                   for i in range(v._extent_halo.left[3]))

    def test_mixed_space_order(self):
        """
        Make sure that no matter whether data objects have different space order,
        as long as they have same domain, the Operator will be executed correctly.
        """
        grid = Grid(shape=(8, 8, 8))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        v = TimeFunction(name='yv4D', grid=grid, space_order=1)
        u.data_with_halo[:] = 1.
        v.data_with_halo[:] = 2.
        op = Operator(Eq(v.forward, u + v))
        op(yu4D=u, yv4D=v, time=0)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(v.data[1] == 3.)
        # Check that the halo planes are untouched
        assert np.all(v.data_with_halo[1, 0, :, :] == 2)
        assert np.all(v.data_with_halo[1, :, 0, :] == 2)
        assert np.all(v.data_with_halo[1, :, :, 0] == 2)

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
        grid = Grid(shape=(12, 12, 12))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        v = TimeFunction(name='yv4D', grid=grid, space_order=0)
        v.data[:] = 0.
        eqs = [Eq(u.indexed[0, x, y, z], 0),
               Eq(u.indexed[1, x, y, z], 0),
               Eq(u.forward, u + 1.),
               Eq(v.indexed[t + 1, 0, 2, z], v.indexed[t + 1, 0, 2, z] + 2.),
               Eq(v.indexed[t + 1, 0, 5, z], v.indexed[t + 1, 0, 5, z] + 2.)]
        op = Operator(eqs)
        op(yu4D=u, yv4D=v, time=0)
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
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        p = SparseTimeFunction(name='points', grid=grid, nt=1, npoint=4)
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u.data[0, i, j, k] = k
        ind = lambda i: p.indexed[0, i]
        eqs = [Eq(p.indexed[0, 0], 3.), Eq(p.indexed[0, 1], 2.),
               Eq(p.indexed[0, 2], 1.), Eq(p.indexed[0, 3], 0.),
               Eq(u.indexed[t + 1, ind(x), ind(y), ind(z)], u.indexed[t, x, y, z])]
        op = Operator(eqs, subs=grid.spacing_map)
        op(yu4D=u, time=0)
        assert 'run_solution' not in str(op)
        assert all(np.all(u.data[1, :, :, i] == 3 - i) for i in range(4))

    def test_reverse_time_loop(self):
        """
        Check that YASK evaluates stencil equations correctly when iterating in the
        reverse time direction.
        """
        grid = Grid(shape=(4, 4, 4))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0, time_order=2)
        u.data[:] = 2.
        eq = Eq(u.backward, u - 1.)
        op = Operator(eq)
        op(yu4D=u, time=2)
        assert 'run_solution' in str(op)
        assert np.all(u.data[2] == 2.)
        assert np.all(u.data[1] == 1.)
        assert np.all(u.data[0] == 0.)

    def test_capture_vector_temporaries(self):
        """
        Check that all vector temporaries appearing in a offloaded stencil
        equation are: ::

            * mapped to a YASK grid, directly in Python-land,
            * so no memory needs to be allocated in C-land, and
            * passed down to the generated code, and
            * re-initializaed to 0. at each operator application
        """
        grid = Grid(shape=(4, 4, 4))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        v = Function(name='yv3D', grid=grid, space_order=0)
        eqs = [Eq(u.forward, u + cos(v)*2. + cos(v)*cos(v)*3.)]
        op = Operator(eqs)
        # Sanity check of the generated code
        assert 'posix_memalign' not in str(op)
        assert 'run_solution' in str(op)
        # No data has been allocated for the temporaries yet
        assert op.yk_soln.grids['r1'].is_storage_allocated() is False
        op.apply(yu4D=u, yv3D=v, time=0)
        # Temporary data has already been released after execution
        assert op.yk_soln.grids['r1'].is_storage_allocated() is False
        assert np.all(v.data == 0.)
        assert np.all(u.data[1] == 5.)

    def test_constants(self):
        """
        Check that :class:`Constant` objects are treated correctly.
        """
        grid = Grid(shape=(4, 4, 4))
        c = Constant(name='c', value=2., dtype=grid.dtype)
        p = SparseTimeFunction(name='points', grid=grid, nt=1, npoint=1)
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        u.data[:] = 0.
        op = Operator([Eq(u.forward, u + c), Eq(p.indexed[0, 0], 1. + c)])
        assert 'run_solution' in str(op)
        op.apply(yu4D=u, c=c, time=9)
        # Check YASK did its job and could read constant grids w/o problems
        assert np.all(u.data[0] == 20.)
        # Check the Constant could be read correctly even in Devito-land, i.e.,
        # outside of run_solution
        assert p.data[0][0] == 3.
        # Check re-executing with another constant gives the correct result
        c2 = Constant(name='c', value=5.)
        op.apply(yu4D=u, c=c2, time=2)
        assert np.all(u.data[0] == 30.)
        assert np.all(u.data[1] == 35.)
        assert p.data[0][0] == 6.

    def test_partial_offloading(self):
        """
        Check that :class:`Function` objects not using any :class:`SpaceDimension`
        are computed in Devito-land, rather than via YASK.
        """
        shape = (4, 4, 4)
        grid = Grid(shape=shape)
        dx = Dimension(name='dx')
        dy = Dimension(name='dy')
        dz = Dimension(name='dz')
        x, y, z = grid.dimensions

        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        f = Function(name='f', dimensions=(dx, dy, dz), shape=shape)

        u.data_with_halo[:] = 0.
        f.data[:] = 0.

        eqns = [Eq(u.forward, u + 1.),
                Eq(f, u[1, dx, dy, dz] + 1.)]
        op = Operator(eqns)

        op(time=0)
        assert np.all(u.data[0] == 0.)
        assert np.all(u.data[1] == 1.)
        assert np.all(f.data == 2.)

    def test_repeated_op_calls(self):
        """
        Tests that calling the same Operator with different input data
        produces the expected results.
        """
        grid = Grid(shape=(4, 4, 4))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        u.data[:] = 0.
        op = Operator(Eq(u.forward, u + 1.))
        # First run
        op(time=0)
        assert np.all(u.data[1] == 1.)
        assert u.data[:].sum() == np.prod(grid.shape)
        # Nothing should have changed at this point
        op(time=0, yu4D=u)
        assert np.all(u.data[1] == 1.)
        assert u.data[:].sum() == np.prod(grid.shape)
        # Now try with a different grid
        grid = Grid(shape=(3, 3, 3))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        u.data[:] = 0.
        op(time=0, yu4D=u)
        assert np.all(u.data[1] == 1.)
        assert u.data[:].sum() == np.prod(grid.shape)


class TestOperatorAdvanced(object):

    """
    Test execution of non-trivial Operators through YASK.
    """

    def setup_method(self, method):
        clear_cache()

    def test_misc_dims(self):
        """
        Tests grid-independent :class:`Function`s, which require YASK's "misc"
        dimensions.
        """
        dx = Dimension(name='dx')
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=4, save=4)
        c = Function(name='c', dimensions=(x, dx), shape=(10, 5))

        step = Eq(u.forward, (
            u[time, x-2, y] * c[x, 0]
            + u[time, x-1, y] * c[x, 1]
            + u[time, x, y] * c[x, 2]
            + u[time, x+1, y] * c[x, 3]
            + u[time, x+2, y] * c[x, 4]))

        for i in range(10):
            c.data[i, 0] = 1.0+i
            c.data[i, 1] = 1.0+i
            c.data[i, 2] = 3.0+i
            c.data[i, 3] = 6.0+i
            c.data[i, 4] = 5.0+i

        u.data[:] = 0.0
        u.data[0, 2, :] = 2.0

        op = Operator(step)
        assert 'run_solution' in str(op)

        op(time_m=0, time_M=0)
        assert(np.all(u.data[1, 0, :] == 10.0))
        assert(np.all(u.data[1, 1, :] == 14.0))
        assert(np.all(u.data[1, 2, :] == 10.0))
        assert(np.all(u.data[1, 3, :] == 8.0))
        assert(np.all(u.data[1, 4, :] == 10.0))
        assert(np.all(u.data[1, 5:10, :] == 0.0))


class TestIsotropicAcoustic(object):

    """
    Test the acoustic wave model through YASK.

    This test is very similar to the one in test_adjointA.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.fixture
    def shape(self):
        return (60, 70, 80)

    @pytest.fixture
    def nbpml(self):
        return 10

    @pytest.fixture
    def space_order(self):
        return 4

    @pytest.fixture
    def dtype(self):
        return np.float64

    @pytest.fixture
    def model(self, space_order, shape, nbpml, dtype):
        return demo_model(spacing=[15., 15., 15.], dtype=dtype,
                          space_order=space_order, shape=shape, nbpml=nbpml,
                          preset='layers-isotropic', ratio=3)

    @pytest.fixture
    def time_params(self, model):
        # Derive timestepping from model spacing
        t0 = 0.0  # Start time
        tn = 500.  # Final time
        dt = model.critical_dt
        return t0, tn, dt

    @pytest.fixture
    def m(self, model):
        return model.m

    @pytest.fixture
    def damp(self, model):
        return model.damp

    @pytest.fixture
    def kernel(self):
        return 'OT2'

    @pytest.fixture
    def u(self, model, space_order, kernel):
        return TimeFunction(name='u', grid=model.grid,
                            space_order=space_order, time_order=2)

    @pytest.fixture
    def eqn(self, m, damp, u, kernel):
        t = u.grid.stepping_dim
        return iso_stencil(u, m, t.spacing, damp, kernel)

    @pytest.fixture
    def src(self, model, dtype):
        t0, tn, dt = self.time_params(model)
        time_range = TimeAxis(start=t0, stop=tn, step=dt)  # Discretized time axis
        # Define source geometry (center of domain, just below surface)
        src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range,
                           dtype=dtype)
        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = 30.
        return src

    @pytest.fixture
    def rec(self, model, src, dtype):
        nrec = 130  # Number of receivers
        t0, tn, dt = self.time_params(model)
        time_range = TimeAxis(start=t0, stop=tn, step=dt)
        rec = Receiver(name='rec', grid=model.grid,
                       time_range=time_range,
                       npoint=nrec, dtype=dtype)
        rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]
        return rec

    def test_acoustic_wo_src_wo_rec(self, model, eqn, m, damp, u):
        """
        Test that the acoustic wave equation runs without crashing in absence
        of sources and receivers.
        """
        dt = model.critical_dt
        u.data[:] = 0.0
        op = Operator(eqn, subs=model.spacing_map)
        assert 'run_solution' in str(op)

        op.apply(u=u, m=m, damp=damp, time=10, dt=dt)

    def test_acoustic_w_src_wo_rec(self, model, eqn, m, damp, u, src):
        """
        Test that the acoustic wave equation runs without crashing in absence
        of receivers.
        """
        dt = model.critical_dt
        u.data[:] = 0.0
        eqns = eqn
        eqns += src.inject(field=u.forward, expr=src * dt**2 / m, offset=model.nbpml)
        op = Operator(eqns, subs=model.spacing_map)
        assert 'run_solution' in str(op)

        op.apply(u=u, m=m, damp=damp, src=src, dt=dt)

        exp_u = 154.05

        assert np.isclose(np.linalg.norm(u.data[:]), exp_u, atol=exp_u*1.e-2)

    def test_acoustic_w_src_w_rec(self, model, eqn, m, damp, u, src, rec):
        """
        Test that the acoustic wave equation forward operator produces the correct
        results when running a 3D model also used in ``test_adjointA.py``.
        """
        dt = model.critical_dt
        u.data[:] = 0.0
        eqns = eqn
        eqns += src.inject(field=u.forward, expr=src * dt**2 / m, offset=model.nbpml)
        eqns += rec.interpolate(expr=u, offset=model.nbpml)
        op = Operator(eqns, subs=model.spacing_map)
        assert 'run_solution' in str(op)

        op.apply(u=u, m=m, damp=damp, src=src, rec=rec, dt=dt)

        # The expected norms have been computed "by hand" looking at the output
        # of test_adjointA's forward operator w/o using the YASK backend.
        exp_u = 154.05
        exp_rec = 212.15

        assert np.isclose(np.linalg.norm(u.data[:]), exp_u, atol=exp_u*1.e-2)
        assert np.isclose(np.linalg.norm(rec.data), exp_rec, atol=exp_rec*1.e-2)

    def test_acoustic_adjoint(self, shape, kernel, space_order, nbpml):
        """
        Full acoustic wave test, forward + adjoint operators
        """
        from test_adjoint import TestAdjoint
        TestAdjoint().test_adjoint_F('layers', shape, kernel, space_order, nbpml)

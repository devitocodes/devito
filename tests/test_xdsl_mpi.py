import numpy as np
import pytest
from cached_property import cached_property

from conftest import skipif, _R, assert_blocking
from devito import (Grid, Constant, Function, TimeFunction, SparseFunction,
                    SparseTimeFunction, Dimension, ConditionalDimension, SubDimension,
                    SubDomain, Eq, Ne, Inc, NODE, Operator, norm, inner, configuration,
                    switchconfig, generic_derivative, XDSLOperator)
from devito.data import LEFT, RIGHT
from devito.ir.iet import (Call, Conditional, Iteration, FindNodes, FindSymbols,
                           retrieve_iteration_tree)
from devito.mpi import MPI
from devito.mpi.routines import HaloUpdateCall, MPICall
from examples.seismic.acoustic import acoustic_setup

pytestmark = skipif(['nompi'], whole_module=True)


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
        op.apply(time=1)

        f.data_with_halo[:] = 1.

        xdsl_op = XDSLOperator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
        xdsl_op.__class__ = XDSLOperator

        xdsl_op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 3.)
        if f.grid.distributor.myrank == 0:
            assert f.data_ro_domain[0, 0] == 5.
            assert np.all(f.data_ro_domain[0, 1:] == 7.)
        elif f.grid.distributor.myrank == f.grid.distributor.nprocs - 1:
            assert f.data_ro_domain[0, -1] == 5.
            assert np.all(f.data_ro_domain[0, :-1] == 7.)
        else:
            assert np.all(f.data_ro_domain[0] == 7.)

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d_asymmetric(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        xdsl_op = XDSLOperator(Eq(f.forward, f[t, x+1] + 1))
        xdsl_op.__class__ = XDSLOperator
        xdsl_op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 2.)
        if f.grid.distributor.myrank == 0:
            assert np.all(f.data_ro_domain[0] == 3.)
        else:
            assert np.all(f.data_ro_domain[0, :-1] == 3.)
            assert f.data_ro_domain[0, -1] == 2.

    @pytest.mark.parallel(mode=1)
    def test_trivial_eq_1d_save(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=5)
        f.data_with_halo[:] = 1.

        xdsl_op = XDSLOperator(Eq(f.forward, f[time, x-1] + f[time, x+1] + 1))
        xdsl_op.__class__ = XDSLOperator
        xdsl_op.apply()

        time_M = xdsl_op._prepare_arguments()['time_M']

        assert np.all(f.data_ro_domain[1] == 3.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[-1, time_M:] == 31.)
        else:
            assert np.all(f.data_ro_domain[-1, :-time_M] == 31.)


def gen_serial_norms(shape, so):
    """
    Computes the norms of the outputs in serial mode to compare with
    """
    day = np.datetime64('today')
    try:
        l = np.load("norms%s.npy" % len(shape), allow_pickle=True)
        assert l[-1] == day
    except:
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape],
                                tn=tn, space_order=so, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)
        # Run forward operator
        rec, u, _ = solver.forward()
        Eu = norm(u)
        Erec = norm(rec)

        # Run adjoint operator
        srca, v, _ = solver.adjoint(rec=rec)
        Ev = norm(v)
        Esrca = norm(srca)

        np.save("norms%s.npy" % len(shape), (Eu, Erec, Ev, Esrca, day), allow_pickle=True)


class TestIsotropicAcoustic(object):

    """
    Test the isotropic acoustic wave equation with MPI.
    """
    _shapes = {1: (60,), 2: (60, 70), 3: (60, 70, 80)}
    _so = {1: 12, 2: 8, 3: 4}
    gen_serial_norms((60,), 12)
    gen_serial_norms((60, 70), 8)
    gen_serial_norms((60, 70, 80), 4)

    @cached_property
    def norms(self):
        return {1: np.load("norms1.npy", allow_pickle=True)[:-1],
                2: np.load("norms2.npy", allow_pickle=True)[:-1],
                3: np.load("norms3.npy", allow_pickle=True)[:-1]}

    @pytest.mark.parametrize('shape,kernel,space_order,save', [
        ((60, ), 'OT2', 4, False),
        ((60, 70), 'OT2', 8, False),
    ])
    @pytest.mark.parallel(mode=1)
    def test_adjoint_codegen(self, shape, kernel, space_order, save):
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape], kernel=kernel,
                                tn=500, space_order=space_order, nrec=130,
                                preset='layers-isotropic', dtype=np.float64)
        op_fwd = solver.op_fwd(save=save)
        fwd_calls = FindNodes(Call).visit(op_fwd)

        op_adj = solver.op_adj()
        adj_calls = FindNodes(Call).visit(op_adj)

        assert len(fwd_calls) == 1
        assert len(adj_calls) == 1

    def run_adjoint_F(self, nd):
        """
        Unlike `test_adjoint_F` in test_adjoint.py, here we explicitly check the norms
        of all Operator-evaluated Functions. The numbers we check against are derived
        "manually" from sequential runs of test_adjoint::test_adjoint_F
        """
        Eu, Erec, Ev, Esrca = self.norms[nd]
        shape = self._shapes[nd]
        so = self._so[nd]
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape],
                                tn=tn, space_order=so, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)
        # Run forward operator
        rec, u, _ = solver.forward()

        assert np.isclose(norm(u) / Eu, 1.0)
        assert np.isclose(norm(rec) / Erec, 1.0)

        # Run adjoint operator
        srca, v, _ = solver.adjoint(rec=rec)

        assert np.isclose(norm(v) / Ev, 1.0)
        assert np.isclose(norm(srca) / Esrca, 1.0)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = inner(srca, solver.geometry.src)
        term2 = norm(rec)**2
        assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)


if __name__ == "__main__":
    configuration['mpi'] = True
    # TestDecomposition().test_reshape_left_right()
    # TestOperatorSimple().test_trivial_eq_2d()
    # TestOperatorSimple().test_num_comms('f[t,x-1,y] + f[t,x+1,y]', {'rc', 'lc'})
    # TestFunction().test_halo_exchange_bilateral()
    # TestSparseFunction().test_ownership(((1., 1.), (1., 3.), (3., 1.), (3., 3.)))
    # TestSparseFunction().test_local_indices([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)], [[0.], [1.], [2.], [3.]])  # noqa
    # TestSparseFunction().test_scatter_gather()
    # TestOperatorAdvanced().test_nontrivial_operator()
    # TestOperatorAdvanced().test_interpolation_dup()
    # TestOperatorAdvanced().test_injection_wodup()
    TestIsotropicAcoustic().test_adjoint_F_no_omp()

import numpy as np

from conftest import assert_structure, get_params, get_arrays, check_array
from devito import Eq, Function, TimeFunction, Grid, Operator, cos, sin


class TestBasic(object):

    def test_v0(self):
        """
        Without prematurely expanding derivatives.
        """
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name='u', grid=grid, space_order=4)

        eqn = Eq(u.forward, (u*cos(f)).dx + 1.)

        op0 = Operator(eqn)
        op1 = Operator(eqn, opt=('advanced', {'expand': False}))

        # Check generated code
        for op in [op0, op1]:
            xs, ys, zs = get_params(op, 'x_size', 'y_size', 'z_size')
            arrays = [i for i in get_arrays(op) if i._mem_heap]
            assert len(arrays) == 1
            check_array(arrays[0], ((2, 2), (0, 0), (0, 0)), (xs+4, ys, zs))

        op0.apply(time_M=10)
        op1.apply(time_M=10, u=u1)

        assert np.allclose(u.data, u1.data, rtol=10e-6)

    def test_v1(self):
        """
        Inspired by test_space_invariant_v5, but now try with unexpanded
        derivatives.
        """
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        u = TimeFunction(name='u', grid=grid, space_order=4)
        v = TimeFunction(name='v', grid=grid, space_order=4)
        u1 = TimeFunction(name='u', grid=grid, space_order=4)
        v1 = TimeFunction(name='v', grid=grid, space_order=4)

        eqns = [Eq(u.forward, (u*cos(f)).dx + v + 1.),
                Eq(v.forward, (v*cos(f)).dy + u.forward.dx + 1.)]

        op0 = Operator(eqns)
        op1 = Operator(eqns, opt=('advanced', {'expand': False}))

        # Check generated code
        for op in [op0, op1]:
            xs, ys, zs = get_params(op, 'x_size', 'y_size', 'z_size')
            arrays = get_arrays(op)
            assert len(arrays) == 1
            check_array(arrays[0], ((2, 2), (2, 2), (0, 0)), (xs+4, ys+4, zs))
        assert op1._profiler._sections['section1'].sops == 44

        op0.apply(time_M=10)
        op1.apply(time_M=10, u=u1, v=v1)

        assert np.allclose(u.data, u1.data, rtol=10e-5)
        assert np.allclose(v.data, v1.data, rtol=10e-5)

    def test_v2(self):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)
        v = TimeFunction(name='v', grid=grid, space_order=4)
        u1 = TimeFunction(name='u', grid=grid, space_order=4)
        v1 = TimeFunction(name='v', grid=grid, space_order=4)

        eqns = [Eq(u.forward, (u.dx.dy + v*u.dx + 1.)),
                Eq(v.forward, (v.dy.dx + u.dx.dz + 1.))]

        op0 = Operator(eqns)
        op1 = Operator(eqns, opt=('advanced', {'expand': False,
                                               'blocklevels': 0}))

        # Check generated code -- expect maximal fusion!
        assert_structure(op1,
                         ['t,x,y,z', 't,x,y,z,i0', 't,x,y,z,i1', 't,x,y,z,i1,i0'],
                         't,x,y,z,i0,i1,i0')

        op0.apply(time_M=5)
        op1.apply(time_M=5, u=u1, v=v1)

        assert np.allclose(u.data, u1.data, rtol=10e-3)
        assert np.allclose(v.data, v1.data, rtol=10e-3)

    def test_v3(self):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)
        v = TimeFunction(name='v', grid=grid, space_order=4)
        u1 = TimeFunction(name='u', grid=grid, space_order=4)
        v1 = TimeFunction(name='v', grid=grid, space_order=4)

        eqns = [Eq(u.forward, (u.dx.dy + v*u + 1.)),
                Eq(v.forward, (v + u.dx.dy + 1.))]

        op0 = Operator(eqns)
        op1 = Operator(eqns, opt=('advanced', {'expand': False}))

        # Check generated code -- redundant IndexDerivatives have been caught!
        op1._profiler._sections['section0'].sops == 65

        op0.apply(time_M=5)
        op1.apply(time_M=5, u=u1, v=v1)

        assert np.allclose(u.data, u1.data, rtol=10e-3)
        assert np.allclose(v.data, v1.data, rtol=10e-3)


class TestAdvanced(object):

    def test_fusion_after_unexpansion(self):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)

        eqn = Eq(u.forward, u.dx + u.dy)

        op = Operator(eqn, opt=('advanced', {'expand': False}))

        assert op._profiler._sections['section0'].sops == 21
        assert_structure(op, ['t,x,y', 't,x,y,i0'], 't,x,y,i0')

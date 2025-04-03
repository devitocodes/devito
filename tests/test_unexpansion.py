import numpy as np
import pytest

from conftest import assert_structure, get_params, get_arrays, check_array
from devito import (Buffer, Eq, Function, TimeFunction, Grid, Operator,
                    Coefficient, Substitutions, cos, sin)
from devito.finite_differences import Weights
from devito.arch.compiler import OneapiCompiler
from devito.ir import Expression, FindNodes, FindSymbols
from devito.parameters import switchconfig, configuration
from devito.types import Symbol, Dimension


class TestLoopScheduling:

    def test_backward_dt2(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        v = TimeFunction(name='v', grid=grid, time_order=2)

        eqns = [Eq(v.backward, v + 1.),
                Eq(f, v.dt2)]

        op = Operator(eqns, opt=('advanced', {'openmp': True,
                                              'expand': False}))
        assert_structure(op, ['t,x,y'], 't,x,y')


class TestSymbolicCoeffs:

    def test_numeric_coeffs(self):
        grid = Grid(shape=(11, 11), extent=(10., 10.))

        u = Function(name='u', grid=grid, space_order=2)
        v = Function(name='v', grid=grid, space_order=2)

        opt = ('advanced', {'expand': False})
        w = np.zeros(3)

        # Pure derivative
        Operator(Eq(u, u.dx2(weights=w)), opt=opt).cfunction

        # Mixed derivative
        Operator(Eq(u, u.dx.dx), opt=opt).cfunction

        # Non-perfect mixed derivative
        Operator(Eq(u, (u.dx(weights=w) + v.dx).dx), opt=opt).cfunction

        # Compound expression
        Operator(Eq(u, (v*u.dx).dy(weights=w)), opt=opt).cfunction

    @pytest.mark.parametrize('coeffs,expected', [
        ((7, 7, 7), 3),  # We've had a bug triggered by identical coeffs
        ((5, 7, 9), 3),
    ])
    def test_multiple_cross_derivs(self, coeffs, expected):
        grid = Grid(shape=(11, 11, 11), extent=(10., 10., 10.))
        x, y, z = grid.dimensions

        p = TimeFunction(name='p', grid=grid, space_order=4)

        c0, c1, c2 = coeffs
        coeffs0 = np.full(5, c0)
        coeffs1 = np.full(5, c1)
        coeffs2 = np.full(5, c2)

        eq = Eq(p.forward, p.dy(weights=coeffs1).dz(weights=coeffs2) +
                p.dx(weights=coeffs0).dy(weights=coeffs1))

        op = Operator(eq, opt=('advanced', {'expand': False}))
        op.cfunction

        # w0, w1, ...
        functions = FindSymbols().visit(op)
        weights = {f for f in functions if isinstance(f, Weights)}
        assert len(weights) == expected

    @pytest.mark.parametrize('order', [1, 2])
    @pytest.mark.parametrize('nweight', [None, +4, -4])
    def test_legacy_api(self, order, nweight):
        grid = Grid(shape=(51, 51, 51))
        x, y, z = grid.dimensions

        nweight = 0 if nweight is None else nweight
        so = 8

        u = TimeFunction(name='u', grid=grid, space_order=so,
                         coefficients='symbolic')

        w0 = np.arange(so + 1 + nweight) + 1
        s = f'({x.spacing}*{x.spacing})' if order == 2 else f'{x.spacing}'
        wstr = f'{{{w0[0]:1.1f}F/{s},'
        wdef = f'[{so + 1 + nweight}] __attribute__ ((aligned (64)))'

        coeffs_x_p1 = Coefficient(order, u, x, w0)

        coeffs = Substitutions(coeffs_x_p1)

        eqn = Eq(u, u.dx.dy + u.dx2 + .37, coefficients=coeffs)

        if nweight > 0:
            with pytest.raises(ValueError):
                op = Operator(eqn, opt=('advanced', {'expand': False}))
        else:
            op = Operator(eqn, opt=('advanced', {'expand': False}))
            assert f'{wdef} = {wstr}' in str(op)

    def test_legacy_api_v2(self):
        grid = Grid(shape=(10, 10, 10))
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid, space_order=4)

        cc = np.array([2, 2, 2, 2, 2])
        coeffs = [Coefficient(1, u, d, cc) for d in grid.dimensions]
        coeffs = Substitutions(*coeffs)

        eq0 = Eq(u.forward, u.dx.dz + 1.0)
        eq1 = Eq(u.forward, u.dx.dz + 1.0, coefficients=coeffs)

        op0 = Operator(eq0, opt=('advanced', {'expand': False}))
        op1 = Operator(eq1, opt=('advanced', {'expand': False}))

        assert (op0._profiler._sections['section0'].sops ==
                op1._profiler._sections['section0'].sops)
        weights = [i for i in FindSymbols().visit(op1) if isinstance(i, Weights)]
        w0, w1 = sorted(weights, key=lambda i: i.name)
        assert all(i.args[1] == 1/x.spacing for i in w0.weights)
        assert all(i.args[1] == 1/z.spacing for i in w1.weights)


class Test1Pass:

    def test_v0(self):
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

    def test_fusion_after_unexpansion(self):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)

        eqn = Eq(u.forward, u.dx + u.dy)

        op = Operator(eqn, opt=('advanced', {'expand': False}))

        assert op._profiler._sections['section0'].sops == 21
        assert_structure(op, ['t,x,y', 't,x,y,i0'], 't,x,y,i0')

    @switchconfig(condition=isinstance(configuration['compiler'],
                  (OneapiCompiler)), safe_math=True)
    def test_v1(self):
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
                                               'blocklevels': 0,
                                               'cire-mingain': 200}))

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
        op1 = Operator(eqns, opt=('advanced', {'expand': False,
                                               'cire-mingain': 200}))

        # Check generated code -- redundant IndexDerivatives have been caught!
        op1._profiler._sections['section0'].sops == 65

        op0.apply(time_M=5)
        op1.apply(time_M=5, u=u1, v=v1)

        assert np.allclose(u.data, u1.data, rtol=10e-3)
        assert np.allclose(v.data, v1.data, rtol=10e-3)

    def test_v4(self):
        grid = Grid(shape=(16, 16, 16))

        eqns = tti_sa_eqns(grid)

        op = Operator(eqns, subs=grid.spacing_map,
                      opt=('advanced', {'expand': False,
                                        'cire-mingain': 400}))

        # Check code generation
        assert op._profiler._sections['section1'].sops == 1443
        assert_structure(op, ['x,y,z',
                              't,x0_blk0,y0_blk0,x,y,z',
                              't,x0_blk0,y0_blk0,x,y,z,i1',
                              't,x0_blk0,y0_blk0,x,y,z,i1,i0'],
                         'x,y,z,t,x0_blk0,y0_blk0,x,y,z,i1,i0')

        op.cfunction

    def test_v5(self):
        grid = Grid(shape=(16, 16))

        p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=4,
                          save=Buffer(2))
        m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=4,
                          save=Buffer(2))

        eqns = [Eq(p0.forward, (p0.dx + m0.dx).dx + p0.backward),
                Eq(m0.forward, m0.dx.dx + m0.backward)]

        op = Operator(eqns, subs=grid.spacing_map,
                      opt=('advanced', {'expand': False,
                                        'cire-mingain': 200}))

        # Check code generation
        assert op._profiler._sections['section0'].sops == 127
        assert_structure(op, ['t,x,y', 't,x,y,i1', 't,x,y,i1,i0'], 't,x,y,i1,i0')

        op.cfunction

    def test_v6(self):
        grid = Grid(shape=(16, 16))

        f = Function(name='f', grid=grid, space_order=4)
        p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=4,
                          save=Buffer(2))
        m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=4,
                          save=Buffer(2))

        s0 = Symbol(name='s0', dtype=np.float32)

        eqns = [Eq(p0.forward, (p0.dx + m0.dx).dx + p0.backward),
                Eq(s0, 4., implicit_dims=p0.dimensions),
                Eq(m0.forward, (m0.dx + s0).dx + f*m0.backward)]

        op = Operator(eqns, subs=grid.spacing_map,
                      opt=('advanced', {'expand': False,
                                        'cire-mingain': 200}))

        # Check code generation
        assert op._profiler._sections['section0'].sops == 133
        assert_structure(op, ['t,x,y', 't,x,y,i1', 't,x,y,i1,i0'], 't,x,y,i1,i0')

        op.cfunction

    def test_transpose(self):
        shape = (11, 11, 11)
        grid = Grid(shape=shape, extent=(10, 10, 10))
        x, _, _ = grid.dimensions

        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name='u', grid=grid, space_order=4)

        # Chessboard-like init
        hshape = u.data_with_halo.shape[1:]
        u.data_with_halo[:] = np.indices(hshape).sum(axis=0) % 10 + 1
        u1.data_with_halo[:] = np.indices(hshape).sum(axis=0) % 10 + 1

        eqn = Eq(u.forward, u.dx(x0=x+x.spacing/2).T + 1.)

        op0 = Operator(eqn)
        op1 = Operator(eqn, opt=('advanced', {'expand': False}))

        op0.apply(time_M=10)
        op1.apply(time_M=10, u=u1)
        assert np.allclose(u.data, u1.data, rtol=10e-6)

    def test_redundant_derivatives(self):
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=4)

        # It's the same `u.dx.dy` appearing multiple times, and the cost models
        # must be smart enough to detect that!
        eq = Eq(u.forward, (f*u.dx.dy + g*u.dx.dy + h*u.dx.dy +
                            (f*g)*u.dx.dy + (f*h)*u.dx.dy + (g*h)*u.dx.dy))

        op = Operator(eq, opt=('advanced', {'expand': False,
                                            'blocklevels': 0}))

        # Check generated code
        nlin = 10 if op._options['linearize'] else 0
        assert len(get_arrays(op)) == 0
        assert op._profiler._sections['section0'].sops == 74
        exprs = FindNodes(Expression).visit(op)
        assert len(exprs) == 5 + nlin
        temps = [i for i in FindSymbols().visit(exprs) if isinstance(i, Symbol)]
        assert len(temps) == 2 + nlin

        op.cfunction

    def test_buffering_timestencil(self):
        grid = Grid((11, 11))
        so = 4
        nt = 11

        u = TimeFunction(name="u", grid=grid, space_order=so, time_order=2, save=nt)
        v = TimeFunction(name="v", grid=grid, space_order=so, time_order=2)

        g = Function(name="g", grid=grid, space_order=so)

        # Make sure operator builds with buffering
        op = Operator([Eq(g, g + u.dt*v.dx + u.dx2)],
                      opt=('buffering', 'streaming', {'expand': False}))

        exprs = FindNodes(Expression).visit(op)
        dims = [d for i in FindSymbols().visit(exprs) for d in i.dimensions
                if isinstance(d, Dimension)]

        # Should only be two stencil dimension for .dx and .dx2
        assert len([d for d in dims if d.is_Stencil]) == 2
        # Should only be one buffer dimension
        assert len([d for d in dims if d.is_Custom]) == 1


class Test2Pass:

    @switchconfig(safe_math=True)
    def test_v0(self):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=8)
        v = TimeFunction(name='v', grid=grid, space_order=8)
        u1 = TimeFunction(name='u', grid=grid, space_order=8)
        v1 = TimeFunction(name='v', grid=grid, space_order=8)

        eqns = [Eq(u.forward, (u.dx.dy + v*u + 1.)),
                Eq(v.forward, (v + u.dx.dz + 1.))]

        op0 = Operator(eqns)
        op1 = Operator(eqns, opt=('advanced', {'expand': False,
                                               'openmp': True}))

        # Check generated code
        assert op1._profiler._sections['section0'].sops == 59
        assert_structure(op1, ['t',
                               't,x0_blk0,y0_blk0,x,y,z',
                               't,x0_blk0,y0_blk0,x,y,z,i0',
                               't,x0_blk0,y0_blk0,x,y,z',
                               't,x0_blk0,y0_blk0,x,y,z,i1'],
                         't,x0_blk0,y0_blk0,x,y,z,i0,y,z,i1')

        op0.apply(time_M=5)
        op1.apply(time_M=5, u=u1, v=v1)

        assert np.allclose(u.data, u1.data, rtol=10e-3)
        assert np.allclose(v.data, v1.data, rtol=10e-3)

    def test_v1(self):
        grid = Grid(shape=(16, 16, 16))

        eqns = tti_sa_eqns(grid)

        op = Operator(eqns, subs=grid.spacing_map,
                      opt=('advanced', {'expand': False,
                                        'openmp': False}))

        # Check code generation
        assert op._profiler._sections['section1'].sops == 191
        assert_structure(op, ['x,y,z',
                              't,x0_blk0,y0_blk0,x,y,z',
                              't,x0_blk0,y0_blk0,x,y,z,i0',
                              't,x0_blk0,y0_blk0,x,y,z',
                              't,x0_blk0,y0_blk0,x,y,z,i1'],
                         'x,y,z,t,x0_blk0,y0_blk0,x,y,z,i0,x,y,z,i1')

        op.cfunction

    def test_diff_first_deriv(self):
        grid = Grid(shape=(16, 16, 16))

        u = TimeFunction(name='u', grid=grid, space_order=16)

        eq = Eq(u.forward, u.dy2.dz + u.dy.dx + 1)

        op = Operator(eq, opt=('advanced', {'expand': False}))

        xs, ys, zs = get_params(op, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = get_arrays(op)
        assert len(arrays) == 2
        check_array(arrays[0], ((8, 8), (0, 0), (8, 8)), (xs+16, ys, zs+16))
        check_array(arrays[1], ((8, 8), (0, 0), (8, 8)), (xs+16, ys, zs+16))


def tti_sa_eqns(grid):
    t = grid.stepping_dim
    x, y, z = grid.dimensions

    so = 4

    a = Function(name='a', grid=grid, space_order=so)
    f = Function(name='f', grid=grid, space_order=so)
    e = Function(name='e', grid=grid, space_order=so)
    r = Function(name='r', grid=grid, space_order=so)
    p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=so)
    m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=so)

    def g1(field, r, e):
        return (cos(e) * cos(r) * field.dx(x0=x+x.spacing/2) +
                cos(e) * sin(r) * field.dy(x0=y+y.spacing/2) -
                sin(e) * field.dz(x0=z+z.spacing/2))

    def g2(field, r, e):
        return - (sin(r) * field.dx(x0=x+x.spacing/2) -
                  cos(r) * field.dy(x0=y+y.spacing/2))

    def g3(field, r, e):
        return (sin(e) * cos(r) * field.dx(x0=x+x.spacing/2) +
                sin(e) * sin(r) * field.dy(x0=y+y.spacing/2) +
                cos(e) * field.dz(x0=z+z.spacing/2))

    def g1_tilde(field, r, e):
        return ((cos(e) * cos(r) * field).dx(x0=x-x.spacing/2) +
                (cos(e) * sin(r) * field).dy(x0=y-y.spacing/2) -
                (sin(e) * field).dz(x0=z-z.spacing/2))

    def g2_tilde(field, r, e):
        return - ((sin(r) * field).dx(x0=x-x.spacing/2) -
                  (cos(r) * field).dy(x0=y-y.spacing/2))

    def g3_tilde(field, r, e):
        return ((sin(e) * cos(r) * field).dx(x0=x-x.spacing/2) +
                (sin(e) * sin(r) * field).dy(x0=y-y.spacing/2) +
                (cos(e) * field).dz(x0=z-z.spacing/2))

    update_p = t.spacing**2 * a**2 / f * \
        (g1_tilde(f * g1(p0, r, e), r, e) +
         g2_tilde(f * g2(p0, r, e), r, e) +
         g3_tilde(f * g3(p0, r, e) + f * g3(m0, r, e), r, e)) + \
        (2 - t.spacing * a)

    update_m = t.spacing**2 * a**2 / f * \
        (g1_tilde(f * g1(m0, r, e), r, e) +
         g2_tilde(f * g2(m0, r, e), r, e) +
         g3_tilde(f * g3(m0, r, e) + f * g3(p0, r, e), r, e)) + \
        (2 - t.spacing * a)

    eqns = [Eq(p0.forward, update_p),
            Eq(m0.forward, update_m)]

    return eqns

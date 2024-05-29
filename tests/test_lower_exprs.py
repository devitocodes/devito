import numpy as np
import pytest

from devito import (Grid, TimeFunction, Function, Operator, Eq, solve,
                    DefaultDimension)
from devito.finite_differences import Derivative
from devito.finite_differences.differentiable import diff2sympy
from devito.ir.equations import LoweredEq
from devito.passes.equations.linearity import collect_derivatives
from devito.tools import timed_region


class TestCollectDerivatives:

    """
    Test collect_derivatives and all mechanisms used by collect_derivatives
    indirectly.
    """

    def test_nocollection_if_diff_dims(self):
        """
        Test that expressions with different time dimensions are not collected.
        """
        grid = Grid((10,))

        f = TimeFunction(name="f", grid=grid, save=10)
        f2 = TimeFunction(name="f2", grid=grid, save=10)
        g = TimeFunction(name="g", grid=grid)
        g2 = TimeFunction(name="g2", grid=grid)
        w = Function(name="w", grid=grid)

        with timed_region('x'):
            eq = Eq(w, f.dt*g + f2.dt*g2)

            # Since all Function are time dependent, there should be no collection
            # and produce the same result as with the pre evaluated expression
            expr = Operator._lower_exprs([eq], options={})[0]
            expr2 = Operator._lower_exprs([eq.evaluate], options={})[0]

        assert expr == expr2

    def test_numeric_constant(self):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        eq = Eq(u.forward, u.dx.dx + 0.3*u.dy.dx)
        leq = collect_derivatives.func([eq])[0]

        assert len(leq.find(Derivative)) == 3

    def test_symbolic_constant(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        eq = Eq(u.forward, u.dx.dx + dt**0.2*u.dy.dx)
        leq = collect_derivatives.func([eq])[0]

        assert len(leq.find(Derivative)) == 3

    def test_symbolic_constant_times_add(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)
        f = Function(name='f', grid=grid, space_order=4)

        eq = Eq(u.forward, u.laplace + dt**0.2*u.biharmonic(1/f))

        leq = collect_derivatives.func([eq])[0]

        assert len(eq.rhs.args) == 3
        assert len(leq.rhs.args) == 2
        assert all(isinstance(i, Derivative) for i in leq.rhs.args)

    def test_solve(self):
        """
        By remaining unevaluated until after Operator's collect_derivatives,
        the Derivatives after a solve() should be collected.
        """
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        pde = u.dt2 - (u.dx.dx + u.dy.dy) - u.dx.dy
        eq = Eq(u.forward, solve(pde, u.forward))
        leq = collect_derivatives.func([eq])[0]

        assert len(eq.rhs.find(Derivative)) == 5
        assert len(leq.rhs.find(Derivative)) == 4
        assert len(leq.rhs.args[2].find(Derivative)) == 3  # Check factorization

    def test_nocollection_if_unworthy(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid)

        eq = Eq(u.forward, (0.4 + dt)*(u.dx + u.dy))
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq

    def test_pull_and_collect(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, _ = grid.spacing_symbols

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid)

        eq = Eq(u.forward, ((0.4 + dt)*u.dx + 0.3)*hx + v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        args = leq.rhs.args
        assert len(args) == 2
        assert diff2sympy(args[0]) == 0.3*hx
        assert args[1] == (hx*(dt + 0.4)*u + v).dx

    def test_pull_and_collect_nested(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, hy = grid.spacing_symbols

        u = TimeFunction(name="u", grid=grid, space_order=2)
        v = TimeFunction(name="v", grid=grid, space_order=2)

        eq = Eq(u.forward, (((0.4 + dt)*u.dx + 0.3)*hx + v.dx).dy + (0.2 + hy)*v.dy)
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        assert leq.rhs == ((v + hx*(0.4 + dt)*u).dx + 0.3*hx + (0.2 + hy)*v).dy

    def test_pull_and_collect_nested_v2(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, hy = grid.spacing_symbols

        u = TimeFunction(name="u", grid=grid, space_order=2)
        v = TimeFunction(name="v", grid=grid, space_order=2)

        eq = Eq(u.forward, ((0.4 + dt*(hy + 1. + hx*hy))*u.dx + 0.3)*hx + v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        assert leq.rhs == 0.3*hx + (hx*(0.4 + dt*(hy + 1. + hx*hy))*u + v).dx

    def test_pull_and_collect_nested_v3(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, hy = grid.spacing_symbols

        a = Function(name="a", grid=grid, space_order=2)
        u = TimeFunction(name="u", grid=grid, space_order=2)
        v = TimeFunction(name="v", grid=grid, space_order=2)

        eq = Eq(u.forward, 0.4 + a*(hx + dt*(u.dx + v.dx)))
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        assert leq.rhs == 0.4 + a*(hx + (dt*u + dt*v).dx)

    def test_nocollection_subdims(self):
        grid = Grid(shape=(10, 10))
        xi, yi = grid.interior.dimensions

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid)
        f = Function(name='f', grid=grid)

        eq = Eq(u.forward, u.dx + 0.2*f[xi, yi]*v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq

    def test_nocollection_staggered(self):
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid, staggered=x)

        eq = Eq(u.forward, u.dx + v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq

    def test_nocollection_mixed_order(self):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=2)

        # First case is obvious...
        eq = Eq(u.forward, u.dx2 + u.dx.dy + 1.)
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq

        # y-derivative should not get collected!
        eq = Eq(u.forward, u.dy2 + u.dx.dy + 1.)
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq


class TestLowering:

    """
    Test that expression lowering works as expected.
    """

    def test_lower_func_as_ind(self):
        grid = Grid((11, 11))
        x, y = grid.dimensions
        t = grid.stepping_dim
        h = DefaultDimension("h", default_value=10)

        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)
        oh = Function(name="ou", dimensions=(h,), shape=(10,), dtype=int)

        eq = [Eq(u.forward, u._subs(x, x + oh))]
        lowered = LoweredEq(Eq(u[t + 1, x + 2, y + 2], u[t, x + oh[h] + 2, y + 2]))

        with timed_region('x'):
            leq = Operator._lower_exprs(eq, options={})

        assert leq[0] == lowered


class TestUnexpanded:

    @pytest.mark.parametrize('expr', [
        'u.dx',
        'u.dx.dy',
        'u.dxdy',
        'u.dx.dy + u.dy.dx',
        'u.dx2 + u.dx.dy',
    ])
    def test_single_eq(self, expr):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=4)
        u1 = TimeFunction(name="u", grid=grid, space_order=4)

        eq = Eq(u.forward, eval(expr) + 1.)

        op0 = Operator(eq)
        op1 = Operator(eq, opt=('advanced', {'expand': False}))

        op0.apply(time_M=5)
        op1.apply(time_M=5, u=u1)

        assert np.allclose(u.data, u1.data, rtol=1e-5)

import numpy as np
import sympy as sp
import pytest

from devito import (Grid, Function, TimeFunction, Eq,
                    Dimension, solve, Operator, div, grad, laplace)
from devito.finite_differences import Differentiable
from devito.finite_differences.coefficients import Coefficient, Substitutions
from devito.finite_differences.finite_difference import _PRECISION
from devito.symbolics import retrieve_derivatives
from devito.tools import as_tuple


class TestSC:
    """
    Class for testing symbolic coefficients functionality
    """

    @pytest.mark.parametrize('expr, sorder, dorder, dim, weights, expected', [
        ('u.dx2', 2, 2, 0, (-0.6, 0.1, 0.6),
         '0.1*u - 0.6*u._subs(x, x - h_x) + 0.6*u._subs(x, x + h_x)'),
        ('u.dy2', 4, 2, 1, (0.121, -0.223, 1.648, -2.904, 0),
         '1.648*u + 0.121*u._subs(y, y - 2*h_y) - 0.223*u._subs(y, y - h_y) \
- 2.904*u._subs(y, y + h_y)')])
    def test_coefficients(self, expr, sorder, dorder, dim, weights, expected):
        """Test that custom coefficients return the expected result"""
        grid = Grid(shape=(10, 10))
        u = Function(name='u', grid=grid, space_order=sorder, coefficients='symbolic')
        x, y = grid.dimensions
        h_x, h_y = grid.spacing_symbols  # noqa

        order = dorder
        dim = grid.dimensions[dim]
        weights = np.array(weights)

        coeffs = Coefficient(order, u, dim, weights)

        eq = Eq(eval(expr), coefficients=Substitutions(coeffs))
        deriv = retrieve_derivatives(eq.lhs)[0]
        assert np.all(deriv.weights == weights)

        assert isinstance(eq.lhs, Differentiable)
        s = dim.spacing**(-dorder)
        assert sp.simplify(eval(expected).evalf(_PRECISION) * s - eq.evaluate.lhs) == 0

    def test_function_coefficients(self):
        """Test that custom function coefficients return the expected result"""
        so = 2
        grid = Grid(shape=(4, 4))
        f0 = TimeFunction(name='f0', grid=grid, space_order=so)
        f1 = TimeFunction(name='f1', grid=grid, space_order=so)
        x, y = grid.dimensions

        s = Dimension(name='s')
        ncoeffs = so+1

        wshape = list(grid.shape)
        wshape.append(ncoeffs)
        wshape = as_tuple(wshape)

        wdims = list(grid.dimensions)
        wdims.append(s)
        wdims = as_tuple(wdims)

        w = Function(name='w', dimensions=wdims, shape=wshape)
        w.data[:, :, 0] = 0.0
        w.data[:, :, 1] = -1.0/grid.spacing[0]
        w.data[:, :, 2] = 1.0/grid.spacing[0]

        eq0 = Eq(f0.dt + f0.dx(weights=w), 1)
        eq1 = Eq(f1.dt + f1.dx, 1)

        stencil0 = solve(eq0.evaluate, f0.forward)
        stencil1 = solve(eq1.evaluate, f1.forward)

        op0 = Operator(Eq(f0.forward, stencil0))
        op1 = Operator(Eq(f1.forward, stencil1))

        op0(time_m=0, time_M=5, dt=1.0)
        op1(time_m=0, time_M=5, dt=1.0)

        assert np.all(np.isclose(f0.data[:] - f1.data[:], 0.0, atol=1e-5, rtol=0))

    def test_function_coefficients_xderiv_legacy(self):
        p = Dimension('p')

        nstc = 8

        grid = Grid(shape=(51, 51, 51))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid, space_order=(2*nstc, 0, 0),
                     coefficients='symbolic')
        g = Function(name='g', grid=grid, space_order=(2*nstc, 0, 0))
        ax = Function(name='DD2x', space_order=0, shape=(2*nstc + 1,),
                      dimensions=(p,))
        ay = Function(name='DD2y', space_order=0, shape=(2*nstc + 1,),
                      dimensions=(p,))
        stencil_coeffs_x_p1 = Coefficient(1, f, x, ax)
        stencil_coeffs_y_p1 = Coefficient(1, f, y, ay)
        stencil_coeffs = Substitutions(stencil_coeffs_x_p1, stencil_coeffs_y_p1)

        eqn = Eq(g, f.dxdy, coefficients=stencil_coeffs)

        op = Operator(eqn)
        op()

    @pytest.mark.parametrize('order', [2, 4, 6, 8])
    def test_function_coefficients_xderiv(self, order):
        p = Dimension('p')

        grid = Grid(shape=(51, 51, 51))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid, space_order=order)
        w = Function(name='w', space_order=0, shape=(*grid.shape, order + 1),
                     dimensions=(x, y, z, p))

        expr0 = f.dx(w=w).dy(w=w).evaluate
        expr1 = f.dxdy(w=w).evaluate
        assert sp.simplify(expr0 - expr1) == 0

    def test_coefficients_expr(self):
        p = Dimension('p')

        grid = Grid(shape=(51, 51, 51))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid, space_order=4)
        w = Function(name='w', space_order=0, shape=(*grid.shape, 5),
                     dimensions=(x, y, z, p))

        expr0 = f.dx(w=w/x.spacing).evaluate
        expr1 = f.dx(w=w).evaluate / x.spacing
        assert sp.simplify(expr0 - expr1) == 0

    def test_coefficients_w_xreplace(self):
        """Test custom coefficients with an xreplace before they are applied"""
        grid = Grid(shape=(4, 4))
        u = Function(name='u', grid=grid, space_order=2)
        x = grid.dimensions[0]
        h_x = x.spacing

        weights = np.array([-0.6, 0.1, 0.6])

        c = sp.Symbol('c')

        eq = Eq(u.dx2(weights=weights) + c)
        eq = eq.xreplace({c: 2})

        s = x.spacing**(-2)
        expected = (0.1*u - 0.6*u._subs(x, x - h_x) + 0.6*u.subs(x, x + h_x)) * s + 2

        assert sp.simplify(expected.evalf(_PRECISION) - eq.evaluate.lhs) == 0

    @pytest.mark.parametrize('order', [2, 4, 6])
    def test_staggered_array(self, order):
        """Test custom coefficients provided as an array on a staggered grid"""
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=order)
        g = Function(name='g', grid=grid, space_order=order, staggered=x)
        f.data[::2] = 1
        g.data[::2] = 1

        weights = np.ones(order+1)/grid.spacing[0]**2

        eq_f = Eq(f, f.dx2(weights=weights))
        eq_g = Eq(g, g.dx2(weights=weights))

        op = Operator([eq_f, eq_g])
        op()

        assert np.allclose(f.data, g.data, atol=1e-7)

    @pytest.mark.parametrize('order', [2, 4, 6])
    def test_staggered_function(self, order):
        """Test custom function coefficients on a staggered grid"""
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=order)
        g = Function(name='g', grid=grid, space_order=order, staggered=x)
        f.data[::2] = 1
        g.data[::2] = 1

        s = Dimension(name='s')
        ncoeffs = order+1

        wshape = grid.shape + (ncoeffs,)
        wdims = grid.dimensions + (s,)

        w = Function(name='w', dimensions=wdims, shape=wshape)
        w.data[:] = 1.0/grid.spacing[0]**2

        eq_f = Eq(f, f.dx2(weights=w))
        eq_g = Eq(g, g.dx2(weights=w))

        Operator([eq_f, eq_g])()

        assert np.allclose(f.data, g.data, atol=1e-7)

    def test_staggered_function_evalat(self):
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=2)
        g = Function(name='g', grid=grid, space_order=2, staggered=x)

        w = Function(name='w', space_order=0, shape=(2,),
                     dimensions=(Dimension("p"),))

        eq_fg = Eq(f, g.dx2(weights=w))

        expected = 'Eq(f(x), g(x - h_x/2)*w(0) + g(x + h_x/2)*w(1))'

        assert str(eq_fg.evaluate) == expected

    def test_staggered_equation(self):
        """
        Check that expressions with substitutions are consistent with
        those without
        """
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=2, staggered=x)

        weights = np.array([1, -2, 1])/grid.spacing[0]**2

        eq_f = Eq(f, f.dx2(weights=weights))

        expected = 'Eq(f(x + h_x/2), f(x - h_x/2)/h_x**2 - 2.0*f(x + h_x/2)/h_x**2 '\
            '+ f(x + 3*h_x/2)/h_x**2)'
        assert(str(eq_f.evaluate) == expected)

    @pytest.mark.parametrize('stagger', [True, False])
    def test_with_timefunction(self, stagger):
        """Check compatibility of custom coefficients and TimeFunctions"""
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]
        if stagger:
            staggered = x
        else:
            staggered = None

        f = TimeFunction(name='f', grid=grid, space_order=2, staggered=staggered)
        g = TimeFunction(name='g', grid=grid, space_order=2, staggered=staggered)

        f.data[:, ::2] = 1
        g.data[:, ::2] = 1

        weights = np.array([-1, 2, -1])/grid.spacing[0]**2

        eq_f = Eq(f.forward, f.dx2)
        eq_g = Eq(g.forward, g.dx2(weights=weights))

        Operator([eq_f, eq_g])(t_m=0, t_M=1)

        assert np.allclose(f.data[-1], -g.data[-1], atol=1e-7)

    def test_nested_subs(self):
        grid = Grid(shape=(11, 11))
        x, y = grid.dimensions
        hx, hy = grid.spacing_symbols

        p = TimeFunction(name='p', grid=grid, space_order=2)

        coeffs0 = np.array([100, 100, 100])
        coeffs1 = np.array([200, 200, 200])

        eq = Eq(p.forward, p.dx2(weights=coeffs0).dy2(weights=coeffs1))

        s = x.spacing**(-2) * y.spacing**(-2)
        mul = lambda e: sp.Mul(e, 200 * s, evaluate=False)
        term0 = mul(p*100 +
                    p.subs(x, x-hx)*100 +
                    p.subs(x, x+hx)*100)
        term1 = mul(p.subs(y, y-hy)*100 +
                    p.subs({x: x-hx, y: y-hy})*100 +
                    p.subs({x: x+hx, y: y-hy})*100)
        term2 = mul(p.subs(y, y+hy)*100 +
                    p.subs({x: x-hx, y: y+hy})*100 +
                    p.subs({x: x+hx, y: y+hy})*100)

        # `str` simply because some objects are of type EvalDerivative
        assert sp.simplify(eq.evaluate.rhs - term0 - term1 - term2) == 0

    def test_compound_subs(self):
        grid = Grid(shape=(11,))
        x, = grid.dimensions
        hx, = grid.spacing_symbols

        f = Function(name='f', grid=grid, space_order=2)
        p = TimeFunction(name='p', grid=grid, space_order=2)

        coeffs0 = np.array([100, 100, 100])

        eq = Eq(p.forward, (f*p).dx2(weights=coeffs0))

        term0 = f*p*100
        term1 = (f*p*100).subs(x, x-hx)
        term2 = (f*p*100).subs(x, x+hx)
        s = x.spacing**(-2)

        # `str` simply because some objects are of type EvalDerivative
        assert sp.simplify(eq.evaluate.rhs - (term0 + term1 + term2) * s) == 0

    def test_compound_nested_subs(self):
        grid = Grid(shape=(11, 11))
        x, y = grid.dimensions
        hx, hy = grid.spacing_symbols

        f = Function(name='f', grid=grid, space_order=2)
        p = TimeFunction(name='p', grid=grid, space_order=2)

        coeffs0 = np.array([100, 100, 100])
        coeffs1 = np.array([200, 200, 200])

        eq = Eq(p.forward, (f*p.dx2(weights=coeffs0)).dy2(weights=coeffs1))

        mul = lambda e, i: sp.Mul(f.subs(y, y+i*hy), e, 200, evaluate=False)
        term0 = mul(p*100 +
                    p.subs(x, x-hx)*100 +
                    p.subs(x, x+hx)*100, 0)
        term1 = mul(p.subs(y, y-hy)*100 +
                    p.subs({x: x-hx, y: y-hy})*100 +
                    p.subs({x: x+hx, y: y-hy})*100, -1)
        term2 = mul(p.subs(y, y+hy)*100 +
                    p.subs({x: x-hx, y: y+hy})*100 +
                    p.subs({x: x+hx, y: y+hy})*100, 1)

        s = x.spacing**(-2) * y.spacing**(-2)
        # `str` simply because some objects are of type EvalDerivative
        assert sp.simplify(eq.evaluate.rhs - (term0 + term1 + term2) * s) == 0

    def test_operators(self):
        grid = Grid(shape=(11, 11))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid, space_order=2)

        coeffs0 = [100, 100, 100]

        # Div
        expr0 = div(f, w=coeffs0)
        assert expr0 == f.dx(weights=coeffs0) + f.dy(weights=coeffs0)
        assert list(expr0.args[0].weights) == coeffs0

        # Grad
        expr2 = grad(f, w=coeffs0)
        assert expr2[0] == f.dx(weights=coeffs0)
        assert expr2[1] == f.dy(weights=coeffs0)
        assert list(expr2[0].weights) == coeffs0

        # Laplace
        expr3 = laplace(f, w=coeffs0)
        assert expr3 == f.dx2(weights=coeffs0) + f.dy2(weights=coeffs0)
        assert list(expr3.args[0].weights) == coeffs0

    def test_spacing(self):
        grid = Grid(shape=(11, 11))
        x, _ = grid.dimensions
        s = x.spacing

        f = Function(name='f', grid=grid, space_order=2)

        coeffs0 = [100, 100, 100]
        coeffs1 = [100/s, 100/s, 100/s]

        df = f.dx(weights=coeffs0)
        df_s = f.dx(weights=coeffs1)

        assert sp.simplify(df_s.evaluate - df.evaluate) == 0

    def test_backward_compat_mixed(self):

        grid = Grid(shape=(11,))
        x, = grid.dimensions

        f = Function(name='f', grid=grid, space_order=8)
        g = Function(name='g', grid=grid, space_order=2)

        coeffs0 = np.arange(0, 9)

        coeffs = Coefficient(1, f, x, coeffs0)

        eq = Eq(f, f.dx * g.dxc, coefficients=Substitutions(coeffs))

        derivs = retrieve_derivatives(eq.rhs)

        assert len(derivs) == 2
        df = [d for d in derivs if d.expr == f][0]
        dg = [d for d in derivs if d.expr == g][0]

        assert np.all(df.weights == coeffs0)
        assert dg.weights is None

        eqe = eq.evaluate
        assert '7.0*f(x + 3*h_x)' in str(eqe.rhs)
        assert '0.5*g(x + h_x)' in str(eqe.rhs)
        assert 'g(x + 2*h_x)' not in str(eqe.rhs)

    def test_backward_compat_array_of_func(self):
        grid = Grid(shape=(11, 11, 11))
        x, _, _ = grid.dimensions
        hx = x.spacing

        f = Function(name='f', grid=grid, space_order=16, coefficients='symbolic')

        # Define stencil coefficients.
        weights = Function(name="w", space_order=0, shape=(9,), dimensions=(x,))
        wdx = [weights[0]]
        for iq in range(1, weights.shape[0]):
            wdx.append(weights[iq])
            wdx.insert(0, weights[iq])

        # Plain numbers for comparison
        wdxn = np.random.rand(17)

        # Number with spacing
        wdxns = wdxn / hx

        dexev = f.dx(weights=wdx).evaluate
        dexevn = f.dx(weights=wdxn).evaluate
        dexevns = f.dx(weights=wdxns).evaluate

        assert all(a.as_coefficient(1/hx) for a in dexevn.args)
        assert all(a.as_coefficient(1/hx) for a in dexevns.args)
        assert all(not a.as_coefficient(1/hx) for a in dexev.args)

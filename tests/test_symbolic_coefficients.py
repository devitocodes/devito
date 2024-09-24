import numpy as np
import sympy as sp
import pytest

from devito import (Grid, Function, TimeFunction, Eq,
                    Dimension, solve, Operator)
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
        assert sp.simplify(eval(expected).evalf(_PRECISION) - eq.evaluate.lhs) == 0

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

        expected = 0.1*u - 0.6*u._subs(x, x - h_x) + 0.6*u.subs(x, x + h_x) + 2

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

        expected = 'Eq(f(x + h_x/2), 1.0*f(x - h_x/2) - 2.0*f(x + h_x/2)'\
            ' + 1.0*f(x + 3*h_x/2))'
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

        mul = lambda e: sp.Mul(e, 200, evaluate=False)
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

        # `str` simply because some objects are of type EvalDerivative
        assert sp.simplify(eq.evaluate.rhs - (term0 + term1 + term2)) == 0

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

        # `str` simply because some objects are of type EvalDerivative
        assert sp.simplify(eq.evaluate.rhs - (term0 + term1 + term2)) == 0

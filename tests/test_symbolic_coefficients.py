import numpy as np
import sympy as sp
import pytest

from devito import (Grid, Function, TimeFunction, Eq, Coefficient, Substitutions,
                    Dimension, solve, Operator)
from devito.finite_differences import Differentiable
from devito.tools import as_tuple

_PRECISION = 9


class TestSC(object):
    """
    Class for testing symbolic coefficients functionality
    """

    @pytest.mark.parametrize('order', [1, 2, 6])
    @pytest.mark.parametrize('stagger', [True, False])
    def test_default_rules(self, order, stagger):
        """
        Test that the default replacement rules return the same
        as standard FD.
        """
        grid = Grid(shape=(20, 20))
        if stagger:
            staggered = grid.dimensions[0]
        else:
            staggered = None
        u0 = TimeFunction(name='u', grid=grid, time_order=order, space_order=order,
                          staggered=staggered)
        u1 = TimeFunction(name='u', grid=grid, time_order=order, space_order=order,
                          staggered=staggered, coefficients='symbolic')
        eq0 = Eq(-u0.dx+u0.dt)
        eq1 = Eq(u1.dt-u1.dx)
        assert(eq0.evalf(_PRECISION).__repr__() == eq1.evalf(_PRECISION).__repr__())

    @pytest.mark.parametrize('expr, sorder, dorder, dim, weights, expected', [
        ('u.dx', 2, 1, 0, (-0.6, 0.1, 0.6),
         '0.1*u(x, y) - 0.6*u(x - h_x, y) + 0.6*u(x + h_x, y)'),
        ('u.dy2', 3, 2, 1, (0.121, -0.223, 1.648, -2.904),
         '1.648*u(x, y) + 0.121*u(x, y - 2*h_y) - 0.223*u(x, y - h_y) \
- 2.904*u(x, y + h_y)')])
    def test_coefficients(self, expr, sorder, dorder, dim, weights, expected):
        """Test that custom coefficients return the expected result"""
        grid = Grid(shape=(10, 10))
        u = Function(name='u', grid=grid, space_order=sorder, coefficients='symbolic')
        x = grid.dimensions

        order = dorder
        dim = x[dim]
        weights = np.array(weights)

        coeffs = Coefficient(order, u, dim, weights)

        eq = Eq(eval(expr), coefficients=Substitutions(coeffs))
        assert isinstance(eq.lhs, Differentiable)
        assert expected == str(eq.evaluate.lhs)

    def test_function_coefficients(self):
        """Test that custom function coefficients return the expected result"""
        so = 2
        grid = Grid(shape=(4, 4))
        f0 = TimeFunction(name='f0', grid=grid, space_order=so, coefficients='symbolic')
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

        f_x_coeffs = Coefficient(1, f0, x, w)

        subs = Substitutions(f_x_coeffs)

        eq0 = Eq(f0.dt + f0.dx, 1, coefficients=subs)
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
        u = Function(name='u', grid=grid, space_order=2, coefficients='symbolic')
        x = grid.dimensions[0]

        dorder = 1
        weights = np.array([-0.6, 0.1, 0.6])

        coeffs = Coefficient(dorder, u, x, weights)

        c = sp.Symbol('c')

        eq = Eq(u.dx+c, coefficients=Substitutions(coeffs))
        eq = eq.xreplace({c: 2})

        expected = '0.1*u(x, y) - 0.6*u(x - h_x, y) + 0.6*u(x + h_x, y) + 2'

        assert expected == str(eq.evaluate.lhs)

    @pytest.mark.parametrize('order', [2, 4, 6])
    def test_staggered_array(self, order):
        """Test custom coefficients provided as an array on a staggered grid"""
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=order,
                     coefficients='symbolic')
        g = Function(name='g', grid=grid, space_order=order,
                     coefficients='symbolic', staggered=x)
        f.data[::2] = 1
        g.data[::2] = 1

        weights = np.ones(order+1)/grid.spacing[0]**2
        coeffs_f = Coefficient(2, f, x, weights)
        coeffs_g = Coefficient(2, g, x, weights)

        eq_f = Eq(f, f.dx2, coefficients=Substitutions(coeffs_f))
        eq_g = Eq(g, g.dx2, coefficients=Substitutions(coeffs_g))

        Operator([eq_f, eq_g])()

        assert np.allclose(f.data, g.data, atol=1e-7)

    @pytest.mark.parametrize('order', [2, 4, 6])
    def test_staggered_function(self, order):
        """Test custom function coefficients on a staggered grid"""
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=order,
                     coefficients='symbolic')
        g = Function(name='g', grid=grid, space_order=order,
                     coefficients='symbolic', staggered=x)
        f.data[::2] = 1
        g.data[::2] = 1

        s = Dimension(name='s')
        ncoeffs = order+1

        wshape = grid.shape + (ncoeffs,)
        wdims = grid.dimensions + (s,)

        w = Function(name='w', dimensions=wdims, shape=wshape)
        w.data[:] = 1.0/grid.spacing[0]**2

        coeffs_f = Coefficient(2, f, x, w)
        coeffs_g = Coefficient(2, g, x, w)

        eq_f = Eq(f, f.dx2, coefficients=Substitutions(coeffs_f))
        eq_g = Eq(g, g.dx2, coefficients=Substitutions(coeffs_g))

        Operator([eq_f, eq_g])()

        assert np.allclose(f.data, g.data, atol=1e-7)

    def test_staggered_equation(self):
        """
        Check that expressions with substitutions are consistent with
        those without
        """
        grid = Grid(shape=(11,), extent=(10.,))
        x = grid.dimensions[0]

        f = Function(name='f', grid=grid, space_order=2,
                     coefficients='symbolic', staggered=x)

        weights = np.array([1, -2, 1])/grid.spacing[0]**2
        coeffs_f = Coefficient(2, f, x, weights)

        eq_f = Eq(f, 1.0*f.dx2, coefficients=Substitutions(coeffs_f))

        expected = 'Eq(f(x + h_x/2), 1.0*f(x - h_x/2) - 2.0*f(x + h_x/2)' \
            + ' + 1.0*f(x + 3*h_x/2))'
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
        g = TimeFunction(name='g', grid=grid, space_order=2, staggered=staggered,
                         coefficients='symbolic')

        f.data[:, ::2] = 1
        g.data[:, ::2] = 1

        weights = np.array([-1, 2, -1])/grid.spacing[0]**2
        coeffs = Coefficient(2, g, x, weights)

        eq_f = Eq(f.forward, f.dx2)
        eq_g = Eq(g.forward, g.dx2, coefficients=Substitutions(coeffs))

        Operator([eq_f, eq_g])(t_m=0, t_M=1)

        assert np.allclose(f.data[-1], -g.data[-1], atol=1e-7)

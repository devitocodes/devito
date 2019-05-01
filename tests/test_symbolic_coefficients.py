import numpy as np
import pytest

from conftest import skipif
from devito import Grid, Function, TimeFunction, Eq, Coefficient, Substitutions
from devito.finite_differences import Differentiable

_PRECISION = 9

pytestmark = skipif(['yask', 'ops'])


class TestSC(object):
    """
    Class for testing symbolic coefficients functionality
    """

    @pytest.mark.parametrize('order', [1, 2, 6])
    def test_default_rules(self, order):
        """
        Test that the default replacement rules return the same
        as standard FD.
        """
        grid = Grid(shape=(20, 20))
        u0 = TimeFunction(name='u', grid=grid, time_order=order, space_order=order)
        u1 = TimeFunction(name='u', grid=grid, time_order=order, space_order=order,
                          coefficients='symbolic')
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

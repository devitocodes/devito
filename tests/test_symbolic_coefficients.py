import numpy as np
import pytest

from conftest import skipif
from devito import Grid, Function, TimeFunction, Eq, Coefficient, CoefficientRules
from devito.finite_differences import Differentiable

_PRECISION = 9

pytestmark = skipif(['yask', 'ops'])


class TestSC(object):
    """
    Class for testing symbolic FD coefficients
    functionality
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

    @pytest.mark.parametrize('expected', [
        ('0.1*u(x) - 0.6*u(x - h_x) + 0.6*u(x + h_x)'),
    ])
    def test_coefficients(self, expected):
        """Test that custom coefficients return the expected result"""
        grid = Grid(shape=(10,))
        u = Function(name='u', grid=grid, space_order=2, coefficients='symbolic')
        x = grid.dimensions

        coeffs = Coefficient(1, u, x[0], np.array([-0.6, 0.1, 0.6]))

        eq = Eq(u.dx, coefficients=CoefficientRules(coeffs))

        assert isinstance(eq.lhs, Differentiable)
        assert expected == str(eq.lhs)

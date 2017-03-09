import numpy as np
import pytest
from sympy import Eq, diff, symbols

from devito import StencilKernel, clear_cache
from devito.interfaces import DenseData


@pytest.mark.parametrize('space_order', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# Only test x and t as y and z are the same as x
@pytest.mark.parametrize('derivative', ['dx', 'dxl', 'dxr', 'dx2'])
def test_fd_space(derivative, space_order):
    """
    This test compare the discrete finite-difference scheme against polynomials
    For a given order p, the fiunite difference scheme should
    be exact for polynomials of order p
    :param derivative: name of the derivative to be tested
    :param space_order: space order of the finite difference stencil
    """
    clear_cache()
    # dummy axis dimension
    nx = 100
    xx = np.linspace(-1, 1, nx)
    dx = xx[1] - xx[0]
    # Symbolic data
    u = DenseData(name="u", shape=(nx,), space_order=space_order, dtype=np.float32)
    du = DenseData(name="du", shape=(nx,), space_order=space_order, dtype=np.float32)
    # Define polynomial with exact fd
    h, y = symbols('h y')
    coeffs = np.ones((space_order,), dtype=np.float32)
    polynome = sum([coeffs[i]*y**i for i in range(0, space_order)])
    polyvalues = np.array([polynome.subs(y, xi) for xi in xx], np.float32)
    # Fill original data with the polynomial values
    u.data[:] = polyvalues
    # True derivative of the polynome
    Dpolynome = diff(diff(polynome)) if derivative == 'dx2' else diff(polynome)
    Dpolyvalues = np.array([Dpolynome.subs(y, xi) for xi in xx], np.float32)
    # FD derivative, symbolic
    u_deriv = getattr(u, derivative)
    # Compute numerical FD
    stencil = Eq(du, u_deriv)
    op = StencilKernel(stencil, subs={h: dx})
    op.apply()

    # Check exactness of the numerical derivative except inside space_brd
    space_border = space_order
    error = abs(du.data[space_border:-space_border] -
                Dpolyvalues[space_border:-space_border])
    assert np.isclose(np.mean(error), 0., atol=1e-3)


if __name__ == "__main__":
    test_fd_space(derivative='dx2', space_order=12)

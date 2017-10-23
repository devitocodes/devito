import numpy as np
import pytest
from conftest import skipif_yask
from sympy import diff

from devito import Grid, Eq, Operator, clear_cache, Function


@skipif_yask
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
    grid = Grid(shape=(nx,), dtype=np.float32)
    x = grid.dimensions[0]
    u = Function(name="u", grid=grid, space_order=space_order)
    du = Function(name="du", grid=grid, space_order=space_order)
    # Define polynomial with exact fd
    coeffs = np.ones((space_order,), dtype=np.float32)
    polynome = sum([coeffs[i]*x**i for i in range(0, space_order)])
    polyvalues = np.array([polynome.subs(x, xi) for xi in xx], np.float32)
    # Fill original data with the polynomial values
    u.data[:] = polyvalues
    # True derivative of the polynome
    Dpolynome = diff(diff(polynome)) if derivative == 'dx2' else diff(polynome)
    Dpolyvalues = np.array([Dpolynome.subs(x, xi) for xi in xx], np.float32)
    # FD derivative, symbolic
    u_deriv = getattr(u, derivative)
    # Compute numerical FD
    stencil = Eq(du, u_deriv)
    op = Operator(stencil, subs={x.spacing: dx})
    op.apply()

    # Check exactness of the numerical derivative except inside space_brd
    space_border = space_order
    error = abs(du.data[space_border:-space_border] -
                Dpolyvalues[space_border:-space_border])
    assert np.isclose(np.mean(error), 0., atol=1e-3)


if __name__ == "__main__":
    test_fd_space(derivative='dx2', space_order=12)

import numpy as np
import pytest
from sympy import simplify, diff, cos, sin

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, Operator, clear_cache, NODE,
                    ConditionalDimension, left, right, centered)
from devito.finite_differences import Derivative, Differentiable

_PRECISION = 9


def x(grid):
    return grid.dimensions[0]


def y(grid):
    return grid.dimensions[1]


def z(grid):
    return grid.dimensions[2]


def t(grid):
    return grid.stepping_dim


@skipif(['yask', 'ops'])
class TestFD(object):
    """
    Class for finite difference testing.
    Tests the accuracy w.r.t polynomials.
    Test that the shortcut produce the same answer as the FD functions.
    """

    def setup_method(self):
        self.shape = (20, 20, 20)
        self.grid = Grid(self.shape)

    def test_diff(self):
        """Test that expr.diff returns an object of type devito.Derivative."""
        u = Function(name='u', grid=self.grid)
        du = u.diff(x(self.grid))
        assert isinstance(du, Derivative)

    @pytest.mark.parametrize('SymbolType, dim', [
        (Function, x), (Function, y),
        (TimeFunction, x), (TimeFunction, y), (TimeFunction, t),
    ])
    def test_stencil_derivative(self, SymbolType, dim):
        """Test symbolic behaviour when expanding stencil derivatives"""
        i = dim(self.grid)
        u = SymbolType(name='u', grid=self.grid)
        u.data[:] = 66.6
        di = u.diff(i)
        dii = u.diff(i, i)
        # Check for sympy Derivative objects
        assert(isinstance(di, Derivative) and isinstance(dii, Derivative))
        s_di = di.as_finite_difference([i - i.spacing, i])
        s_dii = dii.as_finite_difference([i - i.spacing, i, i + i.spacing])
        # Check stencil length of first and second derivatives
        assert(len(s_di.args) == 2 and len(s_dii.args) == 3)
        u_di = s_di.args[0].args[1]
        u_dii = s_di.args[0].args[1]
        # Ensure that devito meta-data survived symbolic transformation
        assert(u_di.grid.shape == self.shape and u_dii.grid.shape == self.shape)
        assert(u_di.shape == u.shape and u_dii.shape == u.shape)
        assert(np.allclose(u_di.data, 66.6))
        assert(np.allclose(u_dii.data, 66.6))

    @pytest.mark.parametrize('SymbolType, derivative, dim, expected', [
        (Function, ['dx2'], 3, 'Derivative(u(x, y, z), (x, 2))'),
        (Function, ['dx2dy'], 3, 'Derivative(u(x, y, z), (x, 2), y)'),
        (Function, ['dx2dydz'], 3, 'Derivative(u(x, y, z), (x, 2), y, z)'),
        (Function, ['dx2', 'dy'], 3, 'Derivative(Derivative(u(x, y, z), (x, 2)), y)'),
        (Function, ['dx2dy', 'dz2'], 3,
         'Derivative(Derivative(u(x, y, z), (x, 2), y), (z, 2))'),
        (TimeFunction, ['dx2'], 3, 'Derivative(u(t, x, y, z), (x, 2))'),
        (TimeFunction, ['dx2dy'], 3, 'Derivative(u(t, x, y, z), (x, 2), y)'),
        (TimeFunction, ['dx2', 'dy'], 3,
         'Derivative(Derivative(u(t, x, y, z), (x, 2)), y)'),
        (TimeFunction, ['dx', 'dy', 'dx2', 'dz', 'dydz'], 3,
         'Derivative(Derivative(Derivative(Derivative(Derivative(u(t, x, y, z), x), y),' +
         ' (x, 2)), z), y, z)')
    ])
    def test_unevaluation(self, SymbolType, derivative, dim, expected):
        u = SymbolType(name='u', grid=self.grid, time_order=2, space_order=2)
        expr = getattr(u, derivative[0])
        for d in derivative[1:]:
            expr = getattr(expr, d)
        assert(expr.__str__() == expected)
        # Make sure the FD evaluation executes
        expr.evaluate

    @pytest.mark.parametrize('expr,expected', [
        ('u.dx + u.dy', 'Derivative(u, x) + Derivative(u, y)'),
        ('u.dxdy', 'Derivative(u, x, y)'),
        ('u.laplace',
         'Derivative(u, (x, 2)) + Derivative(u, (y, 2)) + Derivative(u, (z, 2))'),
        ('(u.dx + u.dy).dx', 'Derivative(Derivative(u, x) + Derivative(u, y), x)'),
        ('((u.dx + u.dy).dx + u.dxdy).dx',
         'Derivative(Derivative(Derivative(u, x) + Derivative(u, y), x) +' +
         ' Derivative(u, x, y), x)'),
        ('(u**4).dx', 'Derivative(u**4, x)'),
        ('(u/4).dx', 'Derivative(u/4, x)'),
        ('((u.dx + v.dy).dx * v.dx).dy.dz',
         'Derivative(Derivative(Derivative(Derivative(u, x) + Derivative(v, y), x) *' +
         ' Derivative(v, x), y), z)')
    ])
    def test_arithmetic(self, expr, expected):
        x, y, z = self.grid.dimensions
        u = Function(name='u', grid=self.grid, time_order=2, space_order=2)  # noqa
        v = Function(name='v', grid=self.grid, time_order=2, space_order=2)  # noqa
        expr = eval(expr)
        expected = eval(expected)
        assert expr == expected

    @pytest.mark.parametrize('expr, rules', [
        ('u.dx + u.dy', '{u.indices[0]: 1, u.indices[1]: 0}'),
        ('u.dxdy - u.dxdz', '{u.indices[0]: u.indices[0] + u.indices[0].spacing,' +
                            'u.indices[1]: 0, u.indices[2]: u.indices[1]}'),
        ('u.dx2dy + u.dz ', '{u.indices[0]: u.indices[0] + u.indices[0].spacing,' +
                            'u.indices[2]: u.indices[2] - 10}'),
    ])
    def test_derivative_eval_at(self, expr, rules):
        u = Function(name='u', grid=self.grid, time_order=2, space_order=2)  # noqa
        expr = eval(expr)
        rules = eval(rules)
        assert expr.evaluate.xreplace(rules) == expr.xreplace(rules).evaluate

    @pytest.mark.parametrize('expr, rules', [
        ('u.dx', '{u.indices[0]: 1}'),
        ('u.dy', '{u.indices[1]: u.indices[2] - 7}'),
        ('u.dz', '{u.indices[2]: u.indices[0] + u.indices[1].spacing}'),
    ])
    def test_derivative_eval_at_expr(self, expr, rules):
        u = Function(name='u', grid=self.grid, time_order=2, space_order=2)  # noqa
        expr = eval(expr)
        rules = eval(rules)
        assert expr.evaluate.xreplace(rules) == expr.xreplace(rules).evaluate
        assert expr.expr == expr.xreplace(rules).expr

    @pytest.mark.parametrize('expr, composite_rules', [
        ('u.dx', '[{u.indices[0]: 1}, {1: 4}]'),
    ])
    def test_derivative_eval_at_composite(self, expr, composite_rules):
        u = Function(name='u', grid=self.grid, time_order=2, space_order=2)  # noqa
        expr = eval(expr)
        evaluated_expr = expr.evaluate
        composite_rules = eval(composite_rules)
        for mapper in composite_rules:
            evaluated_expr = evaluated_expr.xreplace(mapper)
            expr = expr.xreplace(mapper)
        assert evaluated_expr == expr.evaluate

    @pytest.mark.parametrize('SymbolType, derivative, dim', [
        (Function, 'dx2', 3), (Function, 'dy2', 3),
        (TimeFunction, 'dx2', 3), (TimeFunction, 'dy2', 3), (TimeFunction, 'dt', 2)
    ])
    def test_preformed_derivatives(self, SymbolType, derivative, dim):
        """Test the stencil expressions provided by devito objects"""
        u = SymbolType(name='u', grid=self.grid, time_order=2, space_order=2)
        expr = getattr(u, derivative)
        assert(len(expr.evaluate.args) == dim)

    @pytest.mark.parametrize('derivative, dim', [
        ('dx', x), ('dy', y), ('dz', z)
    ])
    @pytest.mark.parametrize('order', [1, 2, 4, 6, 8, 10, 12, 14, 16])
    def test_derivatives_space(self, derivative, dim, order):
        """Test first derivative expressions against native sympy"""
        dim = dim(self.grid)
        u = TimeFunction(name='u', grid=self.grid, time_order=2, space_order=order)
        expr = getattr(u, derivative).evaluate
        # Establish native sympy derivative expression
        width = int(order / 2)
        if order == 1:
            indices = [dim, dim + dim.spacing]
        else:
            indices = [(dim + i * dim.spacing) for i in range(-width, width + 1)]

        s_expr = u.diff(dim).as_finite_difference(indices).evalf(_PRECISION)
        assert(simplify(expr - s_expr) == 0)  # Symbolic equality
        assert(expr == s_expr)  # Exact equailty

    @pytest.mark.parametrize('derivative, dim', [
        ('dx2', x), ('dy2', y), ('dz2', z)
    ])
    @pytest.mark.parametrize('order', [2, 4, 6, 8, 10, 12, 14, 16])
    def test_second_derivatives_space(self, derivative, dim, order):
        """
        Test second derivative expressions against native sympy.
        """
        dim = dim(self.grid)
        u = TimeFunction(name='u', grid=self.grid, time_order=2, space_order=order)
        expr = getattr(u, derivative).evaluate
        # Establish native sympy derivative expression
        width = int(order / 2)
        indices = [(dim + i * dim.spacing) for i in range(-width, width + 1)]
        s_expr = u.diff(dim, dim).as_finite_difference(indices).evalf(_PRECISION)
        assert(simplify(expr - s_expr) == 0)  # Symbolic equality
        assert(expr == s_expr)  # Exact equailty

    @pytest.mark.parametrize('space_order', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    # Only test x and t as y and z are the same as x
    @pytest.mark.parametrize('derivative', ['dx', 'dxl', 'dxr', 'dx2'])
    def test_fd_space(self, derivative, space_order):
        """
        This test compares the discrete finite-difference scheme against polynomials
        For a given order p, the finite difference scheme should
        be exact for polynomials of order p.
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

    @pytest.mark.parametrize('space_order', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    @pytest.mark.parametrize('stagger', [centered, left, right])
    # Only test x and t as y and z are the same as x
    def test_fd_space_staggered(self, space_order, stagger):
        """
        This test compares the discrete finite-difference scheme against polynomials
        For a given order p, the finite difference scheme should
        be exact for polynomials of order p
        """
        clear_cache()
        # dummy axis dimension
        nx = 101
        xx = np.linspace(-1, 1, nx)
        dx = xx[1] - xx[0]
        # Symbolic data
        grid = Grid(shape=(nx,), dtype=np.float32)
        x = grid.dimensions[0]

        # Location of the staggered function
        if stagger == left:
            off = -.5
            side = -x
            xx2 = xx + off * dx
        elif stagger == right:
            off = .5
            side = x
            xx2 = xx + off * dx
        else:
            side = NODE
            xx2 = xx

        u = Function(name="u", grid=grid, space_order=space_order, staggered=side)
        du = Function(name="du", grid=grid, space_order=space_order, staggered=side)
        # Define polynomial with exact fd
        coeffs = np.ones((space_order-1,), dtype=np.float32)
        polynome = sum([coeffs[i]*x**i for i in range(0, space_order-1)])
        polyvalues = np.array([polynome.subs(x, xi) for xi in xx2], np.float32)
        # Fill original data with the polynomial values
        u.data[:] = polyvalues
        # True derivative of the polynome
        Dpolynome = diff(polynome)
        Dpolyvalues = np.array([Dpolynome.subs(x, xi) for xi in xx2], np.float32)
        # Compute numerical FD
        stencil = Eq(du, u.dx)
        op = Operator(stencil, subs={x.spacing: dx})
        op.apply()

        # Check exactness of the numerical derivative except inside space_brd
        space_border = space_order
        error = abs(du.data[space_border:-space_border] -
                    Dpolyvalues[space_border:-space_border])

        assert np.isclose(np.mean(error), 0., atol=1e-3)

    def test_subsampled_fd(self):
        """
        Test that the symbolic interface is working for space subsampled
        functions.
        """
        nt = 19
        grid = Grid(shape=(12, 12), extent=(11, 11))

        u = TimeFunction(name='u', grid=grid, save=nt, space_order=2)
        assert(grid.time_dim in u.indices)

        # Creates subsampled spatial dimensions and according grid
        dims = tuple([ConditionalDimension(d.name+'sub', parent=d, factor=2)
                      for d in u.grid.dimensions])
        grid2 = Grid((6, 6), dimensions=dims)
        u2 = TimeFunction(name='u2', grid=grid2, save=nt, space_order=1)
        for i in range(nt):
            for j in range(u2.data_with_halo.shape[2]):
                u2.data_with_halo[i, :, j] = np.arange(u2.data_with_halo.shape[2])

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2.dx)]
        op = Operator(eqns, dse="advanced")
        op.apply(time_M=nt-2)
        # Verify that u2[1, x,y]= du2/dx[0, x, y]

        assert np.allclose(u.data[-1], nt-1)
        assert np.allclose(u2.data[1], 0.5)

    @pytest.mark.parametrize('expr,expected', [
        ('f.dx', '-f(x)/h_x + f(x + h_x)/h_x'),
        ('f.dx + g.dx', '-f(x)/h_x + f(x + h_x)/h_x - g(x)/h_x + g(x + h_x)/h_x'),
        ('-f', '-f(x)'),
        ('-(f + g)', '-f(x) - g(x)')
    ])
    def test_shortcuts(self, expr, expected):
        grid = Grid(shape=(10,))
        f = Function(name='f', grid=grid)  # noqa
        g = Function(name='g', grid=grid)  # noqa

        expr = eval(expr)

        assert isinstance(expr, Differentiable)
        assert expected == str(expr.evaluate)

    @pytest.mark.parametrize('so', [2, 5, 8])
    def test_all_shortcuts(self, so):
        """
        Test that verify that all fd shortcuts are functional.
        """
        grid = Grid(shape=(10, 10, 10))
        f = Function(name='f', grid=grid, space_order=so)
        g = TimeFunction(name='g', grid=grid, space_order=so)

        for fd in f._fd:
            assert getattr(f, fd)

        for fd in g._fd:
            assert getattr(g, fd)

    @pytest.mark.parametrize('so', [2, 4, 8, 12])
    @pytest.mark.parametrize('ndim', [1, 2])
    @pytest.mark.parametrize('derivative, adjoint_name, adjoint_coeff', [
        ('dx', 'dx', -1),
        ('dx2', 'dx2', 1),
        ('dxl', 'dxr', -1),
        ('dxr', 'dxl', -1)])
    def test_fd_adjoint(self, so, ndim, derivative, adjoint_name, adjoint_coeff):
        clear_cache()
        grid = Grid(shape=tuple([51]*ndim), extent=tuple([25]*ndim))
        x = grid.dimensions[0]
        f = Function(name='f', grid=grid, space_order=so)
        f_deriv = Function(name='f_deriv', grid=grid, space_order=so)
        g = Function(name='g', grid=grid, space_order=so)
        g_deriv = Function(name='g_deriv', grid=grid, space_order=so)

        # Fill f and g with smooth cos/sin
        Operator([Eq(g, cos(2*np.pi*x/5)), Eq(f, sin(2*np.pi*x/8))]).apply()
        # Check symbolic expression are expected ones for the adjoint .T
        deriv = getattr(f, derivative)
        expected = adjoint_coeff * getattr(f, adjoint_name).evaluate
        assert deriv.T.evaluate == expected

        # Compute numerical derivatives and verify dot test
        #  i.e <f.dx, g> = <f, g.dx.T>

        eq_f = Eq(f_deriv, deriv)
        eq_g = Eq(g_deriv, getattr(g, derivative).T)

        op = Operator([eq_f, eq_g])
        op()

        a = np.dot(f_deriv.data.reshape(-1), g.data.reshape(-1))
        b = np.dot(g_deriv.data.reshape(-1), f.data.reshape(-1))
        assert np.isclose(1 - a/b, 0, atol=1e-5)

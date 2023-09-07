import numpy as np
import pytest
from sympy import simplify, diff, Float

from devito import (Grid, Function, TimeFunction, Eq, Operator, NODE, cos, sin,
                    ConditionalDimension, left, right, centered, div, grad)
from devito.finite_differences import Derivative, Differentiable
from devito.finite_differences.differentiable import (Add, EvalDerivative, IndexSum,
                                                      IndexDerivative, Weights)
from devito.symbolics import indexify, retrieve_indexed
from devito.types.dimension import StencilDimension

_PRECISION = 9


def x(grid):
    return grid.dimensions[0]


def y(grid):
    return grid.dimensions[1]


def z(grid):
    return grid.dimensions[2]


def t(grid):
    return grid.stepping_dim


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

    @pytest.mark.parametrize('so', [2, 3, 4, 5])
    def test_fd_indices(self, so):
        """
        Test that shifted derivative have Integer offset after indexification.
        """
        grid = Grid((10,))
        x = grid.dimensions[0]
        x0 = x + .5 * x.spacing
        u = Function(name="u", grid=grid, space_order=so)
        dx = indexify(u.dx(x0=x0).evaluate)
        for f in retrieve_indexed(dx):
            assert len(f.indices[0].atoms(Float)) == 0

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
        if order <= 2:
            indices = [dim, dim + dim.spacing]
        else:
            indices = [(dim + i * dim.spacing) for i in range(-width, width + 1)]

        s_expr = u.diff(dim).as_finite_difference(indices).evalf(_PRECISION)
        assert(simplify(expr - s_expr) == 0)  # Symbolic equality
        assert type(expr) is EvalDerivative
        expr1 = s_expr.func(*expr.args)
        assert(expr1 == s_expr)  # Exact equality

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
        assert type(expr) is EvalDerivative
        expr1 = s_expr.func(*expr.args)
        assert(expr1 == s_expr)  # Exact equality

    @pytest.mark.parametrize('space_order', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    # Only test x and t as y and z are the same as x
    @pytest.mark.parametrize('derivative', ['dx', 'dxl', 'dxr', 'dx2'])
    def test_fd_space(self, derivative, space_order):
        """
        This test compares the discrete finite-difference scheme against polynomials
        For a given order p, the finite difference scheme should
        be exact for polynomials of order p.
        """
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

    @pytest.mark.parametrize('so', [2, 4, 6, 8])
    def test_fd_new_order(self, so):
        grid = Grid((10,))
        u = Function(name="u", grid=grid, space_order=so)
        u1 = Function(name="u", grid=grid, space_order=so//2)
        u2 = Function(name="u", grid=grid, space_order=2*so)
        assert str(u.dx(fd_order=so//2).evaluate) == str(u1.dx.evaluate)
        assert str(u.dx(fd_order=2*so).evaluate) == str(u2.dx.evaluate)

    def test_xderiv_order(self):
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        x, y = grid.dimensions
        f = Function(name='f', grid=grid, space_order=4)

        # Check that supplying a dictionary to fd_order for x-derivs functions correctly
        expr = f.dxdy(fd_order={x: 2, y: 2}).evaluate \
            - f.dx(fd_order=2).dy(fd_order=2).evaluate
        assert simplify(expr) == 0

    def test_xderiv_x0(self):
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        x, y = grid.dimensions
        h_x = x.spacing
        h_y = y.spacing
        f = Function(name='f', grid=grid, space_order=4)

        # Check that supplying a dictionary to x0 for x-derivs functions correctly
        expr = f.dxdy(x0={x: x+h_x/2, y: y+h_y/2}).evaluate \
            - f.dx(x0=x+h_x/2).dy(x0=y+h_y/2).evaluate
        assert simplify(expr) == 0

    def test_fd_new_side(self):
        grid = Grid((10,))
        u = Function(name="u", grid=grid, space_order=4)
        assert u.dx(side=left).evaluate == u.dxl.evaluate
        assert u.dx(side=right).evaluate == u.dxr.evaluate
        assert u.dxl(side=centered).evaluate == u.dx.evaluate

    @pytest.mark.parametrize('so, expected', [
        (2, 'u(x)/h_x - u(x - 1.0*h_x)/h_x'),
        (4, '1.125*u(x)/h_x + 0.0416666667*u(x - 2*h_x)/h_x - '
            '1.125*u(x - h_x)/h_x - 0.0416666667*u(x + h_x)/h_x'),
        (6, '1.171875*u(x)/h_x - 0.0046875*u(x - 3*h_x)/h_x + '
            '0.0651041667*u(x - 2*h_x)/h_x - 1.171875*u(x - h_x)/h_x - '
            '0.0651041667*u(x + h_x)/h_x + 0.0046875*u(x + 2*h_x)/h_x'),
        (8, '1.19628906*u(x)/h_x + 0.000697544643*u(x - 4*h_x)/h_x - '
            '0.0095703125*u(x - 3*h_x)/h_x + 0.0797526042*u(x - 2*h_x)/h_x - '
            '1.19628906*u(x - h_x)/h_x - 0.0797526042*u(x + h_x)/h_x + '
            '0.0095703125*u(x + 2*h_x)/h_x - 0.000697544643*u(x + 3*h_x)/h_x')])
    def test_fd_new_x0(self, so, expected):
        grid = Grid((10,))
        x = grid.dimensions[0]
        u = Function(name="u", grid=grid, space_order=so)
        assert u.dx(x0=x + x.spacing).evaluate == u.dx.evaluate.subs({x: x + x.spacing})
        assert u.dx(x0=x - x.spacing).evaluate == u.dx.evaluate.subs({x: x - x.spacing})
        # half shifted compare to explicit coeffs (Forneberg)
        assert str(u.dx(x0=x - .5 * x.spacing).evaluate) == expected

    def test_new_x0_eval_at(self):
        """
        Make sure that explicitly set x0 does not get overwritten by eval_at.
        """
        grid = Grid((10,))
        x = grid.dimensions[0]
        u = Function(name="u", grid=grid, space_order=2)
        v = Function(name="v", grid=grid, space_order=2)
        assert u.dx(x0=x - x.spacing/2)._eval_at(v).x0 == {x: x - x.spacing/2}

    def test_fd_new_lo(self):
        grid = Grid((10,))
        x = grid.dimensions[0]
        u = Function(name="u", grid=grid, space_order=2)

        dplus = "-u(x)/h_x + u(x + 1.0*h_x)/h_x"
        dminus = "u(x)/h_x - u(x - 1.0*h_x)/h_x"
        assert str(u.dx(x0=x + .5 * x.spacing).evaluate) == dplus
        assert str(u.dx(x0=x - .5 * x.spacing).evaluate) == dminus
        assert str(u.dx(x0=x + .5 * x.spacing, fd_order=1).evaluate) == dplus
        assert str(u.dx(x0=x - .5 * x.spacing, fd_order=1).evaluate) == dminus

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
        grid2 = Grid((6, 6), dimensions=dims, extent=(10, 10))
        u2 = TimeFunction(name='u2', grid=grid2, save=nt, space_order=1)
        for i in range(nt):
            for j in range(u2.data_with_halo.shape[2]):
                u2.data_with_halo[i, :, j] = np.arange(u2.data_with_halo.shape[2])

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2.dx)]
        op = Operator(eqns)

        # NOTE: To assert against u.data[-1] we need to provide explicit bounds
        # for x_M and y_M, otherwise they wouldn't be necessary as it wouldn't
        # matter whether we run up to x_M=10 or x_M=11 !
        op.apply(time_M=nt-2, x_M=11, y_M=11)

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
    @pytest.mark.parametrize('derivative, adjoint_name', [
        ('dx', 'dx'),
        ('dx2', 'dx2'),
        ('dxl', 'dxr'),
        ('dxr', 'dxl')])
    def test_fd_adjoint(self, so, ndim, derivative, adjoint_name):
        grid = Grid(shape=tuple([51]*ndim), extent=tuple([25]*ndim))
        x = grid.dimensions[0]
        f = Function(name='f', grid=grid, space_order=so)
        f_deriv = Function(name='f_deriv', grid=grid, space_order=so)
        g = Function(name='g', grid=grid, space_order=so)
        g_deriv = Function(name='g_deriv', grid=grid, space_order=so)

        # Fill f and g with smooth cos/sin
        Operator([Eq(g, x*cos(2*np.pi*x/5)), Eq(f, sin(2*np.pi*x/8))]).apply()
        # Check symbolic expression are expected ones for the adjoint .T
        deriv = getattr(f, derivative)
        coeff = 1 if derivative == 'dx2' else -1
        expected = coeff * getattr(f, derivative).evaluate.subs({x.spacing: -x.spacing})
        assert simplify(deriv.T.evaluate) == simplify(bypass_uneval(expected))

        # Compute numerical derivatives and verify dot test
        #  i.e <f.dx, g> = <f, g.dx.T>

        eq_f = Eq(f_deriv, deriv)
        eq_g = Eq(g_deriv, getattr(g, derivative).T)

        op = Operator([eq_f, eq_g])
        op()

        a = np.dot(f_deriv.data.reshape(-1), g.data.reshape(-1))
        b = np.dot(g_deriv.data.reshape(-1), f.data.reshape(-1))
        assert np.isclose(1 - a/b, 0, atol=1e-5)

    @pytest.mark.parametrize('shift, ndim', [(None, 2), (.5, 2), (.5, 3),
                                             ((.5, .5, .5), 3)])
    def test_shifted_div(self, shift, ndim):
        grid = Grid(tuple([11]*ndim))
        f = Function(name="f", grid=grid, space_order=4)
        df = div(f, shift=shift).evaluate
        ref = 0
        for i, d in enumerate(grid.dimensions):
            x0 = (None if shift is None else d + shift[i] * d.spacing if
                  type(shift) is tuple else d + shift * d.spacing)
            ref += getattr(f, 'd%s' % d.name)(x0=x0)
        assert df == ref.evaluate

    @pytest.mark.parametrize('shift, ndim', [(None, 2), (.5, 2), (.5, 3),
                                             ((.5, .5, .5), 3)])
    def test_shifted_grad(self, shift, ndim):
        grid = Grid(tuple([11]*ndim))
        f = Function(name="f", grid=grid, space_order=4)
        g = grad(f, shift=shift).evaluate
        for i, (d, gi) in enumerate(zip(grid.dimensions, g)):
            x0 = (None if shift is None else d + shift[i] * d.spacing if
                  type(shift) is tuple else d + shift * d.spacing)
            assert gi == getattr(f, 'd%s' % d.name)(x0=x0).evaluate

    def test_substitution(self):
        grid = Grid((11, 11))
        f = Function(name="f", grid=grid, space_order=4)
        expr = f.dx + f + 1

        assert simplify(expr.subs(f.dx, 1) - (f + 2)) == 0
        # f.dx.subs(f, -1) = 0 and f.subs(f, -1) = -1 so
        # expr.subs(f, -1) = 0
        assert simplify(expr.subs(f, -1)) == 0
        # f.dx -> 1, f -> -1
        assert simplify(expr.subs({f.dx: 1, f: -1}) - 1) == 0

        expr2 = expr.subs({'x0': 2})
        # Expression should now be the same but with a different x0
        assert simplify(expr2 - (f.dx(x0=2) + f + 1)) == 0
        # Different x0, should no replace
        assert simplify(expr2.subs(f.dx, 1) - expr2) == 0

        # x0 and f.dx
        expr3 = expr.subs({f.dx: f.dx2, 'x0': 2})
        assert simplify(expr3 - (f.dx2(x0=2) + f + 1)) == 0

        # Test substitution with reconstructed objects
        x, y = grid.dimensions
        f1 = f.func(x + 1, y)
        f2 = f.func(x + 1, y)
        assert f1 is not f2
        assert f1.subs(f2, -1) == -1


class TestTwoStageEvaluation(object):

    def test_exceptions(self):
        grid = Grid((10,))

        x, = grid.dimensions

        with pytest.raises(TypeError):
            # Missing 1 required positional argument: '_max'
            StencilDimension('i', 3)
        with pytest.raises(ValueError):
            # Spacing must be an integer
            StencilDimension('i', 3, 5, spacing=0.6)
        i = StencilDimension('i', 0, 1)
        assert i.symbolic_size == 2

        u = Function(name="u", grid=grid, space_order=2)

        with pytest.raises(ValueError):
            # Expected Dimension with numeric size, got `1` instead
            IndexSum(u, 1)
        with pytest.raises(ValueError):
            # Expected Dimension with numeric size, got `x` instead
            IndexSum(u, x)
        with pytest.raises(ValueError):
            # Dimension `i` must appear in `expr`
            IndexSum(u, i)

    def test_stencil_dim_comparison(self):
        i1 = StencilDimension('i', 0, 1)
        i2 = StencilDimension('i', 0, 1)
        i3 = StencilDimension('i', 0, 2)
        assert i1 is i2  # Due to caching
        assert i1 == i2  # Obv
        assert i1 != i3

    def test_index_sum_basic(self):
        grid = Grid((10,))

        x, = grid.dimensions
        i = StencilDimension('i', 0, 1)

        u = Function(name="u", grid=grid, space_order=2)

        # Build `u(x + h_x)`
        term0 = u.subs(x, x + x.spacing)

        # `u(x + h_x)` -> `u(x + i*h_x)`
        term = term0.subs(x + x.spacing, x + i*x.spacing)

        # Sum `term` over `i`
        idxsum = IndexSum(term, i)

        # == u(x) + u(x + h_x)
        assert idxsum.evaluate == u + term0

    def test_index_sum_2d(self):
        grid = Grid((10, 10))

        x, y = grid.dimensions
        i = StencilDimension('i', 0, 1)
        j = StencilDimension('j', 0, 1)

        u = Function(name="u", grid=grid, space_order=2)

        # Build `u(x + h_x, y + y_h)`
        term0 = u.xreplace({x: x + x.spacing, y: y + y.spacing})
        term = term0.xreplace({x + x.spacing: x + i*x.spacing,
                               y + y.spacing: y + j*y.spacing})

        # Sum `term` over `i`
        idxsum = IndexSum(term, (i, j))

        # == u(x, y) + u(x, y + h_y) + u(x + h_x, y) + u(x + h_x, y + h_y)
        assert idxsum.evaluate == (u +
                                   u.subs(x, x + x.spacing) +
                                   u.subs(y, y + y.spacing) +
                                   term0)

    def test_index_sum_free_symbols(self):
        grid = Grid((10,))

        x, = grid.dimensions
        i = StencilDimension('i', 0, 1)

        u = Function(name="u", grid=grid)

        idxsum = IndexSum(u.subs(x, x*i), i)

        assert idxsum.free_symbols == {x}

    def test_index_sum_nested(self):
        grid = Grid((10, 10))

        x, y = grid.dimensions
        i = StencilDimension('i', 0, 1)
        j = StencilDimension('j', 0, 1)

        u = Function(name="u", grid=grid, space_order=2)

        term0 = u.xreplace({x: x + x.spacing, y: y + y.spacing})
        term = term0.xreplace({x + x.spacing: x + i*x.spacing,
                               y + y.spacing: y + j*y.spacing})

        idxsum = IndexSum(IndexSum(term, j), i)

        # Expect same output as `test_index_sum_2d`
        assert idxsum.evaluate == (u +
                                   u.subs(x, x + x.spacing) +
                                   u.subs(y, y + y.spacing) +
                                   term0)

    def test_dot_like(self):
        grid = Grid((10, 10))

        x, y = grid.dimensions
        i = StencilDimension('i', 0, 1)

        u = Function(name="u", grid=grid, space_order=2)
        v = Function(name="v", grid=grid, space_order=2)

        ui = u.subs(x, x + i*x.spacing)
        vi = v.subs(y, y + i*y.spacing)

        # Sum `term` over `i`
        idxsum = IndexSum(ui*vi, i)

        assert idxsum.evaluate == u*v + u.subs(x, x + x.spacing)*v.subs(y, y + y.spacing)

    def test_index_derivative(self):
        grid = Grid((10,))
        x, = grid.dimensions

        i = StencilDimension('i', 0, 2)

        u = Function(name="u", grid=grid, space_order=2)

        ui = u.subs(x, x + i*x.spacing)
        w = Weights(name='w0', dimensions=i, initvalue=[-0.5, 0, 0.5])

        idxder = IndexDerivative(ui*w, {x: i})

        assert idxder.evaluate == -0.5*u + 0.5*ui.subs(i, 2)

        # Make sure subs works as expected
        v = Function(name="v", grid=grid, space_order=2)

        vi0 = v.subs(x, x + i*x.spacing)
        vi1 = idxder.subs(ui, vi0)

        assert IndexDerivative(vi0*w, {x: i}) == vi1

    def test_dx2(self):
        grid = Grid(shape=(4, 4))

        f = TimeFunction(name='f', grid=grid, space_order=4)

        term0 = f.dx2.evaluate
        assert isinstance(term0, EvalDerivative)

        term1 = f.dx2._evaluate(expand=False)
        assert isinstance(term1, IndexDerivative)
        assert term1.depth == 1
        term1 = term1.evaluate
        assert isinstance(term1, Add)  # devito.fd.Add

        # Check that the first partially evaluated then fully evaluated
        # `term1` matches up the fully evaluated `term0`
        assert Add(*term0.args) == term1

    def test_dxdy(self):
        grid = Grid(shape=(4, 4))

        f = TimeFunction(name='f', grid=grid, space_order=4)

        term0 = f.dx.dy.evaluate
        assert isinstance(term0, EvalDerivative)

        term1 = f.dx.dy._evaluate(expand=False)
        assert isinstance(term1, IndexDerivative)
        assert term1.depth == 2
        term1 = term1.evaluate
        assert isinstance(term1, Add)  # devito.fd.Add

        # Through expansion and casting we also check that `term0`
        # is indeed mathematically equivalent to `term1`
        assert Add(*term0.expand().args) == term1.expand()

    def test_dxdy_v2(self):
        grid = Grid(shape=(4, 4))

        f = TimeFunction(name='f', grid=grid, space_order=4)

        term1 = f.dxdy._evaluate(expand=False)
        assert len(term1.find(IndexDerivative)) == 2


def bypass_uneval(expr):
    unevals = expr.find(EvalDerivative)
    mapper = {i: Add(*i.args) for i in unevals}
    return expr.xreplace(mapper)

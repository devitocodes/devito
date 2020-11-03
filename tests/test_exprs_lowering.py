from devito import Grid, TimeFunction, Function, Operator, Eq
from devito.passes.equations.linearity import _is_const_coeff
from devito.tools import timed_region


class TestCollectDerivatives():
    """
    Testing Derivatives and expressions of Derivatives collection.
    """

    def test_is_const_coeff_time(self):
        """
        test that subdimension and parent are not miss-interpreted as
        constants.
        """
        grid = Grid((10,))
        f = TimeFunction(name="f", grid=grid, save=10)
        g = TimeFunction(name="g", grid=grid)
        assert not _is_const_coeff(g, f.dt)
        assert not _is_const_coeff(f, g.dt)

    def test_expr_collection(self):
        """
        Test that expressions with different time dimensions are not collected.
        """
        grid = Grid((10,))
        f = TimeFunction(name="f", grid=grid, save=10)
        f2 = TimeFunction(name="f2", grid=grid, save=10)
        g = TimeFunction(name="g", grid=grid)
        g2 = TimeFunction(name="g2", grid=grid)

        w = Function(name="w", grid=grid)
        eq = Eq(w, f.dt*g + f2.dt*g2)

        with timed_region('x'):
            # Since all Function are time dependent, there should be no collection
            # and produce the same result as with the pre evaluated expression
            expr = Operator._lower_exprs([eq])[0]
            expr2 = Operator._lower_exprs([eq.evaluate])[0]

        assert expr == expr2

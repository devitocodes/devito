import pytest
import numpy as np

from devito import (Grid, TimeFunction, SparseTimeFunction, Function, Operator, Eq,
                    configuration)
from devito.ir import FindSymbols, retrieve_iteration_tree
from devito.passes.equations.linearity import _is_const_coeff
from devito.tools import timed_region


class TestCollectDerivatives(object):

    """
    Test collect_derivatives and all mechanisms used by collect_derivatives
    indirectly.
    """

    def test_is_const_coeff_time(self):
        """
        test that subdimension and parent are not misinterpreted as constants.
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


class TestBuffering(object):

    def test_basic(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1
        assert buffers.pop().symbolic_shape[0] == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('async_degree', [2, 4])
    def test_async_degree(self, async_degree):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('buffering', {'buf-async-degree': async_degree}))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1
        assert buffers.pop().symbolic_shape[0] == async_degree

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    def test_two_heterogeneous_buffers(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)
        v1 = TimeFunction(name='v', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + v + 1),
                Eq(v.forward, u + v + v.backward)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 3
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_unread_buffered_function(self):
        nt = 10
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid)
        v1 = TimeFunction(name='v', grid=grid)

        eqns = [Eq(v.forward, v + 1, implicit_dims=time),
                Eq(u, v)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 1
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_over_injection(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
        rec = SparseTimeFunction(name='rec', grid=grid, npoint=1, nt=nt)
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2, save=nt)
        u1 = TimeFunction(name="u", grid=grid, time_order=2, space_order=2, save=nt)

        src.data[:] = 1.

        eqns = ([Eq(u.forward, u + 1)] +
                src.inject(field=u.forward, expr=src) +
                rec.interpolate(expr=u.forward))

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) ==\
            5 + bool(configuration['language'] != 'C')
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

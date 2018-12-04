import pytest

from devito import Eq, Grid, Operator, TimeFunction, configuration  # noqa
from devito.ir.equations import DummyEq
from devito.ir.iet import FindNodes
from devito.ops.transformer import make_ops_ast
from devito.ops.node_factory import OPSNodeFactory

pytestmark = pytest.mark.skipif(configuration['backend'] != 'ops',
                                reason="'ops' wasn't selected as backend on startup")


class TestOPSExpression(object):

    @pytest.mark.parametrize('equation, expected', [
        ('DummyEq(u[t,x,y],1)', 'Eq(ut1[OPS_ACC0(0,0)], ut0[OPS_ACC1(0,0)] + 1)'),
    ])
    def test_ast_convertion(self, equation, expected):
        """
        Tests OPS generated expressions.
        Some basic tests for the generated expression with ops.

        Parameters
        ----------
        equation : str
            An string with a :class:`Eq`to be evalueted.
        expected : str
            Expected expression to be generated from devito.
        """

        grid = Grid(shape=(4, 4))  # noqa
        t = grid.stepping_dim
        x, y = grid.dimensions

        u = TimeFunction(name='u', grid=grid)  # noqa

        test = eval(equation)

        nfops = OPSNodeFactory()
        result = make_ops_ast(test, nfops)

        assert str(result) == expected

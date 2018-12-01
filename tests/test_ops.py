import pytest

from devito import Eq, Grid, Operator, TimeFunction, configuration # noqa
from devito.ir.iet import FindNodes, Expression

pytestmark = pytest.mark.skipif(configuration['backend'] != 'ops',
                                reason="'ops' wasn't selected as backend on startup")


class TestOpsExpression(object):

    @pytest.mark.parametrize('equation, expected', [
        ('Eq(u.forward,u+1)', 'Eq(ut1[OPS_ACC0(0,0)], ut0[OPS_ACC1(0,0)] + 1)'),
        ('Eq(u.forward,2*u+4)',
         'Eq(ut1[OPS_ACC0(0,0)], 2*ut0[OPS_ACC1(0,0)] + 4)'),
    ])
    def test_ops_expression_convertion(self, equation, expected):
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
        u = TimeFunction(name='u', grid=grid)  # noqa

        eq_input = eval(equation)
        op = Operator(eq_input)

        for _, func in op._func_table.items():

            eq_input_calculated = [i.expr for i in FindNodes(Expression).visit(func)]

            for expr in eq_input_calculated:
                assert str(expr) == expected

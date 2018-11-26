import pytest

from devito import Grid, TimeFunction, Operator, Eq, configuration
from devito.ir.iet import FindNodes, Expression, printAST

pytestmark = pytest.mark.skipif(configuration['backend'] != 'ops',
                                reason="'ops' wasn't selected as backend on startup")


class TestOpsSyntax(object):
    """
        Check if the ops syntax is being written correctly.
    """

    def test_ops_declarations(self):
        """
            Creates a simple one dimension grid with 0's and add 1's to the grid.
        """
        grid = Grid(shape=10)

        u = TimeFunction(name='u', grid=grid)

        eq = Eq(u.forward, u + 1.0)

        op = Operator(eq)

        # Checks ops init declaration
        assert 'ops_init(0,NULL,1);' in str(op)
        # Checks ops_exit declaration
        assert 'ops_exit();' in str(op)


class TestOpsExpression(object):
    """
        Tests OPS generated expressions.
        Some basic tests for the generated expression with ops.
    """
    @pytest.mark.parametrize('expr, expected', [
        ('Eq(u.forward,u+1)', '<Expression ut1[OPS_ACC0(0,0)] = ut0[OPS_ACC1(0,0)] + 1>'),
        ('Eq(u.forward,2*u+4)',
         '<Expression ut1[OPS_ACC0(0,0)] = 2*ut0[OPS_ACC1(0,0)] + 4>'),
    ])
    def test_ops_expression_convertion(self, expr, expected):
        """[summary]

        :param expr: Expression as devito uses.
        :type expr: string
        :param expected: Expected output in ops syntax
        :type expected: string
        """
        grid = Grid(shape=(4, 4))  # noqa
        u = TimeFunction(name='u', grid=grid)  # noqa
        expr = eval(expr)
        op = Operator(expr)

        # Check code generation
        for _, func in op._func_table.items():
            equation = FindNodes(Expression).visit(func)
            assert printAST(equation) == str(expected)

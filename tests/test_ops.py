import pytest

from conftest import skipif
from devito import Eq, Grid, Operator, TimeFunction, configuration  # noqa
from devito.ir.equations import ClusterizedEq # noqa
from devito.ir.iet import Callable,  Expression # noqa
from devito.symbolics import indexify

pytestmark = skipif('noops', whole_module=True)

# All ops-specific imports *must* be avoided if `backend != ops`, otherwise
# a backend reinitialization would be triggered via `devito/ops/.__init__.py`,
# thus invalidating all of the future tests. This is guaranteed by the
# `pytestmark` above
from devito.ops.node_factory import OPSNodeFactory  # noqa
from devito.ops.operator import OperatorOPS # noqa
from devito.ops.transformer import opsit, make_ops_ast  # noqa


class TestOPSExpression(object):

    @pytest.mark.parametrize('equation, expected', [
        ('Eq(u,3*a - 4**a)', 'Eq(ut0[OPS_ACC0(0)], -2.97015324253729)'),
        ('Eq(u, u.dxl)',
         'Eq(ut0[OPS_ACC0(0)], -2.0*ut0[OPS_ACC0(-1)]/h_x + '
            '0.5*ut0[OPS_ACC0(-2)]/h_x + 1.5*ut0[OPS_ACC0(0)]/h_x)'),
        ('Eq(v,1)', 'Eq(vt0[OPS_ACC0(0,0)], 1)'),
        ('Eq(v,v.dxl + v.dxr - v.dyr - v.dyl)',
         'Eq(vt0[OPS_ACC0(0,0)], 2.0*vt0[OPS_ACC0(0,-1)]/h_y - '
            '0.5*vt0[OPS_ACC0(0,-2)]/h_y - 2.0*vt0[OPS_ACC0(0,1)]/h_y + '
            '0.5*vt0[OPS_ACC0(0,2)]/h_y - 2.0*vt0[OPS_ACC0(-1,0)]/h_x + '
            '0.5*vt0[OPS_ACC0(-2,0)]/h_x + 2.0*vt0[OPS_ACC0(1,0)]/h_x - '
            '0.5*vt0[OPS_ACC0(2,0)]/h_x)'),
        ('Eq(v,v**2 - 3*v)',
         'Eq(vt0[OPS_ACC0(0,0)], vt0[OPS_ACC0(0,0)]**2 - 3*vt0[OPS_ACC0(0,0)])'),
        ('Eq(v,a*v + b)',
         'Eq(vt0[OPS_ACC0(0,0)], 1.43*vt0[OPS_ACC0(0,0)] + 9.87e-7)'),
        ('Eq(v.dt,1/v + v**-1)',
         'Eq(-vt0[OPS_ACC1(0,0)]/dt + vt1[OPS_ACC0(0,0)]/dt, 2/vt0[OPS_ACC1(0,0)])'),
        ('Eq(w,c*w**2)',
         'Eq(wt0[OPS_ACC0(0,0,0)], 999999999999999*wt0[OPS_ACC0(0,0,0)]**2)'),
        ('Eq(w,u + v )',
         'Eq(wt0[OPS_ACC0(0,0,0)], ut0[OPS_ACC1(0)] + vt0[OPS_ACC2(0,0)])'),
        ('Eq(u.forward,u+1)', 'Eq(ut1[OPS_ACC0(0)], ut0[OPS_ACC1(0)] + 1)'),
        ('Eq(u, u.laplace)',
         'Eq(ut0[OPS_ACC0(0)], ut0[OPS_ACC0(-1)]/h_x**2 - '
            '2.0*ut0[OPS_ACC0(0)]/h_x**2 + ut0[OPS_ACC0(1)]/h_x**2)'),
    ])
    def test_ast_convertion(self, equation, expected):
        """
        Test OPS generated expressions for 1, 2 and 3 space dimensions.

        Parameters
        ----------
        equation : str
            A string with a :class:`Eq`to be evaluated.
        expected : str
            Expected expression to be generated from devito.
        """

        grid_1d = Grid(shape=(4))
        grid_2d = Grid(shape=(4, 4))
        grid_3d = Grid(shape=(4, 4, 4))

        a = 1.43  # noqa
        b = 0.000000987  # noqa
        c = 999999999999999  # noqa

        u = TimeFunction(name='u', grid=grid_1d, space_order=2)  # noqa
        v = TimeFunction(name='v', grid=grid_2d, space_order=2)  # noqa
        w = TimeFunction(name='w', grid=grid_3d, space_order=2)  # noqa

        nfops = OPSNodeFactory()
        result = make_ops_ast(indexify(eval(equation)), nfops)

        assert str(result) == expected


class TestOPSLifting(object):

    @pytest.mark.parametrize('eq, expected', [
        ('Eq(u,3*a - 4**a)', 'void Kernel0(float * ut00)\n\
{\n\
  ut00[OPS_ACC0(0)] = -2.97015324253729F;\n\
}'),
        ('Eq(w,u + v)', 'void Kernel0(const float * ut00, const float * vt00, float * wt00)\n\
{\n\
  wt00[OPS_ACC0(0)] = ut00[OPS_ACC1(0)] + vt00[OPS_ACC2(0)];\n\
}'),
    ])
    def test_expr_lifting(self, eq, expected):
        grid_1d = Grid(shape=(4))

        a = 1.43  # noqa

        u = TimeFunction(name='u', grid=grid_1d, space_order=2)  # noqa
        v = TimeFunction(name='v', grid=grid_1d, space_order=2)  # noqa
        w = TimeFunction(name='w', grid=grid_1d, space_order=2)  # noqa

        op = OperatorOPS(eval(eq))

        assert str(op._func_table["Kernel0"].root) == expected

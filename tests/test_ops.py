import pytest

from conftest import skipif
from devito import Eq, Grid, Operator, TimeFunction, configuration  # noqa

pytestmark = skipif('noops', whole_module=True)

# All ops-specific imports *must* be avoided if `backend != ops`, otherwise
# a backend reinitialization would be triggered via `devito/ops/.__init__.py`,
# thus invalidating all of the future tests. This is guaranteed by the
# `pytestmark` above
from devito.ops.node_factory import OPSNodeFactory  # noqa
from devito.ops.operator import OperatorOPS # noqa
from devito.ops.transformer import make_ops_ast  # noqa


class TestOPSExpression(object):

    @pytest.mark.parametrize('equation, expected', [
        ('Eq(u,3*a - 4**a)', 'void OPS_Kernel_0(ACC<float> & ut0)\n'
         '{\n  ut0(0) = -2.97015324253729F;\n}'),
        ('Eq(u, u.dxl)',
         'void OPS_Kernel_0(ACC<float> & ut0, const float *h_x)\n'
         '{\n  r0 = 1.0/*h_x;\n  '
         'ut0(0) = -2.0F*ut0(-1)*r0 + 5.0e-1F*ut0(-2)*r0 + 1.5F*ut0(0)*r0;\n}'),
        ('Eq(v,1)', 'void OPS_Kernel_0(ACC<float> & vt0)\n'
         '{\n  vt0(0, 0) = 1;\n}'),
        ('Eq(v,v.dxl + v.dxr - v.dyr - v.dyl)',
         'void OPS_Kernel_0(ACC<float> & vt0, const float *h_x, const float *h_y)\n'
         '{\n  r1 = 1.0/*h_y;\n  r0 = 1.0/*h_x;\n  '
         'vt0(0, 0) = 5.0e-1F*(-vt0(2, 0)*r0 + vt0(-2, 0)*r0 - '
         'vt0(0, -2)*r1 + vt0(0, 2)*r1) + 2.0F*(-vt0(2, 0)*r0 + '
         'vt0(-2, 0)*r0 - vt0(0, -2)*r1 + vt0(0, 2)*r1);\n}'),
        ('Eq(v,v**2 - 3*v)',
         'void OPS_Kernel_0(ACC<float> & vt0)\n'
         '{\n  vt0(0, 0) = -3*vt0(0, 0) + vt0(0, 0)*vt0(0, 0);\n}'),
        ('Eq(v,a*v + b)',
         'void OPS_Kernel_0(ACC<float> & vt0)\n'
         '{\n  vt0(0, 0) = 9.87e-7F + 1.43F*vt0(0, 0);\n}'),
        ('Eq(w,c*w**2)',
         'void OPS_Kernel_0(ACC<float> & wt0)\n'
         '{\n  wt0(0, 0, 0) = 999999999999999*(wt0(0, 0, 0)*wt0(0, 0, 0));\n}'),
        ('Eq(u.forward,u+1)',
         'void OPS_Kernel_0(const ACC<float> & ut0, ACC<float> & ut1)\n'
         '{\n  ut1(0) = 1 + ut0(0);\n}'),
    ])
    def test_kernel_generation(self, equation, expected):
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

        operator = OperatorOPS(eval(equation))

        assert str(operator._ops_kernels[0]) == expected

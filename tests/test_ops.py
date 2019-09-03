import itertools
import pytest
import numpy as np

from conftest import skipif
from sympy import Integer
from sympy.core.numbers import Zero, One  # noqa

pytestmark = skipif('noops', whole_module=True)

# All ops-specific imports *must* be avoided if `backend != ops`, otherwise
# a backend reinitialization would be triggered via `devito/ops/.__init__.py`,
# thus invalidating all of the future tests. This is guaranteed by the
# `pytestmark` above
from devito import Eq, Function, Grid, Operator, TimeFunction, configuration  # noqa
from devito.ops.node_factory import OPSNodeFactory  # noqa
from devito.ops.transformer import create_ops_arg, create_ops_dat, make_ops_ast, to_ops_stencil  # noqa
from devito.ops.types import OpsAccessible, OpsDat, OpsStencil, OpsBlock  # noqa
from devito.ops.utils import namespace  # noqa
from devito.symbolics import Byref, indexify, Literal  # noqa
from devito.tools import dtype_to_cstr  # noqa
from devito.types import Buffer, Constant, Symbol  # noqa


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
         'vt0(0, -2)*r1 + vt0(0, 2)*r1) + 2.0F*(-vt0(-1, 0)*r0 + '
         'vt0(1, 0)*r0 - vt0(0, 1)*r1 + vt0(0, -1)*r1);\n}'),
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
            A string with a Eq to be evaluated.
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

        operator = Operator(eval(equation))

        assert str(operator._ops_kernels[0]) == expected

    @pytest.mark.parametrize('equation, expected', [
        ('Eq(u,3*a - 4**a)', '{ "ut": [[0]] }'),
        ('Eq(u, u.dxl)', '{ "ut": [[0], [-1], [-2]] }'),
        ('Eq(u,v+1)', '{ "ut": [[0]], "vt": [[0]] }')
    ])
    def test_accesses_extraction(self, equation, expected):
        grid_1d = Grid(shape=(4))
        grid_3d = Grid(shape=(4, 4, 4))

        a = 1.43  # noqa
        c = 999999999999999  # noqa
        u = TimeFunction(name='u', grid=grid_1d, space_order=2)  # noqa
        v = TimeFunction(name='v', grid=grid_1d, space_order=2)  # noqa
        w = TimeFunction(name='w', grid=grid_3d, space_order=2)  # noqa

        node_factory = OPSNodeFactory()

        make_ops_ast(indexify(eval(equation).evaluate), node_factory)

        result = eval(expected)

        for k, v in node_factory.ops_args_accesses.items():
            assert len(v) == len(result[k.name])
            for idx in result[k.name]:
                assert idx in v

    @pytest.mark.parametrize('_accesses', [
        '[[Zero(), Zero()]]', '[[Zero(), Zero()], [One(), One()]]'
    ])
    def test_to_ops_stencil(self, _accesses):
        param = Symbol('foo')
        accesses = eval(_accesses)

        stencil_name = 's2d_foo_%spt' % len(accesses)

        stencil, result = to_ops_stencil(param, accesses)

        assert stencil.name == stencil_name.upper()

        assert result[0].expr.lhs.name == stencil_name
        assert result[0].expr.rhs.params == tuple(itertools.chain(*accesses))

        assert result[1].expr.lhs == stencil
        assert type(result[1].expr.rhs) == namespace['ops_decl_stencil']
        assert result[1].expr.rhs.args == (
            2,
            len(accesses),
            Symbol(stencil_name),
            Literal('"%s"' % stencil_name.upper())
        )

    @pytest.mark.parametrize('equation,expected', [
        ('Eq(u.forward, u + 1)',
         '[\'ops_dat u_dat[2] = {ops_decl_dat(block, 1, u_dim, u_base, u_d_m, u_d_p, '
         '&(u[0]), "float", "ut0"), ops_decl_dat(block, 1, u_dim, u_base, u_d_m, u_d_p, '
         '&(u[1]), "float", "ut1")}\']')
    ])
    def test_create_ops_dat(self, equation, expected):
        grid = Grid(shape=(4))

        u = TimeFunction(name='u', grid=grid, space_order=2)  # noqa
        v = Function(name='u', grid=grid, space_order=2)  # noqa

        op = Operator(eval(equation))

        for i in eval(expected):
            assert i in str(op)

    def test_create_ops_dat_function(self):
        grid = Grid(shape=(4))

        u = Function(name='u', grid=grid, space_order=2)

        block = OpsBlock('block')

        name_to_ops_dat = {}

        result = create_ops_dat(u, name_to_ops_dat, block)

        assert name_to_ops_dat['u'].name == namespace['ops_dat_name'](u.name)
        assert name_to_ops_dat['u']._C_typename == namespace['ops_dat_type']

        assert result[0].expr.lhs.name == namespace['ops_dat_dim'](u.name)
        assert result[0].expr.rhs.params == (Integer(4),)

        assert result[1].expr.lhs.name == namespace['ops_dat_base'](u.name)
        assert result[1].expr.rhs.params == (Zero(),)

        assert result[2].expr.lhs.name == namespace['ops_dat_d_p'](u.name)
        assert result[2].expr.rhs.params == (Integer(2),)

        assert result[3].expr.lhs.name == namespace['ops_dat_d_m'](u.name)
        assert result[3].expr.rhs.params == (Integer(-2),)

        assert result[4].expr.lhs == name_to_ops_dat['u']
        assert type(result[4].expr.rhs) == namespace['ops_decl_dat']
        assert result[4].expr.rhs.args == (
            block,
            1,
            Symbol(namespace['ops_dat_dim'](u.name)),
            Symbol(namespace['ops_dat_base'](u.name)),
            Symbol(namespace['ops_dat_d_m'](u.name)),
            Symbol(namespace['ops_dat_d_p'](u.name)),
            Byref(u.indexify((0,))),
            Literal('"%s"' % u._C_typedata),
            Literal('"u"')
        )

    def test_create_ops_arg_constant(self):
        a = Constant(name='*a')

        res = create_ops_arg(a, {}, {})

        assert type(res) == namespace['ops_arg_gbl']
        assert str(res.args[0]) == str(Byref(Constant(name='a')))
        assert res.args[1] == 1
        assert res.args[2] == Literal('"%s"' % dtype_to_cstr(a.dtype))
        assert res.args[3] == namespace['ops_read']

    @pytest.mark.parametrize('read', [True, False])
    def test_create_ops_arg_function(self, read):
        u = OpsAccessible('u', np.float32, read)

        dat = OpsDat('u_dat')
        stencil = OpsStencil('stencil')

        res = create_ops_arg(u, {'u': dat}, {u: stencil})

        assert type(res) == namespace['ops_arg_dat']
        assert res.args == (
            dat,
            1,
            stencil,
            Literal('"%s"' % dtype_to_cstr(u.dtype)),
            namespace['ops_read'] if read else namespace['ops_write']
        )

    @pytest.mark.parametrize('equation, expected', [
        ('Eq(u.forward, u.dt2 + u.dxr - u.dyr - u.dyl)',
            'ops_block block = ops_decl_block(2, "block");'),
        ('Eq(u.forward,u+1)',
            'ops_block block = ops_decl_block(2, "block");')
    ])
    def test_create_ops_block(self, equation, expected):
        """
        Test if ops_block has been successfully generated
        """
        grid_2d = Grid(shape=(4, 4))
        u = TimeFunction(name='u', grid=grid_2d, time_order=2, save=Buffer(10))  # noqa
        operator = Operator(eval(equation))

        assert expected in str(operator.ccode)

    @pytest.mark.parametrize('equation,expected', [
        ('Eq(u.forward, u + 1)',
         '[\'ops_dat_fetch_data(u_dat[(time_M)%(2)],0,&(u[(time_M)%(2)]));\','
         '\'ops_dat_fetch_data(u_dat[(time_M - 1)%(2)],0,&(u[(time_M - 1)%(2)]));\']'),
        ('Eq(v, v.dx + u)',
         '[\'ops_dat_fetch_data(v_dat[(time_M)%(2)],0,&(v[(time_M)%(2)]));\','
         '\'ops_dat_fetch_data(v_dat[(time_M - 1)%(2)],0,&(v[(time_M - 1)%(2)]));\','
         '\'ops_dat_fetch_data(u_dat[(time_M)%(2)],0,&(u[(time_M)%(2)]));\','
         '\'ops_dat_fetch_data(u_dat[(time_M - 1)%(2)],0,&(u[(time_M - 1)%(2)]));\']'),
    ])
    def test_create_ops_dat_fetch_data(self, equation, expected):

        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa

        op = Operator(eval(equation))
        for i in eval(expected):
            assert i in str(op)

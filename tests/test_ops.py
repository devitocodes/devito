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
from devito.ir.iet import Call, Callable # noqa
from devito.ops.node_factory import OPSNodeFactory  # noqa
from devito.ops.operator import OperatorOPS  # noqa
from devito.ops.transformer import create_ops_dat, make_ops_ast, to_ops_stencil, create_ops_arg, create_ops_par_loop  # noqa
from devito.ops.types import Array, OpsAccessible, OpsBlock, OpsDat, OpsStencil  # noqa
from devito.ops.utils import namespace  # noqa
from devito.symbolics import Byref, Literal, indexify  # noqa
from devito.types import Constant, DefaultDimension, Symbol  # noqa


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

        result = to_ops_stencil(param, accesses, {})

        assert result[0].expr.lhs.name == stencil_name
        assert result[0].expr.rhs.params == tuple(itertools.chain(*accesses))

        assert result[1].expr.lhs.name == stencil_name.upper()
        assert result[1].expr.rhs.name == namespace['ops_decl_stencil'].name
        assert result[1].expr.rhs.args == (
            2,
            len(accesses),
            Symbol(stencil_name),
            Literal('"%s"' % stencil_name.upper())
        )

    def test_create_ops_dat_time_function(self):
        grid = Grid(shape=(4))

        u = TimeFunction(name='u', grid=grid, space_order=2)

        block = OpsBlock('block')

        name_to_ops_dat = {}

        result = create_ops_dat(u, name_to_ops_dat, block)

        assert name_to_ops_dat['ut0'].base.name == namespace['ops_dat_name'](u.name)
        assert name_to_ops_dat['ut0'].indices == (Symbol('t0'),)
        assert name_to_ops_dat['ut1'].base.name == namespace['ops_dat_name'](u.name)
        assert name_to_ops_dat['ut1'].indices == (Symbol('t1'),)

        assert result[0].expr.lhs.name == namespace['ops_dat_dim'](u.name)
        assert result[0].expr.rhs.params == (Integer(4),)

        assert result[1].expr.lhs.name == namespace['ops_dat_base'](u.name)
        assert result[1].expr.rhs.params == (Zero(),)

        assert result[2].expr.lhs.name == namespace['ops_dat_d_p'](u.name)
        assert result[2].expr.rhs.params == (Integer(2),)

        assert result[3].expr.lhs.name == namespace['ops_dat_d_m'](u.name)
        assert result[3].expr.rhs.params == (Integer(-2),)

        assert result[4].expr.lhs.name == namespace['ops_dat_name'](u.name)
        assert len(result[4].expr.rhs.params) == 2
        assert result[4].expr.rhs.params[0].name == namespace['ops_decl_dat'].name
        assert result[4].expr.rhs.params[0].args == (
            block,
            1,
            Symbol(namespace['ops_dat_dim'](u.name)),
            Symbol(namespace['ops_dat_base'](u.name)),
            Symbol(namespace['ops_dat_d_m'](u.name)),
            Symbol(namespace['ops_dat_d_p'](u.name)),
            Byref(u.indexify((0,))),
            Literal('"%s"' % u._C_typedata),
            Literal('"ut0"')
        )
        assert result[4].expr.rhs.params[1].name == namespace['ops_decl_dat'].name
        assert result[4].expr.rhs.params[1].args == (
            block,
            1,
            Symbol(namespace['ops_dat_dim'](u.name)),
            Symbol(namespace['ops_dat_base'](u.name)),
            Symbol(namespace['ops_dat_d_m'](u.name)),
            Symbol(namespace['ops_dat_d_p'](u.name)),
            Byref(u.indexify((1,))),
            Literal('"%s"' % u._C_typedata),
            Literal('"ut1"')
        )

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
        assert result[4].expr.rhs.name == namespace['ops_decl_dat'].name
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

    def test_create_ops_arg_gbl(self):
        arg_const1 = Constant(name='*arg_const')
        arg_const2 = Constant(name='arg_const')

        result1 = create_ops_arg(arg_const1, {}, {})
        result2 = create_ops_arg(arg_const2, {}, {})

        assert result1.is_Call

        assert result1.name == namespace['ops_arg_gbl']
        assert result2.name == namespace['ops_arg_gbl']

        assert result1.ccode.text == 'ops_arg_gbl(&arg_const,1,"float",OPS_READ)'
        assert result2.ccode.text == 'ops_arg_gbl(&arg_const,1,"float",OPS_READ)'

    def test_create_ops_arg_dat(self):
        u = OpsAccessible('u', np.float32, True)
        v = OpsAccessible('v', np.float32, False)

        read = create_ops_arg(u, {'u': 'dat'}, {'u': 'stencil'})
        write = create_ops_arg(v, {'v': 'dat'}, {'v': 'stencil'})

        assert read.is_Call
        assert write.is_Call

        assert read.name == namespace['ops_arg_dat']
        assert write.name == namespace['ops_arg_dat']

        assert read.ccode.text == 'ops_arg_dat(dat,1,stencil,"float",OPS_READ)'
        assert write.ccode.text == 'ops_arg_dat(dat,1,stencil,"float",OPS_WRITE)'

    def test_create_ops_par_loop(self):
        block = OpsBlock('block')
        name_to_ops_dat = {'arg_const': 'arg_const_dat',
                           'arg_dat1': 'arg_dat1_dat',
                           'arg_dat2': 'arg_dat2_dat'}
        name_to_ops_stencil = {'arg_const': 'arg_const_stencil',
                               'arg_dat1': 'arg_dat1_stencil',
                               'arg_dat2': 'arg_dat2_stencil'}

        arg_const = Constant(name='*arg_const')
        arg_dat1 = OpsAccessible('arg_dat1', np.float32, True)
        arg_dat2 = OpsAccessible('arg_dat2', np.float32, False)

        arguments = [arg_const, arg_dat1, arg_dat2]
        args = [create_ops_arg(a, name_to_ops_dat, name_to_ops_stencil)
                for a in arguments]

        ops_kernel = Callable(namespace['ops_kernel'](0), [], "void")
        par_loop_array = Array(
            name=namespace['ops_par_loop_range_name'],
            dimensions=(DefaultDimension(name='len', default_value=0),),
            dtype=np.int32,
        )

        par_loop = create_ops_par_loop(block, ops_kernel, par_loop_array, args)

        mandatory_parameters = ['kernel', 'name', 'block', 'dims', 'kernel_range']

        assert par_loop.ccode.text.replace(' ', '') == 'ops_par_loop(OPS_Kernel_0,"OPS_Kernel_0",block,2,par_loop_range,ops_arg_gbl(&arg_const,1,"float",OPS_READ),ops_arg_dat(arg_dat1_dat,1,arg_dat1_stencil,"float",OPS_READ),ops_arg_dat(arg_dat2_dat,1,arg_dat2_stencil,"float",OPS_WRITE))' # noqa
        assert len(par_loop.arguments) == len(mandatory_parameters) + len(args)
        assert isinstance(par_loop, Call)
        for arg in par_loop.arguments[len(mandatory_parameters):]:
            assert isinstance(arg, Call)

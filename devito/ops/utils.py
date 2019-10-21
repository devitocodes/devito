from collections import OrderedDict, namedtuple
from sympy import Function

from devito.ir.iet.nodes import Call
from devito.symbolics import Macro


# OPS Conventions
namespace = OrderedDict()
AccessibleInfo = namedtuple(
    'AccessibleInfo',
    ['accessible', 'time', 'origin_name'])

OpsDatDecl = namedtuple(
    'OpsDatDecl',
    ['dim_val', 'base_val', 'd_p_val', 'd_m_val', 'ops_decl_dat'])

OpsArgDecl = namedtuple(
    'OpsArgDecl',
    ['ops_type', 'ops_name', 'elements_per_point', 'dtype', 'rw_flag'])

# OPS API
namespace['ops_init'] = 'ops_init'
namespace['ops_partition'] = 'ops_partition'
namespace['ops_timing_output'] = 'ops_timing_output'
namespace['ops_exit'] = 'ops_exit'
namespace['ops_par_loop'] = 'ops_par_loop'
namespace['ops_dat_fetch_data'] = lambda ops_dat, data: Call(
    name='ops_dat_fetch_data', arguments=[ops_dat, 0, data])

namespace['ops_decl_stencil'] = Function(name='ops_decl_stencil')
namespace['ops_decl_block'] = Function(name='ops_decl_block')
namespace['ops_decl_dat'] = Function(name='ops_decl_dat')
namespace['ops_arg_dat'] = Function(name='ops_arg_dat')
namespace['ops_arg_gbl'] = Function(name='ops_arg_gbl')

namespace['ops_read'] = Macro('OPS_READ')
namespace['ops_write'] = Macro('OPS_WRITE')

namespace['ops_stencil_type'] = 'ops_stencil'
namespace['ops_block_type'] = 'ops_block'
namespace['ops_dat_type'] = 'ops_dat'

# Naming conventions
namespace['ops_define_dimension'] = lambda i: '#define OPS_%sD' % i
namespace['ops_kernel'] = lambda i: 'OPS_Kernel_%s' % i
namespace['ops_stencil_name'] = lambda dims, name, pts: 's%dd_%s_%dpt' % (dims, name, pts)
namespace['ops_dat_dim'] = lambda i: '%s_dim' % i
namespace['ops_dat_base'] = lambda i: '%s_base' % i
namespace['ops_dat_d_p'] = lambda i: '%s_d_p' % i
namespace['ops_dat_d_m'] = lambda i: '%s_d_m' % i
namespace['ops_dat_name'] = lambda i: '%s_dat' % i

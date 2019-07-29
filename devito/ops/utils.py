from collections import OrderedDict
from sympy import Function


# OPS Conventions
namespace = OrderedDict()

# OPS Kernel strings.
namespace['ops_init'] = 'ops_init'
namespace['ops_partition'] = 'ops_partition'
namespace['ops_timing_output'] = 'ops_timing_output'
namespace['ops_exit'] = 'ops_exit'
namespace['ops_define_dimension'] = lambda i: '#define OPS_%sD' % i
namespace['ops_kernel'] = lambda i: 'OPS_Kernel_%s' % i
namespace['ops_stencil_name'] = lambda dims, name, pts: 's%dd_%s_%dpt' % (dims, name, pts)
namespace['ops_decl_stencil'] = Function(name='ops_decl_stencil')
namespace['ops_decl_block'] = Function(name='ops_decl_block')
namespace['ops_decl_dat'] = Function(name='ops_decl_dat')
namespace['ops_stencil_type'] = 'ops_stencil'
namespace['ops_block_type'] = 'ops_block'
namespace['ops_dat_type'] = 'ops_dat'
namespace['ops_dat_dim'] = lambda i: '%s_dim' % i
namespace['ops_dat_base'] = lambda i: '%s_base' % i
namespace['ops_dat_d_p'] = lambda i: '%s_d_p' % i
namespace['ops_dat_d_m'] = lambda i: '%s_d_m' % i
namespace['ops_dat_name'] = lambda i: '%s_dat' % i

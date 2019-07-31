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
namespace['ops_stencil_type'] = 'ops_stencil'

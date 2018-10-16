from collections import OrderedDict


# OPS Conventions
namespace = OrderedDict()

namespace['call-ops_init'] = 'ops_init'

namespace['name-ops_grid'] = 'grid'
namespace['type-ops_grid'] = 'ops_block'
namespace['call-ops_grid'] = 'ops_decl_block'

namespace['name-ops_dat'] = lambda i: 'dat_u%s' % i
namespace['type-ops_dat'] = 'ops_dat'
namespace['call-ops_dat'] = 'ops_decl_dat'

namespace['name-ops_stencil'] = lambda i: 'stencil_%s' % i
namespace['type-ops_stencil'] = 'ops_stencil'
namespace['call-ops_stencil'] = 'ops_decl_stencil'

namespace['call-ops_par_loop'] = 'ops_par_loop'

namespace['call-ops_exit'] = 'ops_exit'

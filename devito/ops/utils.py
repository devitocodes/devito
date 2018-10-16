from collections import OrderedDict


# OPS Conventions
namespace = OrderedDict()

namespace['call-ops_init'] = 'ops_init'

namespace['name-ops_grid'] = 'grid'
namespace['type-ops_block'] = 'ops_block'
namespace['call-ops_block'] = 'ops_decl_block'

namespace['name-ops_dat'] = lambda i: 'dat_u%s' % i
namespace['type-ops_dat'] = 'ops_dat'
namespace['call-ops_dat'] = 'ops_decl_dat'

namespace['call-ops_get_data'] = 'ops_get_data'

namespace['call-ops_exit'] = 'ops_exit'



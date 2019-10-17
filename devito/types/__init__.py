# Need only from within Devito
from devito.types.basic import *  # noqa

# Needed both within and outside Devito
from devito.types.dimension import *  # noqa
from devito.types.utils import *  # noqa
from devito.types.caching import _SymbolCache, CacheManager  # noqa

# Needed only outside Devito
from devito.types.constant import *  # noqa
from devito.types.grid import *  # noqa
from devito.types.dense import * # noqa
from devito.types.sparse import *  # noqa

from __future__ import absolute_import
import gc

from sympy.core import cache

from devito.operator import *  # noqa
from devito.finite_difference import *  # noqa
from devito.iteration import *  # noqa
from devito.dimension import *  # noqa
from devito.expression import *  # noqa
from devito.stencilkernel import *  # noqa
from devito.interfaces import *  # noqa
from devito.interfaces import _SymbolCache


def clear_cache():
    cache.clear_cache()
    gc.collect()

    for key, val in list(_SymbolCache.items()):
        if val() is None:
            del _SymbolCache[key]


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions

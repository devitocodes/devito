from __future__ import absolute_import
import gc

from sympy.core import cache

from devito.interfaces import *  # noqa
from devito.interfaces import _SymbolCache
from devito.operator import *  # noqa
from devito.finite_difference import *  # noqa
from devito.iteration import *  # noqa


def clear_cache():
    cache.clear_cache()
    gc.collect()

    for key, val in _SymbolCache.items():
        if val() is None:
            del _SymbolCache[key]

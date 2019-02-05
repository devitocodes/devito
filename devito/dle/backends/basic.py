from collections import OrderedDict

import cgen as c
import numpy as np

from devito.dle.backends import AbstractRewriter, dle_pass, complang_ALL
from devito.ir.iet import (Denormals, Call, Callable, List, ArrayCast,
                           Transformer, FindSymbols, retrieve_iteration_tree,
                           filter_iterations, iet_insert_C_decls, derive_parameters)
from devito.parameters import configuration
from devito.symbolics import as_symbol
from devito.types import Scalar, IncrDimension


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)

    @dle_pass
    def _avoid_denormals(self, nodes, state):
        """
        Introduce nodes in the Iteration/Expression tree that will expand to C
        macros telling the CPU to flush denormal numbers in hardware. Denormals
        are normally flushed when using SSE-based instruction sets, except when
        compiling shared objects.
        """
        return (List(body=(Denormals(), nodes)),
                {'includes': ('xmmintrin.h', 'pmmintrin.h')})

    def _compiler_decoration(self, name, default=None):
        key = configuration['compiler'].__class__.__name__
        complang = complang_ALL.get(key, {})
        return complang.get(name, default)

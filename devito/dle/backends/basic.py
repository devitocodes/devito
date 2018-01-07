from __future__ import absolute_import

from collections import OrderedDict
from operator import attrgetter

import cgen as c
import numpy as np

from devito.cgen_utils import ccode
from devito.dle.backends import AbstractRewriter, dle_pass, complang_ALL
from devito.ir.iet import (Denormals, Expression, Call, Callable, List,
                           UnboundedIndex, FindNodes, FindSymbols,
                           NestedTransformer, Transformer,
                           retrieve_iteration_tree, filter_iterations)
from devito.symbolics import as_symbol
from devito.tools import filter_sorted, flatten
from devito.types import Scalar


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._create_elemental_functions(state)

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

    @dle_pass
    def _create_elemental_functions(self, nodes, state):
        """
        Extract :class:`Iteration` sub-trees and move them into :class:`Callable`s.

        Currently, only tagged, elementizable Iteration objects are targeted.
        """
        noinline = self._compiler_decoration('noinline', c.Comment('noinline?'))

        functions = OrderedDict()
        mapper = {}
        for tree in retrieve_iteration_tree(nodes, mode='superset'):
            # Search an elementizable sub-tree (if any)
            tagged = filter_iterations(tree, lambda i: i.tag is not None, 'asap')
            if not tagged:
                continue
            root = tagged[0]
            if not root.is_Elementizable:
                continue
            target = tree[tree.index(root):]

            # Elemental function arguments
            args = []  # Found so far (scalars, tensors)
            maybe_required = set()  # Scalars that *may* have to be passed in
            not_required = set()  # Elemental function locally declared scalars

            # Build a new Iteration/Expression tree with free bounds
            free = []
            for i in target:
                name, bounds = i.dim.name, i.bounds_symbolic
                # Iteration bounds
                start = Scalar(name='%s_start' % name, dtype=np.int32)
                finish = Scalar(name='%s_finish' % name, dtype=np.int32)
                args.extend(zip([ccode(j) for j in bounds], (start, finish)))
                # Iteration unbounded indices
                ufunc = [Scalar(name='%s_ub%d' % (name, j), dtype=np.int32)
                         for j in range(len(i.uindices))]
                args.extend(zip([ccode(j.start) for j in i.uindices], ufunc))
                limits = [Scalar(name=start.name, dtype=np.int32),
                          Scalar(name=finish.name, dtype=np.int32), 1]
                uindices = [UnboundedIndex(j.index, i.dim + as_symbol(k))
                            for j, k in zip(i.uindices, ufunc)]
                free.append(i._rebuild(limits=limits, offsets=None, uindices=uindices))
                not_required.update({i.dim}, set(j.index for j in i.uindices))

            # Construct elemental function body, and inspect it
            free = NestedTransformer(dict((zip(target, free)))).visit(root)
            expressions = FindNodes(Expression).visit(free)
            fsymbols = FindSymbols('symbolics').visit(free)

            # Add all definitely-required arguments
            not_required.update({i.output for i in expressions if i.is_scalar})
            for i in fsymbols:
                if i in not_required:
                    continue
                elif i.is_Array:
                    args.append(("(%s*)%s" % (c.dtype_to_ctype(i.dtype), i.name), i))
                elif i.is_TensorFunction:
                    args.append(("%s_vec" % i.name, i))
                elif i.is_Scalar:
                    args.append((i.name, i))

            # Add all maybe-required arguments that turn out to be required
            maybe_required.update(set(FindSymbols(mode='free-symbols').visit(free)))
            for i in fsymbols:
                not_required.update({as_symbol(i), i.indexify()})
                for j in i.symbolic_shape:
                    maybe_required.update(j.free_symbols)
            required = filter_sorted(maybe_required - not_required,
                                     key=attrgetter('name'))
            args.extend([(i.name, Scalar(name=i.name, dtype=i.dtype)) for i in required])

            call, params = zip(*args)
            name = "f_%d" % root.tag

            # Produce the new Call
            mapper[root] = List(header=noinline, body=Call(name, call))

            # Produce the new Callable
            functions.setdefault(name, Callable(name, free, 'void', flatten(params),
                                                ('static',)))

        # Transform the main tree
        processed = Transformer(mapper).visit(nodes)

        return processed, {'elemental_functions': functions.values()}

    def _compiler_decoration(self, name, default=None):
        key = self.params['compiler'].__class__.__name__
        complang = complang_ALL.get(key, {})
        return complang.get(name, default)

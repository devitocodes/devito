from __future__ import absolute_import
from operator import attrgetter

import cgen as c
import numpy as np

from devito.dse import as_symbol, estimate_cost
from devito.dle import retrieve_iteration_tree
from devito.dle.backends import AbstractRewriter, dle_pass, complang_ALL
from devito.interfaces import ScalarFunction
from devito.nodes import Denormals, Expression, FunCall, Function, List
from devito.tools import filter_sorted, flatten
from devito.visitors import FindNodes, FindSymbols, Transformer


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._create_elemental_functions(state)

    @dle_pass
    def _avoid_denormals(self, state, **kwargs):
        """
        Introduce nodes in the Iteration/Expression tree that will expand to C
        macros telling the CPU to flush denormal numbers in hardware. Denormals
        are normally flushed when using SSE-based instruction sets, except when
        compiling shared objects.
        """
        return {'nodes': (Denormals(),) + state.nodes,
                'includes': ('xmmintrin.h', 'pmmintrin.h')}

    @dle_pass
    def _create_elemental_functions(self, state, **kwargs):
        """
        Extract :class:`Iteration` sub-trees and move them into :class:`Function`s.

        By default, only innermost Iteration objects containing more than
        ``self.thresholds['elemental']`` operations are extracted. One can specify a
        different extraction rule through the lambda function ``kwargs['rule']``,
        which takes as input an iterable of :class:`Iteration`s and returns an
        :class:`Iteration` node.
        """
        noinline = self._compiler_decoration('noinline', c.Comment('noinline?'))
        rule = kwargs.get('rule', lambda tree: tree[-1])

        functions = []
        processed = []
        for node in state.nodes:
            mapper = {}
            for tree in retrieve_iteration_tree(node, mode='superset'):
                if len(tree) <= 1:
                    continue
                root = rule(tree)

                # Has an identical body been encountered already?
                view = root.view
                if view in mapper:
                    mapper[view][1].append(root)
                    continue

                name = "f_%d" % len(functions)

                # Heuristic: create elemental functions only if more than
                # self.thresholds['elemental_functions'] operations are present
                expressions = FindNodes(Expression).visit(root)
                ops = estimate_cost([e.expr for e in expressions])
                if ops < self.thresholds['elemental'] and not root.is_Elementizable:
                    continue

                # Determine the arguments required by the elemental function
                in_scope = [i.dim for i in tree[tree.index(root):]]
                required = FindSymbols(mode='free-symbols').visit(root)
                for i in FindSymbols('symbolics').visit(root):
                    required.extend(flatten(j.free_symbols for j in i.symbolic_shape))
                required = set([as_symbol(i) for i in required if i not in in_scope])

                # Add tensor arguments
                args = []
                seen = {e.output for e in expressions if e.is_scalar}
                for i in FindSymbols('symbolics').visit(root):
                    if i.is_SymbolicFunction:
                        handle = "(%s*) %s" % (c.dtype_to_ctype(i.dtype), i.name)
                    else:
                        handle = "%s_vec" % i.name
                    args.append((handle, i))
                    seen |= {as_symbol(i)}
                # Add scalar arguments
                handle = filter_sorted(required - seen, key=attrgetter('name'))
                args.extend([(i.name, ScalarFunction(name=i.name, dtype=np.int32))
                             for i in handle])

                # Track info to transform the main tree
                call, parameters = zip(*args)
                mapper[view] = (List(header=noinline, body=FunCall(name, call)), [root])

                # Produce the new function
                functions.append(Function(name, root, 'void', parameters, ('static',)))

            # Transform the main tree
            imapper = {}
            for v, keys in mapper.values():
                imapper.update({k: v for k in keys})
            processed.append(Transformer(imapper).visit(node))

        return {'nodes': processed, 'elemental_functions': functions}

    def _compiler_decoration(self, name, default=None):
        key = self.params['compiler'].__class__.__name__
        complang = complang_ALL.get(key, {})
        return complang.get(name, default)

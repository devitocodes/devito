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
        Move :class:`Iteration` sub-trees to separate functions.

        By default, inner iteration trees are moved. To move different types of
        :class:`Iteration`, one can provide a lambda function in ``kwargs['rule']``,
        taking as input an iterable of :class:`Iteration` and returning an iterable
        of :class:`Iteration` (eg, a subset, the whole iteration tree).
        """
        noinline = self._compiler_decoration('noinline', c.Comment('noinline?'))
        rule = kwargs.get('rule', lambda tree: tree[-1:])

        functions = []
        processed = []
        for i, node in enumerate(state.nodes):
            mapper = {}
            for j, tree in enumerate(retrieve_iteration_tree(node)):
                if len(tree) <= 1:
                    continue

                name = "f_%d_%d" % (i, j)

                candidate = rule(tree)
                root = candidate[0]
                expressions = FindNodes(Expression).visit(candidate)

                # Heuristic: create elemental functions only if more than
                # self.thresholds['elemental_functions'] operations are present
                ops = estimate_cost([e.expr for e in expressions])
                if ops < self.thresholds['elemental'] and not root.is_Elementizable:
                    continue

                # Determine the elemental function's arguments ...
                already_in_scope = [k.dim for k in candidate]
                required = [k for k in FindSymbols(mode='free-symbols').visit(candidate)
                            if k not in already_in_scope]
                required += [as_symbol(k) for k in
                             set(flatten(k.free_symbols for k in root.bounds_symbolic))]
                required = set(required)

                args = []
                seen = {e.output for e in expressions if e.is_scalar}
                for d in FindSymbols('symbolics').visit(candidate):
                    # Add a necessary Symbolic object
                    handle = "(float*) %s" if d.is_SymbolicFunction else "%s_vec"
                    args.append((handle % d.name, d))
                    seen |= {as_symbol(d)}
                    # Add necessary information related to Dimensions
                    for k in d.indices:
                        if k.size is not None:
                            continue
                        # Dimension size
                        size = k.symbolic_size
                        if size not in seen:
                            args.append((k.ccode, k))
                            seen |= {size}
                        # Dimension index may be required too
                        if k in required - seen:
                            index_arg = (k.name, ScalarFunction(name=k.name,
                                                                dtype=np.int32))
                            args.append(index_arg)
                            seen |= {k}

                # Add non-temporary scalars to the elemental function's arguments
                handle = filter_sorted(required - seen, key=attrgetter('name'))
                args.extend([(k.name, ScalarFunction(name=k.name, dtype=np.int32))
                             for k in handle])

                # Track info to transform the main tree
                call, parameters = zip(*args)
                mapper[root] = List(header=noinline, body=FunCall(name, call))

                # Produce the new function
                functions.append(Function(name, root, 'void', parameters, ('static',)))

            # Transform the main tree
            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed, 'elemental_functions': functions}

    def _compiler_decoration(self, name, default=None):
        key = self.params['compiler'].__class__.__name__
        complang = complang_ALL.get(key, {})
        return complang.get(name, default)

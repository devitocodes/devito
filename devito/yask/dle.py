from collections import OrderedDict

from devito.dle.backends import AdvancedRewriter, Ompizer
from devito.ir.iet import Expression, FindNodes, Transformer
from devito.logger import yask_warning as warning

from devito.yask.utils import namespace, split_increment

__all__ = ['YaskRewriter']


class YaskOmpizer(Ompizer):

    def _make_parallel_tree(self, root, candidates):
        """
        Return a mapper to parallelize the :class:`Iteration`s within /root/.
        """
        ncollapse = self._ncollapse(root, candidates)
        parallel = self.lang['for'](ncollapse)

        yask_add = namespace['code-grid-add']

        # Introduce the `omp for` pragma
        mapper = OrderedDict()
        if root.is_ParallelAtomic:
            # Turn increments into atomic increments
            subs = {}
            for e in FindNodes(Expression).visit(root):
                if not e.is_Increment:
                    continue
                # Try getting the increment components
                try:
                    target, value, indices = split_increment(e.expr)
                except (AttributeError, ValueError):
                    warning("Found a parallelizable tree, but couldn't ompize it "
                            "because couldn't understand the increment %s" % e.expr)
                    return mapper
                # All good, can atomicize the increment
                subs[e] = e._rebuild(expr=e.expr.func(yask_add, target, (value, indices)))
            handle = Transformer(subs).visit(root)
            mapper[root] = handle._rebuild(pragmas=root.pragmas + (parallel,))
        else:
            mapper[root] = root._rebuild(pragmas=root.pragmas + (parallel,))

        return mapper


class YaskRewriter(AdvancedRewriter):

    _parallelizer = YaskOmpizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_wrapping(state)
        if self.params['openmp'] is True:
            self._parallelize(state)

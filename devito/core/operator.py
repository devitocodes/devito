from __future__ import absolute_import

from collections import OrderedDict

from devito.core.autotuning import autotune
from devito.cgen_utils import printmark
from devito.ir.iet import List, Transformer, filter_iterations, retrieve_iteration_tree
from devito.ir.support import align_accesses
from devito.operator import OperatorRunnable
from devito.tools import flatten

__all__ = ['Operator']


class OperatorCore(OperatorRunnable):

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain
        expressions = [align_accesses(e) for e in expressions]
        return super(OperatorCore, self)._specialize_exprs(expressions)

    def _autotune(self, args):
        """
        Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use.
        """
        if self.dle_flags.get('blocking', False):
            # AT assumes and ordered dict, so let's feed it one
            args = OrderedDict([(p.name, args[p.name]) for p in self.parameters])
            return autotune(self, args, self.dle_args)
        else:
            return args


class OperatorDebug(OperatorCore):
    """
    Decorate the generated code with useful print statements.
    """

    def __init__(self, expressions, **kwargs):
        super(OperatorDebug, self).__init__(expressions, **kwargs)
        self._includes.append('stdio.h')

        # Minimize the trip count of the sequential loops
        iterations = set(flatten(retrieve_iteration_tree(self.body)))
        mapper = {i: i._rebuild(limits=(max(i.offsets) + 2))
                  for i in iterations if i.is_Sequential}
        self.body = Transformer(mapper).visit(self.body)

        # Mark entry/exit points of each non-sequential Iteration tree in the body
        iterations = [filter_iterations(i, lambda i: not i.is_Sequential, 'any')
                      for i in retrieve_iteration_tree(self.body)]
        iterations = [i[0] for i in iterations if i]
        mapper = {t: List(header=printmark('In nest %d' % i), body=t)
                  for i, t in enumerate(iterations)}
        self.body = Transformer(mapper).visit(self.body)


class Operator(object):

    def __new__(cls, *args, **kwargs):
        cls = OperatorDebug if kwargs.pop('debug', False) else OperatorCore
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj

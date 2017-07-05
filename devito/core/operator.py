from __future__ import absolute_import

from collections import OrderedDict

import cgen as c
import numpy as np

from devito.autotuning import autotune
from devito.cgen_utils import blankline, printmark
from devito.dle import filter_iterations, retrieve_iteration_tree
from devito.logger import bar, error, info
from devito.nodes import List
from devito.parameters import configuration
from devito.profiling import create_profile
from devito.visitors import Transformer
from devito.tools import flatten
import devito.operator as operator

__all__ = ['Operator']


class OperatorForeign(operator.Operator):
    """
    A special :class:`Operator` for use outside of Python.
    """

    def arguments(self, *args, **kwargs):
        arguments, _ = super(OperatorForeign, self).arguments(*args, **kwargs)
        return arguments.items()


class OperatorRunnable(operator.Operator):
    """
    A special :class:`Operator` that, besides generation and compilation of
    C code evaluating stencil expressions, can also execute the computation.
    """

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply the stencil kernel to a set of data objects"""
        # Build the arguments list to invoke the kernel function
        arguments, dim_sizes = self.arguments(*args, **kwargs)

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Output summary of performance achieved
        summary = self.profiler.summary(dim_sizes, self.dtype)
        with bar():
            for k, v in summary.items():
                name = '%s<%s>' % (k, ','.join('%d' % i for i in v.itershape))
                info("Section %s with OI=%.2f computed in %.3f s [Perf: %.2f GFlops/s]" %
                     (name, v.oi, v.time, v.gflopss))

        return summary

    def _arg_data(self, argument):
        # Ensure we're dealing or deriving numpy arrays
        data = argument.data
        if not isinstance(data, np.ndarray):
            error('No array data found for argument %s' % argument.name)
        return data

    def _arg_shape(self, argument):
        return argument.data.shape

    def _profile_sections(self, nodes):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        return create_profile(nodes)

    def _extra_arguments(self):
        return OrderedDict([(self.profiler.typename, self.profiler.setup())])

    @property
    def _cparameters(self):
        cparameters = super(OperatorRunnable, self)._cparameters
        cparameters += [c.Pointer(c.Value('struct %s' % self.profiler.typename,
                                          self.profiler.varname))]
        return cparameters

    @property
    def _cglobals(self):
        return [self.profiler.ctype, blankline]


class OperatorCore(OperatorRunnable):

    def _autotune(self, arguments):
        """Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use."""
        if self.dle_flags.get('blocking', False):
            return autotune(self, arguments, self.dle_arguments,
                            mode=configuration['autotuning'])
        else:
            return arguments


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
        # What type of Operator should I create ?
        if kwargs.pop('external', False):
            cls = OperatorForeign
        elif kwargs.pop('debug', False):
            cls = OperatorDebug
        else:
            cls = OperatorCore

        # Trigger instantiation
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj

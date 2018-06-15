from __future__ import absolute_import

from collections import OrderedDict, namedtuple
from functools import reduce
from operator import mul

from ctypes import Structure, byref, c_double
from cgen import Struct, Value

from devito.ir.iet import (ExpressionBundle, TimedList, Section, FindNodes,
                           Transformer)
from devito.ir.support import IntervalGroup
from devito.parameters import configuration
from devito.symbolics import estimate_cost
from devito.tools import flatten

__all__ = ['create_profile']


def create_profile(name):
    """
    Create a new :class:`Profiler`.
    """
    return profiler_registry[configuration['profiling']](name)


class Profiler(object):

    def __init__(self, name):
        self.name = name
        self._sections = OrderedDict()

    def new(self):
        """
        Allocate and return a pointer to a new C-level Struct capable of storing
        all timers inserted by :meth:`instrument`.
        """
        return byref(self.dtype())

    def instrument(self, iet):
        """
        Enrich the Iteration/Expression tree ``iet`` adding nodes for C-level
        performance profiling. In particular, turn all :class:`Section`s within ``iet``
        into :class:`TimedList`s.
        """
        sections = FindNodes(Section).visit(iet)
        for section in sections:
            bundles = FindNodes(ExpressionBundle).visit(section)

            # Total operation count
            ops = sum(i.ops for i in bundles)

            # Operation count at each section iteration
            sops = sum(estimate_cost(i.expr) for i in flatten(b.exprs for b in bundles))

            # Total memory traffic
            mapper = {}
            for i in bundles:
                for k, v in i.traffic.items():
                    mapper.setdefault(k, []).append(v)
            traffic = [IntervalGroup.generate('merge', *i) for i in mapper.values()]
            traffic = sum(i.extent for i in traffic)

            # Each ExpressionBundle lives in its own iteration space
            itershapes = [i.shape for i in bundles]

            # Track how many grid points are written within `section`
            points = []
            for i in bundles:
                writes = {e.write for e in i.exprs
                          if e.is_tensor and e.write.is_TimeFunction}
                points.append(reduce(mul, i.shape)*len(writes))
            points = sum(points)

            self._sections[section] = SectionData(ops, sops, points, traffic, itershapes)

        # Transform the Iteration/Expression tree introducing the C-level timers
        mapper = {i: TimedList(gname=self.name, lname=i.name, body=i.body)
                  for i in sections}
        iet = Transformer(mapper).visit(iet)

        return iet

    def summary(self, arguments, dtype):
        """
        Return a :class:`PerformanceSummary` of the profiled sections.

        :param arguments: A mapper from argument names to run-time values from which
                          the Profiler infers iteration space and execution times
                          of a run.
        :param dtype: The data type of the objects in the profiled sections. Used
                      to compute the operational intensity.
        """
        summary = PerformanceSummary()
        for section, data in self._sections.items():
            # Time to run the section
            time = max(getattr(arguments[self.name]._obj, section.name), 10e-7)

            # Number of FLOPs performed
            ops = data.ops.subs(arguments)

            # Number of grid points computed
            points = data.points.subs(arguments)

            # Compulsory traffic
            traffic = float(data.traffic.subs(arguments)*dtype().itemsize)

            # Runtime itershapes
            itershapes = [tuple(i.subs(arguments) for i in j) for j in data.itershapes]

            # Do not show unexecuted Sections (i.e., because the loop trip count was 0)
            if ops == 0 or traffic == 0:
                continue

            # Derived metrics
            gflops = float(ops)/10**9
            gpoints = float(points)/10**9
            gflopss = gflops/time
            gpointss = gpoints/time
            oi = ops/traffic

            # Keep track of performance achieved
            summary.add(section.name, time, gflopss, gpointss, oi, data.sops, itershapes)

        return summary

    @property
    def dtype(self):
        """
        Return the profiler C type in ctypes format.
        """
        return type(Profiler.__name__, (Structure,),
                    {"_fields_": [(i.name, c_double) for i in self._sections]})

    @property
    def cdef(self):
        """
        Return a :class:`cgen.Struct` representing the profiler data structure in C
        (a ``struct``).
        """
        return Struct(Profiler.__name__,
                      [Value('double', i.name) for i in self._sections])


class AdvisorProfiler(Profiler):

    pass


class PerformanceSummary(OrderedDict):

    """
    A special dictionary to track and quickly access performance data.
    """

    def add(self, key, time, gflopss, gpointss, oi, ops, itershapes):
        self[key] = PerfEntry(time, gflopss, gpointss, oi, ops, itershapes)

    @property
    def gflopss(self):
        return OrderedDict([(k, v.gflopss) for k, v in self.items()])

    @property
    def oi(self):
        return OrderedDict([(k, v.oi) for k, v in self.items()])

    @property
    def timings(self):
        return OrderedDict([(k, v.time) for k, v in self.items()])


SectionData = namedtuple('SectionData', 'ops sops points traffic itershapes')
"""Metadata for a profiled code section."""


PerfEntry = namedtuple('PerfEntry', 'time gflopss gpointss oi ops itershapes')
"""Runtime profiling data for a :class:`Section`."""


# Set up profiling levels
profiler_registry = {
    'basic': Profiler,
    'advisor': AdvisorProfiler
}
configuration.add('profiling', 'basic', list(profiler_registry))

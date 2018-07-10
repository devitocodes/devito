from __future__ import absolute_import

from collections import OrderedDict, namedtuple
from functools import reduce
from operator import mul
from pathlib import Path
import os

from ctypes import Structure, byref, c_double
from cgen import Struct, Value

from devito.ir.iet import (Call, ExpressionBundle, List, TimedList, Section,
                           FindNodes, Transformer)
from devito.ir.support import IntervalGroup
from devito.logger import warning
from devito.parameters import configuration
from devito.symbolics import estimate_cost
from devito.tools import flatten
from devito.types import Object

__all__ = ['Timer', 'create_profile']


class Profiler(object):

    _default_includes = []
    _default_libs = []
    _ext_calls = []

    def __init__(self, name):
        self.name = name
        self._sections = OrderedDict()

        self.initialized = True

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
        mapper = {i: TimedList(gname=self.name, lname=i.name, body=i) for i in sections}
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
            # Note: some casts to `float` are simply to turn `sympy.Float` into `float`
            gflops = float(ops)/10**9
            gpoints = float(points)/10**9
            gflopss = gflops/time
            gpointss = gpoints/time
            oi = float(ops/traffic)

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

    """Rely on Intel Advisor ``v >= 2018`` for performance profiling."""

    _api_resume = '__itt_resume'
    _api_pause = '__itt_pause'

    _default_includes = ['ittnotify.h']
    _default_libs = ['ittnotify']
    _ext_calls = [_api_resume, _api_pause]

    def __init__(self, name):

        self.path = locate_intel_advisor()
        if self.path is None:
            self.initialized = False
        else:
            super(AdvisorProfiler, self).__init__(name)
            # Make sure future compilations will get the proper header and
            # shared object files
            compiler = configuration['compiler']
            compiler.add_include_dirs(self.path.joinpath('include').as_posix())
            compiler.add_libraries(self._default_libs)
            libdir = self.path.joinpath('lib64').as_posix()
            compiler.add_library_dirs(libdir)
            compiler.add_ldflags('-Wl,-rpath,%s' % libdir)

    def instrument(self, iet):
        sections = FindNodes(Section).visit(iet)

        # Transform the Iteration/Expression tree introducing Advisor calls that
        # resume and stop data collection
        mapper = {i: List(body=[Call(self._api_resume), i, Call(self._api_pause)])
                  for i in sections}
        iet = Transformer(mapper).visit(iet)

        return iet


class Timer(Object):

    def __init__(self, profiler):
        self.profiler = profiler

    @property
    def name(self):
        return self.profiler.name

    @property
    def dtype(self):
        return self.profiler.dtype

    @property
    def value(self):
        return self.profiler.new


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


def create_profile(name):
    """
    Create a new :class:`Profiler`.
    """
    level = configuration['profiling']
    profiler = profiler_registry[level](name)
    if profiler.initialized:
        return profiler
    else:
        warning("Couldn't set up `%s` profiler; reverting to `basic`" % level)
        profiler = profiler_registry['basic'](name)
        # We expect the `basic` profiler to always initialize successfully
        assert profiler.initialized
        return profiler


# Set up profiling levels
profiler_registry = {
    'basic': Profiler,
    'advisor': AdvisorProfiler
}
configuration.add('profiling', 'basic', list(profiler_registry))


def locate_intel_advisor():
    try:
        path = Path(os.environ['ADVISOR_HOME'])
        # Little hack: assuming a 64bit system
        if path.joinpath('bin64').joinpath('advixe-cl').is_file():
            return path
        else:
            warning("Requested `advisor` profiler, but couldn't locate executable")
            return None
    except KeyError:
        warning("Requested `advisor` profiler, but ADVISOR_HOME isn't set")
        return None

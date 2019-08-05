from collections import OrderedDict, namedtuple
from ctypes import c_double
from functools import reduce
from operator import mul
from pathlib import Path
import os

from cached_property import cached_property

from devito.ir.iet import (Call, ExpressionBundle, List, TimedList, Section,
                           FindNodes, Transformer)
from devito.ir.support import IntervalGroup
from devito.logger import warning
from devito.parameters import configuration
from devito.symbolics import estimate_cost
from devito.tools import flatten
from devito.types import CompositeObject

__all__ = ['Timer', 'create_profile']


class Profiler(object):

    _default_includes = []
    _default_libs = []
    _ext_calls = []

    def __init__(self, name):
        self.name = name
        self._sections = OrderedDict()

        self.initialized = True

    def instrument(self, iet):
        """
        Enrich the Iteration/Expression tree ``iet`` adding nodes for C-level
        performance profiling. In particular, turn all Sections within ``iet``
        into TimedLists.
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
            traffic = [IntervalGroup.generate('union', *i) for i in mapper.values()]
            traffic = sum(i.size for i in traffic)

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
        mapper = {i: TimedList(timer=self.timer, lname=i.name, body=i) for i in sections}
        iet = Transformer(mapper).visit(iet)

        return iet

    def summary(self, arguments, dtype):
        """
        Return a PerformanceSummary of the profiled sections. See
        summary under the class AdvancedProfiler below for further details.
        """
        summary = PerformanceSummary()
        for section, data in self._sections.items():
            # Time to run the section
            time = max(getattr(arguments[self.name]._obj, section.name), 10e-7)

            # In basic mode only return runtime. Other arguments are filled with
            # dummy values.
            summary.add(section.name, time, float(), float(), float(), int(), [])

        return summary

    @cached_property
    def timer(self):
        return Timer(self.name, [i.name for i in self._sections])


class AdvancedProfiler(Profiler):

    # Override basic summary so that arguments other than runtime are computed.
    def summary(self, arguments, dtype):
        """
        Return a PerformanceSummary of the profiled sections.

        Parameters
        ----------
        arguments : dict
            A mapper from argument names to run-time values from which the Profiler
            infers iteration space and execution times of a run.
        dtype : data-type, optional
            The data type of the objects in the profiled sections. Used to compute
            the operational intensity.
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


class AdvisorProfiler(AdvancedProfiler):

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


class Timer(CompositeObject):

    def __init__(self, name, sections):
        super(Timer, self).__init__(name, 'profiler', [(i, c_double) for i in sections])

    def reset(self):
        for i in self.fields:
            setattr(self.value._obj, i, 0.0)
        return self.value

    @property
    def total(self):
        return sum(getattr(self.value._obj, i) for i in self.fields)

    @property
    def sections(self):
        return self.fields

    # Pickling support
    _pickle_args = ['name', 'sections']


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
"""Runtime profiling data for a Section."""


def create_profile(name):
    """Create a new Profiler."""
    if configuration['log-level'] == 'DEBUG':
        # Enforce performance profiling in DEBUG mode
        level = 'advanced'
    else:
        level = configuration['profiling']
    profiler = profiler_registry[level](name)
    if profiler.initialized:
        return profiler
    else:
        warning("Couldn't set up `%s` profiler; reverting to `advanced`" % level)
        profiler = profiler_registry['basic'](name)
        # We expect the `advanced` profiler to always initialize successfully
        assert profiler.initialized
        return profiler


profiler_registry = {
    'basic': Profiler,
    'advanced': AdvancedProfiler,
    'advisor': AdvisorProfiler
}
"""Profiling levels."""


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

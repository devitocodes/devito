from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from ctypes import c_double
from functools import reduce
from operator import mul
from pathlib import Path
from time import time as seq_time
import os

from cached_property import cached_property

from devito.ir.iet import (Call, ExpressionBundle, List, TimedList, Section,
                           FindNodes, Transformer)
from devito.ir.support import IntervalGroup
from devito.logger import warning
from devito.mpi import MPI
from devito.parameters import configuration
from devito.types import CompositeObject

__all__ = ['Timer', 'create_profile']


SectionData = namedtuple('SectionData', 'ops sops points traffic itermaps')
PerfKey = namedtuple('PerfKey', 'name rank')
PerfInput = namedtuple('PerfInput', 'time ops points traffic sops itershapes')
PerfEntry = namedtuple('PerfEntry', 'time gflopss gpointss oi ops itershapes')


class Profiler(object):

    _default_includes = []
    _default_libs = []
    _ext_calls = []

    """Metadata for a profiled code section."""

    def __init__(self, name):
        self.name = name

        # Operation reductions observed in sections
        self._ops = []

        # C-level code sections
        self._sections = OrderedDict()

        # Python-level timers
        self.py_timers = OrderedDict()

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
            ops = sum(i.ops*i.ispace.size for i in bundles)

            # Operation count at each section iteration
            sops = sum(i.ops for i in bundles)

            # Total memory traffic
            mapper = {}
            for i in bundles:
                for k, v in i.traffic.items():
                    mapper.setdefault(k, []).append(v)
            traffic = 0
            for i in mapper.values():
                try:
                    traffic += IntervalGroup.generate('union', *i).size
                except ValueError:
                    # Over different iteration spaces
                    traffic += sum(j.size for j in i)

            # Each ExpressionBundle lives in its own iteration space
            itermaps = [i.ispace.dimension_map for i in bundles]

            # Track how many grid points are written within `section`
            points = []
            for i in bundles:
                writes = {e.write for e in i.exprs
                          if e.is_tensor and e.write.is_TimeFunction}
                points.append(i.size*len(writes))
            points = sum(points)

            self._sections[section] = SectionData(ops, sops, points, traffic, itermaps)

        # Transform the Iteration/Expression tree introducing the C-level timers
        mapper = {i: TimedList(timer=self.timer, lname=i.name, body=i) for i in sections}
        iet = Transformer(mapper).visit(iet)

        return iet

    @contextmanager
    def timer_on(self, name, comm=None):
        """
        Measure the execution time of a Python-level code region.

        Parameters
        ----------
        name : str
            A representative string for the timed region.
        comm : MPI communicator, optional
            If provided, the global execution time is derived by a single MPI
            rank, with timers started and stopped right after an MPI barrier.
        """
        if comm and comm is not MPI.COMM_NULL:
            comm.Barrier()
            tic = MPI.Wtime()
            yield
            comm.Barrier()
            toc = MPI.Wtime()
        else:
            tic = seq_time()
            yield
            toc = seq_time()
        self.py_timers[name] = toc - tic

    def record_ops_variation(self, initial, final):
        """
        Record the variation in operation count experienced by a section due to
        a flop-reducing transformation.
        """
        self._ops.append((initial, final))

    def summary(self, args, dtype, reduce_over=None):
        """
        Return a PerformanceSummary of the profiled sections.

        Parameters
        ----------
        args : dict
            A mapper from argument names to run-time values from which the Profiler
            infers iteration space and execution times of a run.
        dtype : data-type
            The data type of the objects in the profiled sections. Used to compute
            the operational intensity.
        """
        comm = args.comm

        summary = PerformanceSummary()
        for section, data in self._sections.items():
            name = section.name

            # Time to run the section
            time = max(getattr(args[self.name]._obj, name), 10e-7)

            # Add performance data
            if comm is not MPI.COMM_NULL:
                # With MPI enabled, we add one entry per section per rank
                times = comm.allgather(time)
                assert comm.size == len(times)
                for rank in range(comm.size):
                    summary.add(name, rank, times[rank])
            else:
                summary.add(name, None, time)

        return summary

    @cached_property
    def timer(self):
        return Timer(self.name, [i.name for i in self._sections])


class AdvancedProfiler(Profiler):

    # Override basic summary so that arguments other than runtime are computed.
    def summary(self, args, dtype, reduce_over=None):
        grid = args.grid
        comm = args.comm

        summary = PerformanceSummary()
        for section, data in self._sections.items():
            name = section.name

            # Time to run the section
            time = max(getattr(args[self.name]._obj, name), 10e-7)

            # Number of FLOPs performed
            ops = int(data.ops.subs(args))

            # Number of grid points computed
            points = int(data.points.subs(args))

            # Compulsory traffic
            traffic = float(data.traffic.subs(args)*dtype().itemsize)

            # Runtime itermaps/itershapes
            itermaps = [OrderedDict([(k, int(v.subs(args))) for k, v in i.items()])
                        for i in data.itermaps]
            itershapes = tuple(tuple(i.values()) for i in itermaps)

            # Add local performance data
            if comm is not MPI.COMM_NULL:
                # With MPI enabled, we add one entry per section per rank
                times = comm.allgather(time)
                assert comm.size == len(times)
                opss = comm.allgather(ops)
                pointss = comm.allgather(points)
                traffics = comm.allgather(traffic)
                sops = [data.sops]*comm.size
                itershapess = comm.allgather(itershapes)
                items = list(zip(times, opss, pointss, traffics, sops, itershapess))
                for rank in range(comm.size):
                    summary.add(name, rank, *items[rank])
            else:
                summary.add(name, None, time, ops, points, traffic, data.sops, itershapes)

        # Add global performance data
        if reduce_over is not None:
            # Vanilla metrics
            if comm is not MPI.COMM_NULL:
                summary.add_glb_vanilla(self.py_timers[reduce_over])

            # Typical finite difference benchmark metrics
            if grid is not None:
                dimensions = (grid.time_dim,) + grid.dimensions
                if all(d.max_name in args for d in dimensions):
                    max_t = args[grid.time_dim.max_name] or 0
                    min_t = args[grid.time_dim.min_name] or 0
                    nt = max_t - min_t + 1
                    points = reduce(mul, (nt,) + grid.shape)
                    summary.add_glb_fdlike(points, self.py_timers[reduce_over])

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

    def __init__(self, *args, **kwargs):
        super(PerformanceSummary, self).__init__(*args, **kwargs)
        self.input = OrderedDict()
        self.globals = {}

    def add(self, name, rank, time,
            ops=None, points=None, traffic=None, sops=None, itershapes=None):
        """
        Add performance data for a given code section. With MPI enabled, the
        performance data is local, that is "per-rank".
        """
        # Do not show unexecuted Sections (i.e., loop trip count was 0)
        if ops == 0 or traffic == 0:
            return

        k = PerfKey(name, rank)

        if ops is None:
            self[k] = PerfEntry(time, 0.0, 0.0, 0.0, 0, [])
        else:
            gflops = float(ops)/10**9
            gpoints = float(points)/10**9
            gflopss = gflops/time
            gpointss = gpoints/time
            oi = float(ops/traffic)

            self[k] = PerfEntry(time, gflopss, gpointss, oi, sops, itershapes)

        self.input[k] = PerfInput(time, ops, points, traffic, sops, itershapes)

    def add_glb_vanilla(self, time):
        """
        Reduce the following performance data:

            * ops
            * points
            * traffic

        over a global "wrapping" timer.
        """
        if not self.input:
            return

        ops = sum(v.ops for v in self.input.values())
        points = sum(v.points for v in self.input.values())
        traffic = sum(v.traffic for v in self.input.values())

        gflops = float(ops)/10**9
        gpoints = float(points)/10**9
        gflopss = gflops/time
        gpointss = gpoints/time
        oi = float(ops/traffic)

        self.globals['vanilla'] = PerfEntry(time, gflopss, gpointss, oi, None, None)

    def add_glb_fdlike(self, points, time):
        """
        Add "finite-difference-like" performance metrics, that is GPoints/s and
        GFlops/s as if the code looked like a trivial n-D jacobi update

            .. code-block:: c

              for t = t_m to t_M
                for x = 0 to x_size
                  for y = 0 to y_size
                    u[t+1, x, y] = f(...)
        """
        gpoints = float(points)/10**9
        gpointss = gpoints/time

        self.globals['fdlike'] = PerfEntry(time, None, gpointss, None, None, None)

    @property
    def gflopss(self):
        return OrderedDict([(k, v.gflopss) for k, v in self.items()])

    @property
    def oi(self):
        return OrderedDict([(k, v.oi) for k, v in self.items()])

    @property
    def timings(self):
        return OrderedDict([(k, v.time) for k, v in self.items()])


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

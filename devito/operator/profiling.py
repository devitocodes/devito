from collections import OrderedDict, defaultdict, namedtuple
from contextlib import contextmanager
from functools import reduce
from operator import mul
from time import time as seq_time

import cgen as c
import numpy as np
from sympy import S

from devito.arch import get_advisor_path
from devito.ir.iet import (ExpressionBundle, List, TimedList, Section,
                           Iteration, FindNodes, Transformer)
from devito.ir.support import IntervalGroup
from devito.logger import warning
from devito.mpi import MPI
from devito.parameters import configuration
from devito.symbolics import subs_op_args
from devito.tools import DefaultOrderedDict, flatten

__all__ = ['create_profile']


SectionData = namedtuple('SectionData', 'ops sops points traffic itermaps')
PerfKey = namedtuple('PerfKey', 'name rank')
PerfInput = namedtuple('PerfInput', 'time ops points traffic sops itershapes')
PerfEntry = namedtuple('PerfEntry', 'time gflopss gpointss oi ops itershapes')


class Profiler:

    _default_includes = []
    _default_libs = []
    _ext_calls = []

    _include_dirs = []
    _lib_dirs = []

    _supports_async_sections = False

    _verbosity = 0

    _attempted_init = False

    def __init__(self, name):
        self.name = name

        # Operation reductions observed in sections
        self._ops = []

        # C-level code sections
        self._sections = OrderedDict()
        self._subsections = OrderedDict()

        # Python-level timers
        self.py_timers = OrderedDict()

        self._attempted_init = True

    def analyze(self, iet):
        """
        Analyze the Sections in the given IET. This populates `self._sections`.
        """
        sections = FindNodes(Section).visit(iet)
        for s in sections:
            if s.name in self._sections:
                continue

            bundles = FindNodes(ExpressionBundle).visit(s)

            # Total operation count
            ops = sum(i.ops*i.ispace.size for i in bundles)

            # Operation count at each section iteration
            # NOTE: for practical reasons, it makes much more sense to "flatten"
            # the StencilDimensions, because one expects to see the number of
            # operations per time- or space-Dimension
            sops = sum(i.ops*max(i.ispace.project(lambda d: d.is_Stencil).size, 1)
                       for i in bundles)

            # Total memory traffic
            mapper = {}
            for i in bundles:
                for k, v in i.traffic.items():
                    mapper.setdefault(k, []).append(v)
            traffic = 0
            for i in mapper.values():
                try:
                    traffic += IntervalGroup.generate('union', *i).size
                except (ValueError, TypeError):
                    # Over different iteration spaces
                    traffic += sum(j.size for j in i)

            # Each ExpressionBundle lives in its own iteration space
            itermaps = [i.ispace.dimension_map for i in bundles if i.ops != 0]

            # Track how many grid points are written within `s`
            points = set()
            for i in bundles:
                if any(e.write.is_TimeFunction for e in i.exprs):
                    # The `ispace` is zero-ed because we don't care if a point
                    # is touched redundantly
                    points.add(i.ispace.zero().size)
            points = sum(points, S.Zero)

            self._sections[s.name] = SectionData(ops, sops, points, traffic, itermaps)

    def track_subsection(self, sname, name):
        v = self._subsections.setdefault(sname, OrderedDict())
        v[name] = SectionData(S.Zero, S.Zero, S.Zero, S.Zero, [])

    def group_as_subsections(self, sname, sections):
        ops = sum(self._sections[i].ops for i in sections)
        points = sum(self._sections[i].points for i in sections)
        traffic = sum(self._sections[i].traffic for i in sections)
        sectiondata = SectionData(ops, S.Zero, points, traffic, [])

        v = self._subsections.setdefault(sname, OrderedDict())
        v.update({i: self._sections[i] for i in sections})

        new_sections = OrderedDict()
        for k, v in self._sections.items():
            try:
                if sections.index(k) == len(sections) - 1:
                    new_sections[sname] = sectiondata
            except ValueError:
                new_sections[k] = v
        self._sections.clear()
        self._sections.update(new_sections)

    def instrument(self, iet, timer):
        """
        Instrument the given IET for C-level performance profiling.
        """
        sections = FindNodes(Section).visit(iet)
        if sections:
            mapper = {}
            for i in sections:
                n = i.name
                assert n in timer.fields
                mapper[i] = i._rebuild(body=TimedList(timer=timer, lname=n, body=i.body))
            return Transformer(mapper, nested=True).visit(iet)
        else:
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

    @property
    def all_sections(self):
        return list(self._sections) + flatten(self._subsections.values())

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
        for name, data in self._sections.items():
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


class ProfilerVerbose1(Profiler):
    _verbosity = 1


class ProfilerVerbose2(Profiler):
    _verbosity = 2


class AdvancedProfiler(Profiler):

    _supports_async_sections = True

    def _evaluate_section(self, name, data, args, dtype):
        # Time to run the section
        time = max(getattr(args[self.name]._obj, name), 10e-7)

        # Number of FLOPs performed
        try:
            ops = int(subs_op_args(data.ops, args))
        except (AttributeError, TypeError):
            # E.g., a section comprising just function calls, or at least
            # a sequence of unrecognized or non-conventional expr statements
            ops = np.nan

        try:
            # Number of grid points computed
            points = int(subs_op_args(data.points, args))

            # Compulsory traffic
            traffic = float(subs_op_args(data.traffic, args)*dtype().itemsize)
        except (AttributeError, TypeError):
            # E.g., the section has a dynamic loop size
            points = np.nan

            traffic = np.nan

        # Nmber of FLOPs performed at each iteration
        sops = data.sops

        # Runtime itermaps/itershapes
        try:
            itermaps = [OrderedDict([(k, int(subs_op_args(v, args)))
                                     for k, v in i.items()])
                        for i in data.itermaps]
            itershapes = tuple(tuple(i.values()) for i in itermaps)
        except TypeError:
            # E.g., a section comprising just function calls, or at least
            # a sequence of unrecognized or non-conventional expr statements
            itershapes = ()

        return time, ops, points, traffic, sops, itershapes

    def _allgather_from_comm(self, comm, time, ops, points, traffic, sops, itershapes):
        times = comm.allgather(time)
        assert comm.size == len(times)

        opss = comm.allgather(ops)
        pointss = comm.allgather(points)
        traffics = comm.allgather(traffic)
        sops = [sops]*comm.size
        itershapess = comm.allgather(itershapes)

        return list(zip(times, opss, pointss, traffics, sops, itershapess))

    # Override basic summary so that arguments other than runtime are computed.
    def summary(self, args, dtype, reduce_over=None):
        grid = args.grid
        comm = args.comm

        # Produce sections summary
        summary = PerformanceSummary()
        for name, data in self._sections.items():
            items = self._evaluate_section(name, data, args, dtype)

            # Add local performance data
            if comm is not MPI.COMM_NULL:
                # With MPI enabled, we add one entry per section per rank
                items = self._allgather_from_comm(comm, *items)
                for rank in range(comm.size):
                    summary.add(name, rank, *items[rank])
            else:
                summary.add(name, None, *items)

        # Enrich summary with subsections data
        for sname, v in self._subsections.items():
            for name, data in v.items():
                items = self._evaluate_section(name, data, args, dtype)

                # Add local performance data
                if comm is not MPI.COMM_NULL:
                    # With MPI enabled, we add one entry per section per rank
                    items = self._allgather_from_comm(comm, *items)
                    for rank in range(comm.size):
                        summary.add_subsection(sname, name, rank, *items[rank])
                else:
                    summary.add_subsection(sname, name, None, *items)

        # Add global performance data
        if reduce_over is not None:
            # Vanilla metrics
            summary.add_glb_vanilla('vanilla', reduce_over)

            # Same as above but without setup overheads (e.g., host-device
            # data transfers)
            mapper = defaultdict(list)
            for (name, rank), v in summary.items():
                mapper[name].append(v.time)
            reduce_over_nosetup = sum(max(i) for i in mapper.values())
            if reduce_over_nosetup == 0:
                reduce_over_nosetup = reduce_over
            summary.add_glb_vanilla('vanilla-nosetup', reduce_over_nosetup)

            # Typical finite difference benchmark metrics
            if grid is not None:
                dimensions = (grid.time_dim,) + grid.dimensions
                if all(d.max_name in args for d in dimensions):
                    max_t = args[grid.time_dim.max_name] or 0
                    min_t = args[grid.time_dim.min_name] or 0
                    nt = max_t - min_t + 1
                    points = reduce(mul, (nt,) + grid.shape)
                    summary.add_glb_fdlike('fdlike', points, reduce_over)

                    # Same as above but without setup overheads (e.g., host-device
                    # data transfers)
                    summary.add_glb_fdlike('fdlike-nosetup', points, reduce_over_nosetup)

        return summary


class AdvancedProfilerVerbose(AdvancedProfiler):
    pass


class AdvancedProfilerVerbose1(AdvancedProfilerVerbose):
    _verbosity = 1


class AdvancedProfilerVerbose2(AdvancedProfilerVerbose):
    _verbosity = 2


class AdvisorProfiler(AdvancedProfiler):

    """
    Rely on Intel Advisor 2025.1 for performance profiling.
    """

    _api_resume = '__itt_resume'
    _api_pause = '__itt_pause'

    _default_includes = ['ittnotify.h']
    _default_libs = ['ittnotify']
    _ext_calls = [_api_resume, _api_pause]

    def __init__(self, name):
        if self._attempted_init:
            return

        super().__init__(name)

        path = get_advisor_path()
        if path:
            self._include_dirs.append(path.joinpath('include').as_posix())
            self._lib_dirs.append(path.joinpath('lib64').as_posix())
            self._attempted_init = True
        else:
            self._attempted_init = False

    def analyze(self, iet):
        """
        A no-op, as the Advisor profiler does not need to analyze the IET.
        """
        return

    def instrument(self, iet, timer):
        # Look for the presence of a time loop within the IET of the Operator
        mapper = {}
        for i in FindNodes(Iteration).visit(iet):
            if i.dim.is_Time:
                # The calls to Advisor's Collection Control API are only for Operators
                # with a time loop
                mapper[i] = List(header=c.Statement(f'{self._api_resume}()'),
                                 body=i,
                                 footer=c.Statement(f'{self._api_pause}()'))
                return Transformer(mapper).visit(iet)

        # Return the IET intact if no time loop is found
        return iet


class PerformanceSummary(OrderedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsections = DefaultOrderedDict(lambda: OrderedDict())
        self.input = OrderedDict()
        self.globals = {}

    def add(self, name, rank, time,
            ops=None, points=None, traffic=None, sops=None, itershapes=None):
        """
        Add performance data for a given code section. With MPI enabled, the
        performance data is local, that is "per-rank".
        """
        # Do not show unexecuted Sections (i.e., loop trip count was 0)
        if traffic == 0:
            return

        k = PerfKey(name, rank)

        if not ops or any(not np.isfinite(i) for i in [ops, points, traffic]):
            self[k] = PerfEntry(time, 0.0, 0.0, 0.0, 0, [])
        else:
            gflops = float(ops)/10**9
            gpoints = float(points)/10**9
            gflopss = gflops/time
            gpointss = gpoints/time
            oi = float(ops/traffic)

            self[k] = PerfEntry(time, gflopss, gpointss, oi, sops, itershapes)

        self.input[k] = PerfInput(time, ops, points, traffic, sops, itershapes)

    def add_subsection(self, sname, name, rank, time, *args):
        k0 = PerfKey(sname, rank)
        assert k0 in self

        self.subsections[sname][name] = PerfEntry(time, None, None, None, None, [])

    def add_glb_vanilla(self, key, time):
        """
        Reduce the following performance data:

            * ops
            * traffic

        over a given global timing.
        """
        if not self.input:
            return

        ops = sum(v.ops for v in self.input.values() if not np.isnan(v.ops))
        traffic = sum(v.traffic for v in self.input.values())

        gflops = float(ops)/10**9
        gflopss = gflops/time

        if np.isnan(traffic) or traffic == 0:
            oi = None
        else:
            oi = float(ops/traffic)

        self.globals[key] = PerfEntry(time, gflopss, None, oi, None, None)

    def add_glb_fdlike(self, key, points, time):
        """
        Add the typical GPoints/s finite-difference metric.
        """
        if np.isnan(points):
            return

        gpoints = float(points)/10**9
        gpointss = gpoints/time

        self.globals[key] = PerfEntry(time, None, gpointss, None, None, None)

    @property
    def globals_all(self):
        v0 = self.globals['vanilla']
        v1 = self.globals['fdlike']
        return PerfEntry(v0.time, v0.gflopss, v1.gpointss, v0.oi, None, None)

    @property
    def globals_nosetup_all(self):
        v0 = self.globals['vanilla-nosetup']
        v1 = self.globals['fdlike-nosetup']
        return PerfEntry(v0.time, v0.gflopss, v1.gpointss, v0.oi, None, None)

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
    if configuration['log-level'] in ['DEBUG', 'PERF'] and \
       configuration['profiling'] == 'basic':
        # Enforce performance profiling in DEBUG mode
        level = 'advanced'
    else:
        level = configuration['profiling']
    profiler = profiler_registry[level](name)

    if profiler._attempted_init:
        return profiler
    else:
        warning(f"Couldn't set up `{level}` profiler; reverting to 'advanced'")
        profiler = profiler_registry['advanced'](name)
        # We expect the `advanced` profiler to always initialize successfully
        assert profiler._attempted_init
        return profiler


profiler_registry = {
    'basic': Profiler,
    'basic1': ProfilerVerbose1,
    'basic2': ProfilerVerbose2,
    'advanced': AdvancedProfiler,
    'advanced1': AdvancedProfilerVerbose1,
    'advanced2': AdvancedProfilerVerbose2,
    'advisor': AdvisorProfiler
}
"""Profiling levels."""

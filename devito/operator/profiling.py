from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from functools import reduce
from operator import mul
from pathlib import Path
from subprocess import DEVNULL, PIPE, run
from time import time as seq_time
import os

import cgen as c

from devito.ir.iet import (ExpressionBundle, List, TimedList, Section,
                           Iteration, FindNodes, Transformer)
from devito.ir.support import IntervalGroup
from devito.logger import warning, error
from devito.mpi import MPI
from devito.parameters import configuration
from devito.symbolics import subs_op_args

__all__ = ['create_profile']


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

            # Track how many grid points are written within `s`
            points = []
            for i in bundles:
                writes = {e.write for e in i.exprs
                          if e.is_tensor and e.write.is_TimeFunction}
                points.append(i.size*len(writes))
            points = sum(points)

            self._sections[s.name] = SectionData(ops, sops, points, traffic, itermaps)

    def instrument(self, iet, timer):
        """
        Instrument the given IET for C-level performance profiling.
        """
        sections = FindNodes(Section).visit(iet)
        if sections:
            mapper = {}
            for i in sections:
                assert i.name in timer.fields
                mapper[i] = TimedList(timer=timer, lname=i.name, body=i)
            return Transformer(mapper).visit(iet)
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


class AdvancedProfiler(Profiler):

    # Override basic summary so that arguments other than runtime are computed.
    def summary(self, args, dtype, reduce_over=None):
        grid = args.grid
        comm = args.comm

        summary = PerformanceSummary()
        for name, data in self._sections.items():
            # Time to run the section
            time = max(getattr(args[self.name]._obj, name), 10e-7)

            # Number of FLOPs performed
            ops = int(subs_op_args(data.ops, args))

            # Number of grid points computed
            points = int(subs_op_args(data.points, args))

            # Compulsory traffic
            traffic = float(subs_op_args(data.traffic, args)*dtype().itemsize)

            # Runtime itermaps/itershapes
            itermaps = [OrderedDict([(k, int(subs_op_args(v, args)))
                                     for k, v in i.items()])
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

    """
    Rely on Intel Advisor ``v >= 2020`` for performance profiling.
    Tested versions of Intel Advisor:
    - As contained in Intel Parallel Studio 2020 v 2020 Update 2
    - As contained in Intel oneAPI 2021 beta08
    """

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

    def analyze(self, iet):
        return

    def instrument(self, iet):
        # Look for the presence of a time loop within the IET of the Operator
        found = False
        for node in FindNodes(Iteration).visit(iet):
            if node.dim.is_Time:
                found = True
                break

        if found:
            # The calls to Advisor's Collection Control API are only for Operators with
            # a time loop
            return List(header=c.Statement('%s()' % self._api_resume),
                        body=iet,
                        footer=c.Statement('%s()' % self._api_pause))

        return iet


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

        if self.input:
            traffic = sum(v.traffic for v in self.input.values())
            ops = sum(v.ops for v in self.input.values())

            gflops = float(ops)/10**9
            gflopss = gflops/time
            oi = float(ops/traffic)
        else:
            gflopss = None
            oi = None

        self.globals['fdlike'] = PerfEntry(time, gflopss, gpointss, oi, None, None)

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
    """
    Detect if Intel Advisor is installed on the machine and return
    its location if it is.

    """
    path = None

    try:
        # Check if the directory to Intel Advisor is specified
        path = Path(os.environ['DEVITO_ADVISOR_DIR'])
    except KeyError:
        # Otherwise, 'sniff' the location of Advisor's directory
        error_msg = 'Intel Advisor cannot be found on your system, consider if you'\
                    ' have sourced its environment variables correctly. Information can'\
                    ' be found at https://software.intel.com/content/www/us/en/develop/'\
                    'documentation/advisor-user-guide/top/launch-the-intel-advisor/'\
                    'intel-advisor-cli/setting-and-using-intel-advisor-environment'\
                    '-variables.html'
        try:
            res = run(["advixe-cl", "--version"], stdout=PIPE, stderr=DEVNULL)
            ver = res.stdout.decode("utf-8")
            if not ver:
                error(error_msg)
                return None
        except (UnicodeDecodeError, FileNotFoundError):
            error(error_msg)
            return None

        env_path = os.environ["PATH"]
        env_path_dirs = env_path.split(":")

        for env_path_dir in env_path_dirs:
            # intel/advisor is the advisor directory for Intel Parallel Studio,
            # intel/oneapi/advisor is the directory for Intel oneAPI
            if "intel/advisor" in env_path_dir or "intel/oneapi/advisor" in env_path_dir:
                path = Path(env_path_dir)
                if path.name.startswith('bin'):
                    path = path.parent

        if not path:
            error(error_msg)
            return None

    if path.joinpath('bin64').joinpath('advixe-cl').is_file():
        return path
    else:
        warning("Requested `advisor` profiler, but couldn't locate executable"
                "in advisor directory")
        return None

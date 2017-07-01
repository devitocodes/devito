from __future__ import absolute_import

import operator
from collections import OrderedDict, namedtuple
from functools import reduce

from ctypes import Structure, byref, c_double
from cgen import Struct, Value

from devito.dse import estimate_cost, estimate_memory
from devito.nodes import Expression, TimedList
from devito.visitors import IsPerfectIteration, FindSections, FindNodes, Transformer

__all__ = ['Profile', 'create_profile']


def create_profile(node):
    """
    Create a :class:`Profiler` for the Iteration/Expression tree ``node``.
    The following code sections are profiled: ::

        * The whole ``node``;
        * A sequence of perfectly nested loops that have common :class:`Iteration`
          dimensions, but possibly different extent. For example: ::

            for x = 0 to N
              ..
            for x = 1 to N-1
              ..

          Both Iterations have dimension ``x``, and will be profiled as a single
          section, though their extent is different.
        * Any perfectly nested loops.
    """
    profiler = Profiler()

    # Group by root Iteration
    mapper = OrderedDict()
    for itspace in FindSections().visit(node):
        mapper.setdefault(itspace[0], []).append(itspace)

    # Group sections if their iteration spaces overlap
    key = lambda itspace: set([i.dim for i in itspace])
    found = []
    for v in mapper.values():
        queue = list(v)
        handle = []
        while queue:
            item = queue.pop(0)
            if not handle or key(item) == key(handle[0]):
                handle.append(item)
            else:
                # Found a timing section
                found.append(tuple(handle))
                handle = [item]
        if handle:
            found.append(tuple(handle))

    # Create and track C-level timers
    mapper = OrderedDict()
    for i, group in enumerate(found):
        name = 'section_%d' % i
        section, remainder = group[0], group[1:]

        index = len(section) > 1 and not IsPerfectIteration().visit(section[0])
        root = section[index]

        # Prepare to transform the Iteration/Expression tree
        body = tuple(j[index] for j in group)
        mapper[root] = TimedList(gname=profiler.varname, lname=name, body=body)
        for j in remainder:
            mapper[j[index]] = None

        # Estimate computational properties of the profiled section
        expressions = FindNodes(Expression).visit(body)
        ops = estimate_cost([e.expr for e in expressions])
        memory = estimate_memory([e.expr for e in expressions])

        # Keep track of the new profiled section
        profiler.add(name, section, ops, memory)

    # Transform the Iteration/Expression tree introducing the C-level timers
    processed = Transformer(mapper).visit(node)

    return processed, profiler


class Profiler(object):

    """
    A Profiler is used to manage profiling information for Devito generated C code.
    """

    varname = "timings"
    typename = "profiler"

    def __init__(self):
        # To be populated as new sections are tracked
        self._sections = OrderedDict()
        self._C_timings = None

    def add(self, name, section, ops, memory):
        """
        Add a profiling section.

        :param name: The name which uniquely identifies the profiled code section.
        :param section: The code section, represented as a tuple of :class:`Iteration`s.
        :param ops: The number of floating-point operations in the section.
        :param memory: The memory traffic in the section, as bytes moved from/to memory.
        """
        self._sections[section] = Profile(name, ops, memory)

    def setup(self):
        """
        Allocate and return a pointer to the timers C-level Struct, which includes
        all timers added to ``self`` through ``self.add(...)``.
        """
        cls = type("Timings", (Structure,),
                   {"_fields_": [(i.name, c_double) for i in self._sections.values()]})
        self._C_timings = cls()
        return byref(self._C_timings)

    def summary(self, dim_sizes, dtype):
        """
        Return a summary of the performance numbers measured.

        :param dim_sizes: The run-time extent of each :class:`Iteration` tracked
                          by this Profiler. Used to compute the operational intensity
                          and the perfomance achieved in GFlops/s.
        :param dtype: The data type of the objects in the profiled sections. Used
                      to compute the operational intensity.
        """

        summary = PerformanceSummary()
        for itspace, profile in self._sections.items():
            dims = {i: i.dim.parent if i.dim.is_Buffered else i.dim for i in itspace}

            # Time
            time = self.timings[profile.name]

            # Flops
            itershape = [i.extent(finish=dim_sizes.get(dims[i].name)) for i in itspace]
            iterspace = reduce(operator.mul, itershape)
            flops = float(profile.ops*iterspace)
            gflops = flops/10**9

            # Compulsory traffic
            datashape = [i.dim.size or dim_sizes[dims[i].name] for i in itspace]
            dataspace = reduce(operator.mul, datashape)
            traffic = profile.memory*dataspace*dtype().itemsize

            # Derived metrics
            oi = flops/traffic
            gflopss = gflops/time

            # Keep track of performance achieved
            summary.setsection(profile.name, time, gflopss, oi, itershape, datashape)

        # Rename the most time consuming section as 'main'
        summary['main'] = summary.pop(max(summary, key=summary.get))

        return summary

    @property
    def timings(self):
        """
        Return the timings, up to microseconds, as a dictionary.
        """
        if self._C_timings is None:
            raise RuntimeError("Cannot extract timings with non-finalized Profiler.")
        return {field: max(getattr(self._C_timings, field), 10**-6)
                for field, _ in self._C_timings._fields_}

    @property
    def ctype(self):
        """
        Returns a :class:`cgen.Struct` relative to the profiler.
        """
        return Struct(Profiler.typename,
                      [Value('double', i.name) for i in self._sections.values()])


class PerformanceSummary(OrderedDict):

    """
    A special dictionary to track and quickly access performance data.
    """

    def setsection(self, key, time, gflopss, oi, itershape, datashape):
        self[key] = PerfEntry(time, gflopss, oi, itershape, datashape)

    @property
    def gflopss(self):
        return OrderedDict([(k, v.gflopss) for k, v in self.items()])

    @property
    def oi(self):
        return OrderedDict([(k, v.oi) for k, v in self.items()])

    @property
    def timings(self):
        return OrderedDict([(k, v.time) for k, v in self.items()])


Profile = namedtuple('Profile', 'name ops memory')
"""Metadata for a profiled code section."""


PerfEntry = namedtuple('PerfEntry', 'time gflopss oi itershape datashape')
"""Structured performance data."""

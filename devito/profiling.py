from __future__ import absolute_import

import operator
from collections import OrderedDict, namedtuple
from functools import reduce

from ctypes import Structure, byref, c_double
from cgen import Struct, Value

from devito.ir.iet import (Expression, TimedList, FindNodes, Transformer,
                           FindAdjacentIterations, retrieve_iteration_tree)
from devito.symbolics import estimate_cost, estimate_memory
from devito.tools import flatten

__all__ = ['Profile', 'create_profile']


def create_profile(name, node):
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
    profiler = Profiler(name)

    trees = retrieve_iteration_tree(node)
    if not trees:
        return node, profiler

    adjacents = [flatten(i) for i in FindAdjacentIterations().visit(node).values() if i]

    def are_adjacent(tree, last):
        for i, j in zip(tree, last):
            if i == j:
                continue
            try:
                return any(abs(a.index(j) - a.index(i)) == 1 for a in adjacents)
            except ValueError:
                return False

    # Group Iterations based on timing region
    key, groups = lambda itspace: {i.defines for i in itspace}, []
    handle = [trees[0]]
    for tree in trees[1:]:
        last = handle[-1]
        if key(tree) == key(last) and are_adjacent(tree, last):
            handle.append(tree)
        else:
            groups.append(tuple(handle))
            handle = [tree]
    groups.append(tuple(handle))

    # Create and track C-level timers
    mapper = OrderedDict()
    for group in groups:
        # We time at the single timestep level
        for i in zip(*group):
            root = i[0]
            remainder = tuple(j for j in i if j is not root)
            if not root.dim.is_Time:
                break
        if root in mapper:
            continue

        # Prepare to transform the Iteration/Expression tree
        body = (root,) + remainder
        lname = 'section_%d' % len(mapper)
        mapper[root] = TimedList(gname=name, lname=lname, body=body)
        mapper.update(OrderedDict([(j, None) for j in remainder]))

        # Estimate computational properties of the profiled section
        expressions = FindNodes(Expression).visit(body)
        ops = estimate_cost([e.expr for e in expressions])
        memory = estimate_memory([e.expr for e in expressions])

        # Keep track of the new profiled section
        profiler.add(lname, group[0], ops, memory)

    # Transform the Iteration/Expression tree introducing the C-level timers
    processed = Transformer(mapper).visit(node)

    return processed, profiler


class Profiler(object):

    """
    A Profiler is used to manage profiling information for Devito generated C code.
    """

    structname = "profile"

    def __init__(self, name):
        self.name = name
        self._sections = OrderedDict()

    def add(self, name, section, ops, memory):
        """
        Add a profiling section.

        :param name: The name which uniquely identifies the profiled code section.
        :param section: The code section, represented as a tuple of :class:`Iteration`s.
        :param ops: The number of floating-point operations in the section.
        :param memory: The memory traffic in the section, as bytes moved from/to memory.
        """
        self._sections[section] = Profile(name, ops, memory)

    def new(self):
        """
        Allocate and return a pointer to a new C-level Struct capable of storing
        all timers added through ``self.add(...)``.
        """
        return byref(self.dtype())

    def summary(self, arguments, dtype):
        """
        Return a :class:`PerformanceSummary` of the tracked sections.

        :param arguments: A mapper from argument names to run-time values from which
                          the Profiler infers iteration space and execution times
                          of a run.
        :param dtype: The data type of the objects in the profiled sections. Used
                      to compute the operational intensity.
        """

        summary = PerformanceSummary()
        for itspace, profile in self._sections.items():
            dims = {i: i.dim.parent if i.dim.is_Stepping else i.dim for i in itspace}

            # Time
            time = max(getattr(arguments[self.name]._obj, profile.name), 10e-7)

            # Flops
            itershape = [i.extent(finish=arguments[dims[i].max_name],
                                  start=arguments[dims[i].min_name]) for i in itspace]
            iterspace = reduce(operator.mul, itershape)
            flops = float(profile.ops*iterspace)
            gflops = flops/10**9
            gpoints = iterspace/10**9

            # Compulsory traffic
            datashape = [(arguments[dims[i].max_name] - arguments[dims[i].min_name] + 1)
                         for i in itspace]
            dataspace = reduce(operator.mul, datashape)
            traffic = float(profile.memory*dataspace*dtype().itemsize)

            # Derived metrics
            oi = flops/traffic
            gflopss = gflops/time
            gpointss = gpoints/time

            # Keep track of performance achieved
            summary.setsection(profile.name, time, gflopss, gpointss, oi, profile.ops,
                               itershape, datashape)

        # Rename the most time consuming section as 'main'
        if len(summary) > 0:
            summary['main'] = summary.pop(max(summary, key=summary.get))

        return summary

    @property
    def dtype(self):
        """
        Return the profiler C type in ctypes format.
        """
        return type(Profiler.structname, (Structure,),
                    {"_fields_": [(i.name, c_double) for i in self._sections.values()]})

    @property
    def cdef(self):
        """
        Returns a :class:`cgen.Struct` representing the profiler data structure in C
        (a ``struct``).
        """
        return Struct(Profiler.structname,
                      [Value('double', i.name) for i in self._sections.values()])


class PerformanceSummary(OrderedDict):

    """
    A special dictionary to track and quickly access performance data.
    """

    def setsection(self, key, time, gflopss, gpointss, oi, ops, itershape, datashape):
        self[key] = PerfEntry(time, gflopss, gpointss, oi, ops, itershape, datashape)

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


PerfEntry = namedtuple('PerfEntry', 'time gflopss gpointss oi ops itershape datashape')
"""Structured performance data."""

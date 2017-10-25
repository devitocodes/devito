"""
A collection of algorithms to analyze and decorate :class:`Iteration` in an
Iteration/Expression tree. Decoration comes in the form of :class:`IterationProperty`
objects, attached to nodes in the Iteration/Expression tree.

The current implementation is based on simplistic analysis of Iteration/Expression
trees. A better implementation would use a control flow graph, perhaps derived
the Iteration/Expression tree itself. An ever better implementation would use
a library such as ISL (the Integer Set Library).
"""

from collections import OrderedDict

from devito.nodes import Iteration, SEQUENTIAL, PARALLEL, VECTOR, WRAPPABLE
from devito.symbolics import as_symbol
from devito.tools import as_tuple, flatten
from devito.visitors import FindSections, IsPerfectIteration, NestedTransformer


def analyze_iterations(nodes):
    """
    Attach :class:`IterationProperty` to :class:`Iteration` objects within
    ``nodes``. The recognized IterationProperty decorators are listed in
    ``nodes.IterationProperty._KNOWN``.
    """
    sections = FindSections().visit(nodes)

    # Local analysis: detect Iteration properties, inspecting trees in isolation
    mapper = OrderedDict()
    for tree, exprs in sections.items():
        deps_graph = compute_dependency_graph(exprs)

        mapper = detect_fully_parallel(tree, deps_graph, mapper)
        mapper = detect_outermost_parallel(tree, deps_graph, mapper)
        mapper = detect_outermost_sequential_inner_parallel(tree, deps_graph, mapper)
        mapper = detect_innermost_unitstride(tree, deps_graph, mapper)
        mapper = detect_wrappable_iterations(tree, deps_graph, mapper)

    # Global analysis
    for k, v in list(mapper.items()):
        args = k.args
        # SEQUENTIAL kills PARALLEL
        properties = [i for i in v if i != PARALLEL] if SEQUENTIAL in v else v
        properties = as_tuple(args.pop('properties')) + as_tuple(properties)
        mapper[k] = Iteration(properties=properties, **args)

    # Store the discovered properties in the Iteration/Expression tree
    processed = NestedTransformer(mapper).visit(nodes)

    return processed


def compute_dependency_graph(exprs):
    """
    Given an ordered list of :class:`Expression`, build a mapper from lvalues
    to reads occurring in the rvalues.
    """
    deps_graph = OrderedDict()
    writes = [e.output for e in exprs if e.is_tensor]
    for i in writes:
        for e in exprs:
            i_reads = [j for j in e.reads if as_symbol(i) == as_symbol(j)]
            deps_graph.setdefault(i, []).extend(i_reads)
    return deps_graph


def detect_fully_parallel(tree, deps_graph, mapper=None):
    """
    Update ``mapper``, a dictionary from :class:`Iteration`s to
    :class:`IterationProperty`s, by annotating nested, fully-parallel Iterations.
    """
    if mapper is None:
        mapper = OrderedDict()
    is_FP = True
    for k, v in deps_graph.items():
        is_FP &= all(k.indices == i.indices for i in v)
    if is_FP:
        for i in tree:
            mapper.setdefault(i, []).append(PARALLEL)
    return mapper


def detect_outermost_parallel(tree, deps_graph, mapper=None):
    """
    Update ``mapper``, a dictionary from :class:`Iteration`s to
    :class:`IterationProperty`s, by annotating the outermost Iteration if this
    turns out to be parallel.
    """
    if mapper is None:
        mapper = OrderedDict()
    is_OP = True
    for k, v in deps_graph.items():
        for i in v:
            is_OP &= k.indices[0] == i.indices[0]
            is_OP &= all(k.indices[0].free_symbols.isdisjoint(j.free_symbols)
                         for j in i.indices[1:])  # not A[x,y] = A[x,x+1]
    if is_OP:
        mapper.setdefault(tree[0], []).append(PARALLEL)
    return mapper


def detect_outermost_sequential_inner_parallel(tree, deps_graph, mapper=None):
    """
    Update ``mapper``, a dictionary from :class:`Iteration`s to
    :class:`IterationProperty`s, by annotating the outermost Iteration if this
    turns out to be sequential and, if that's the case, by annotating the inner
    Iterations as parallel.
    """
    if mapper is None:
        mapper = OrderedDict()
    candidate = tree[0]
    filtered_mapper = OrderedDict()
    for k, v in deps_graph.items():
        if candidate.dim == k.base.function.indices[0] and len(v) > 0:
            filtered_mapper[k] = v
    if not filtered_mapper:
        # The outermost Iteration is actually parallel
        return mapper
    if len({i.base.function.indices[0] for i in filtered_mapper}) > 1:
        # Must be u[t+1] = ... v[t+1] = ... , not u[t+1] = ... v[t+2] = ...
        return mapper
    is_OS = True
    for k, v in filtered_mapper.items():
        # At least one access along the candidate dimension differs (u[t+1] = u[t] ...)
        # AND the others either differ too or are identical to the LHS, that is
        # u[t+1, x] = u[t, x] + u[t+1, x] OK, but u[t+1, x] = u[t, x] + u[t+1, x-1] NO
        is_OS &= all(k.base.function.indices[0] == i.base.function.indices[0] for i in v)
        is_OS &= any(k.indices[0] != i.indices[0] for i in v)
        is_OS &= all(k.indices[0] != i.indices[0] or
                     k.indices[1:] == i.indices[1:] for i in v)
    if is_OS:
        mapper.setdefault(candidate, []).append(SEQUENTIAL)
        for i in tree[tree.index(candidate) + 1:]:
            mapper.setdefault(i, []).append(PARALLEL)
    return mapper


def detect_innermost_unitstride(tree, deps_graph, mapper=None):
    """
    Update ``mapper``, a dictionary from :class:`Iteration`s to
    :class:`IterationProperty`s, by annotating the innermost Iteration as
    vectorizable if all array accesses along its dimension turn out to
    be unit-strided.
    """
    if mapper is None:
        mapper = OrderedDict()
    innermost = tree[-1]
    if not IsPerfectIteration().visit(innermost):
        return mapper
    if len(tree) == 1 or SEQUENTIAL in mapper.get(tree[-2], []):
        # Heuristic: there should be at least an outer parallel Iteration
        # to mark /innermost/ as vectorizable, otherwise it is preferable
        # to save it for shared-memory parallelism
        return mapper
    is_US = True
    for k, v in deps_graph.items():
        is_US &= all(k.indices[-1] == i.indices[-1] for i in v)
    if is_US or PARALLEL in mapper.get(innermost, []):
        mapper.setdefault(innermost, []).append(VECTOR)
    return mapper


def detect_wrappable_iterations(tree, deps_graph, mapper=None):
    """
    Update ``mapper``, a dictionary from :class:`Iteration`s to
    :class:`IterationProperty`s, by annotating an Iteration as wrappable
    if the first and last slots accessed through modulo buffered iteration
    can be mapped to a single slot, thus reducing the working set.
    """
    if mapper is None:
        mapper = OrderedDict()
    stepping = [i for i in tree if i.dim.is_Stepping]
    if len(stepping) != 1:
        return mapper
    stepping = stepping[0]
    is_WP = all(stepping.dim == i.base.function.indices[0] for i in deps_graph)
    if is_WP:
        accesses = {i.indices[0] for i in deps_graph}
        accesses |= {i.indices[0] for i in flatten(deps_graph.values())}
        candidate = sorted(accesses, key=lambda i: i.subs(stepping.dim, 0))[0]
        for k, v in deps_graph.items():
            is_WP &= all(k.indices[1:] == i.indices[1:] for i in v
                         if candidate == i.indices[0])
    if is_WP:
        mapper.setdefault(stepping, []).append(WRAPPABLE)
    return mapper

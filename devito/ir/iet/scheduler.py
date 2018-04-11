from collections import OrderedDict

import numpy as np

from devito.cgen_utils import Allocator
from devito.dimension import LoweredDimension
from devito.ir.iet import (Expression, LocalExpression, Element, Iteration, List,
                           Conditional, UnboundedIndex, MetaCall, MapExpressions,
                           Transformer, NestedTransformer, SubstituteExpression,
                           iet_analyze, filter_iterations, retrieve_iteration_tree)
from devito.ir.support import IterationSpace
from devito.tools import filter_ordered, flatten
from devito.types import Scalar

__all__ = ['iet_build', 'iet_insert_C_decls']


def iet_build(clusters):
    """
    Create an Iteration/Expression tree (IET) given an iterable of :class:`Cluster`s.
    The nodes in the returned IET are decorated with properties deriving from
    data dependence analysis.
    """
    # Clusters -> Iteration/Expression tree
    iet = iet_make(clusters)

    # Data dependency analysis. Properties are attached directly to nodes
    iet = iet_analyze(iet)

    # Substitute derived dimensions (e.g., t -> t0, t + 1 -> t1)
    # This is postponed up to this point to ease /iet_analyze/'s life
    subs = {}
    for tree in retrieve_iteration_tree(iet):
        uindices = flatten(i.uindices for i in tree)
        subs.update({i.expr: LoweredDimension(name=i.index.name, origin=i.expr)
                     for i in uindices})
    iet = SubstituteExpression(subs).visit(iet)

    return iet


def iet_make(clusters):
    """
    Create an Iteration/Expression tree (IET) given an iterable of :class:`Cluster`s.

    :param clusters: The iterable :class:`Cluster`s for which the IET is built.
    """
    # {Iteration -> [c0, c1, ...]}, shared clusters
    shared = {}
    # The constructed IET
    processed = []
    # {Interval -> Iteration}, carried from preceding cluster
    schedule = OrderedDict()

    # Build IET
    for cluster in clusters:
        body = [Expression(e) for e in cluster.exprs]

        if cluster.ispace.empty:
            # No Iterations are needed
            processed.extend(body)
            continue

        root = None
        itintervals = cluster.ispace.iteration_intervals

        # Can I reuse any of the previously scheduled Iterations ?
        index = 0
        for i0, i1 in zip(itintervals, list(schedule)):
            if i0 != i1 or i0.dim in cluster.atomics:
                break
            root = schedule[i1]
            index += 1
        needed = itintervals[index:]

        # Build Expressions
        if not needed:
            body = List(body=body)

        # Build Iterations
        scheduling = []
        for i in reversed(needed):
            # Update IET and scheduling
            if i.dim in cluster.guards:
                # Must wrap within an if-then scope
                body = Conditional(cluster.guards[i.dim], body)
                # Adding (None, None) ensures that nested iterations won't
                # be reused by the next cluster
                scheduling.insert(0, (None, None))
            iteration = Iteration(body, i.dim, i.dim.limits, offsets=i.limits,
                                  direction=i.direction)
            scheduling.insert(0, (i, iteration))

            # Prepare for next dimension
            body = iteration

        # If /needed/ is != [], root.dim might be a guarded dimension for /cluster/
        if root is not None and root.dim in cluster.guards:
            body = Conditional(cluster.guards[root.dim], body)

        # Update the current schedule
        if root is None:
            processed.append(body)
        else:
            nodes = list(root.nodes) + [body]
            transf = Transformer({root: root._rebuild(nodes, **root.args_frozen)})
            processed = list(transf.visit(processed))
            scheduling = list(schedule.items())[:index] + list(scheduling)
            scheduling = [(k, transf.rebuilt.get(v, v)) for k, v in scheduling]
            shared = {transf.rebuilt.get(k, k): v for k, v in shared.items()}
        schedule = OrderedDict(scheduling)

        # Record that /cluster/ was used to build the iterations in /schedule/
        shared.update({i: shared.get(i, []) + [cluster] for i in schedule.values() if i})
    iet = List(body=processed)

    # Add in unbounded indices, if needed
    mapper = {}
    for k, v in shared.items():
        uindices = []
        ispace = IterationSpace.merge(*[i.ispace.project([k.dim]) for i in v])
        for j, offs in ispace.sub_iterators.get(k.dim, []):
            modulo = len(offs)
            for n, o in enumerate(filter_ordered(offs)):
                name = "%s%d" % (j.name, n)
                vname = Scalar(name=name, dtype=np.int32)
                value = (k.dim + o) % modulo
                uindices.append(UnboundedIndex(vname, value, value, j, j + o))
        mapper[k] = k._rebuild(uindices=uindices)
    iet = NestedTransformer(mapper).visit(iet)

    return iet


def iet_insert_C_decls(iet, func_table):
    """
    Given an Iteration/Expression tree ``iet``, build a new tree with the
    necessary symbol declarations. Declarations are placed as close as
    possible to the first symbol use.

    :param iet: The input Iteration/Expression tree.
    :param func_table: A mapper from callable names to :class:`Callable`s
                       called from within ``iet``.
    """
    # Resolve function calls first
    scopes = []
    me = MapExpressions()
    for k, v in me.visit(iet).items():
        if k.is_Call:
            func = func_table[k.name]
            if func.local:
                scopes.extend(me.visit(func.root, queue=list(v)).items())
        else:
            scopes.append((k, v))

    # Determine all required declarations
    allocator = Allocator()
    mapper = OrderedDict()
    for k, v in scopes:
        if k.is_scalar:
            # Inline declaration
            mapper[k] = LocalExpression(**k.args)
        elif k.write._mem_external:
            # Nothing to do, variable passed as kernel argument
            continue
        elif k.write._mem_stack:
            # On the stack, as established by the DLE
            key = lambda i: not i.is_Parallel
            site = filter_iterations(v, key=key, stop='asap') or [iet]
            allocator.push_stack(site[-1], k.write)
        else:
            # On the heap, as a tensor that must be globally accessible
            allocator.push_heap(k.write)

    # Introduce declarations on the stack
    for k, v in allocator.onstack:
        mapper[k] = tuple(Element(i) for i in v)
    iet = NestedTransformer(mapper).visit(iet)
    for k, v in list(func_table.items()):
        if v.local:
            func_table[k] = MetaCall(Transformer(mapper).visit(v.root), v.local)

    # Introduce declarations on the heap (if any)
    if allocator.onheap:
        decls, allocs, frees = zip(*allocator.onheap)
        iet = List(header=decls + allocs, body=iet, footer=frees)

    return iet

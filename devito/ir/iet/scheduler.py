from collections import OrderedDict

import numpy as np

from devito.cgen_utils import Allocator
from devito.dimension import LoweredDimension
from devito.ir.iet import (Expression, LocalExpression, Element, Iteration, List,
                           UnboundedIndex, MetaCall, MapExpressions, Transformer,
                           NestedTransformer, SubstituteExpression, iet_analyze,
                           compose_nodes, filter_iterations, retrieve_iteration_tree)
from devito.tools import filter_ordered, flatten
from devito.types import Scalar

__all__ = ['iet_build', 'iet_insert_C_decls']


def iet_build(clusters, dtype):
    """
    Create an Iteration/Expression tree (IET) given an iterable of :class:`Cluster`s.
    The nodes in the returned IET are decorated with properties deriving from
    data dependence analysis.
    """
    # Clusters -> Iteration/Expression tree
    iet = iet_make(clusters, dtype)

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


def iet_make(clusters, dtype):
    """
    Create an Iteration/Expression tree (IET) given an iterable of :class:`Cluster`s.

    :param clusters: The iterable :class:`Cluster`s for which the IET is built.
    :param dtype: The data type of the scalar expressions.
    """
    processed = []
    schedule = OrderedDict()
    for cluster in clusters:
        if not cluster.ispace.empty:
            root = None
            intervals = cluster.ispace.intervals

            # Can I reuse any of the previously scheduled Iterations ?
            index = 0
            for i0, i1 in zip(intervals, list(schedule)):
                if i0 != i1 or i0.dim in clusters.atomics[cluster]:
                    break
                root = schedule[i1]
                index += 1
            needed = intervals[index:]

            # Build Iterations, including any necessary unbounded index
            iters = []
            for i in needed:
                uindices = []
                for j, offs in cluster.ispace.sub_iterators.get(i.dim, []):
                    for n, o in enumerate(filter_ordered(offs)):
                        name = "%s%d" % (j.name, n)
                        vname = Scalar(name=name, dtype=np.int32)
                        value = (i.dim + o) % j.modulo
                        uindices.append(UnboundedIndex(vname, value, value, j, j + o))
                iters.append(Iteration([], i.dim, i.dim.limits, offsets=i.limits,
                                       uindices=uindices))

            # Build Expressions
            exprs = [Expression(v, np.int32 if cluster.trace.is_index(k) else dtype)
                     for k, v in cluster.trace.items()]

            # Compose Iterations and Expressions
            body, tree = compose_nodes(iters + [exprs], retrieve=True)

            # Update the current scheduling
            scheduling = OrderedDict(zip(needed, tree))
            if root is None:
                processed.append(body)
                schedule = scheduling
            else:
                nodes = list(root.nodes) + [body]
                mapper = {root: root._rebuild(nodes, **root.args_frozen)}
                transformer = Transformer(mapper)
                processed = list(transformer.visit(processed))
                schedule = OrderedDict(list(schedule.items())[:index] +
                                       list(scheduling.items()))
                for k, v in list(schedule.items()):
                    schedule[k] = transformer.rebuilt.get(v, v)
        else:
            # No Iterations are needed
            processed.extend([Expression(e, dtype) for e in cluster.exprs])

    return List(body=processed)


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

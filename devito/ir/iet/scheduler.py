from collections import OrderedDict

import numpy as np

from devito.cgen_utils import Allocator
from devito.dimension import LoweredDimension
from devito.ir.iet import (Expression, LocalExpression, Element, Iteration, List,
                           Conditional, Section, ExpressionBundle, UnboundedIndex,
                           MetaCall, MapExpressions, Transformer, NestedTransformer,
                           SubstituteExpression, iet_analyze, filter_iterations,
                           retrieve_iteration_tree)
from devito.tools import filter_ordered, flatten
from devito.types import Scalar

__all__ = ['iet_build', 'iet_insert_C_decls']


def iet_build(stree):
    """
    Create an Iteration/Expression tree (IET) from a :class:`ScheduleTree`.
    The nodes in the returned IET are decorated with properties deriving from
    data dependence analysis.
    """
    # Schedule tree -> Iteration/Expression tree
    iet = iet_make(stree)

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


def iet_make(stree):
    """
    Create an Iteration/Expression tree (IET) from a :class:`ScheduleTree`.
    """
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            return List(body=queues.pop(i))

        elif i.is_Exprs:
            exprs = [Expression(e) for e in i.exprs]
            body = [ExpressionBundle(i.shape, i.ops, i.traffic, body=exprs)]

        elif i.is_Conditional:
            body = [Conditional(i.guard, queues.pop(i))]

        elif i.is_Iteration:
            # Generate `uindices`
            uindices = []
            for d, offs in i.sub_iterators:
                modulo = len(offs)
                for n, o in enumerate(filter_ordered(offs)):
                    value = (i.dim + o) % modulo
                    symbol = Scalar(name="%s%d" % (d.name, n), dtype=np.int32)
                    uindices.append(UnboundedIndex(symbol, value, value, d, d + o))
            # Generate Iteration
            body = [Iteration(queues.pop(i), i.dim, i.dim.limits, offsets=i.limits,
                              direction=i.direction, uindices=uindices)]

        elif i.is_Section:
            body = [Section('section%d' % nsections, body=queues.pop(i))]
            nsections += 1

        queues.setdefault(i.parent, []).extend(body)

    assert False


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
        elif k.write is None or k.write._mem_external:
            # Nothing to do, e.g., variable passed as kernel argument
            continue
        elif k.write._mem_stack:
            # On the stack
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

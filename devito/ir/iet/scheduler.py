from collections import OrderedDict

from devito.cgen_utils import Allocator
from devito.ir.iet import (Expression, LocalExpression, Element, Iteration, List,
                           Conditional, Section, ExpressionBundle, MetaCall,
                           MapExpressions, Transformer, NestedTransformer, FindNodes,
                           ReplaceStepIndices, iet_analyze, filter_iterations)
from devito.tools import as_mapper

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

    # Replace stepping dimensions with modulo dimensions
    iet = iet_lower_steppers(iet)

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
            # Order to ensure deterministic code generation
            uindices = sorted(i.sub_iterators, key=lambda d: d.name)
            # Generate Iteration
            body = [Iteration(queues.pop(i), i.dim, i.dim.limits, offsets=i.limits,
                              direction=i.direction, uindices=uindices)]

        elif i.is_Section:
            body = [Section('section%d' % nsections, body=queues.pop(i))]
            nsections += 1

        queues.setdefault(i.parent, []).extend(body)

    assert False


def iet_lower_steppers(iet):
    """
    Replace the :class:`SteppingDimension`s within ``iet``'s expressions with
    suitable :class:`ModuloDimension`s.
    """
    for i in FindNodes(Iteration).visit(iet):
        if not i.uindices:
            # Be quick: avoid uselessy reconstructing nodes
            continue
        # In an expression, there could be `u[t+1, ...]` and `v[t+1, ...]`, where
        # `u` and `v` are TimeFunction with circular time buffers (save=None) *but*
        # different modulo extent. The `t+1` indices above are therefore conceptually
        # different, so they will be replaced with the proper ModuloDimension through
        # two different calls to `xreplace`
        groups = as_mapper(i.uindices, lambda d: d.modulo)
        for k, v in groups.items():
            mapper = {d.origin: d for d in v}
            rule = lambda i: i.function._time_size == k
            iet = ReplaceStepIndices(mapper, rule).visit(iet)
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

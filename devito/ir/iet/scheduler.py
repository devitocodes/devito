from collections import OrderedDict

from devito.cgen_utils import Allocator
from devito.ir.iet import (Expression, Increment, LocalExpression, Element, Iteration,
                           List, Conditional, Section, HaloSpot, ExpressionBundle,
                           MapExpressions, Transformer, FindNodes, FindSymbols, XSubs,
                           iet_analyze, filter_iterations)
from devito.symbolics import IntDiv, xreplace_indices
from devito.tools import as_mapper
from devito.types import ConditionalDimension

__all__ = ['iet_build', 'iet_insert_C_decls']


def iet_build(stree):
    """
    Create an Iteration/Expression tree (IET) from a ScheduleTree.

    The nodes in the returned IET are decorated with properties deriving from
    data dependence analysis.
    """
    # Schedule tree -> Iteration/Expression tree
    iet = iet_make(stree)

    # Data dependency analysis. Properties are attached directly to nodes
    iet = iet_analyze(iet)

    # Turn DerivedDimensions into lower-level Dimensions or Symbols
    iet = iet_lower_dimensions(iet)

    return iet


def iet_make(stree):
    """Create an IET from a ScheduleTree."""
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            return List(body=queues.pop(i))

        elif i.is_Exprs:
            exprs = [Increment(e) if e.is_Increment else Expression(e) for e in i.exprs]
            body = ExpressionBundle(i.shape, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            # Order to ensure deterministic code generation
            uindices = sorted(i.sub_iterators, key=lambda d: d.name)
            # Generate Iteration
            body = Iteration(queues.pop(i), i.dim, i.dim._limits, offsets=i.limits,
                             direction=i.direction, uindices=uindices)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            body = HaloSpot(i.halo_scheme, body=queues.pop(i))

        queues.setdefault(i.parent, []).append(body)

    assert False


def iet_lower_dimensions(iet):
    """
    Replace all DerivedDimensions within the ``iet``'s expressions with
    lower-level symbolic objects (other Dimensions or Symbols).

        * Array indices involving SteppingDimensions are turned into ModuloDimensions.
          Example: ``u[t+1, x] = u[t, x] + 1 >>> u[t1, x] = u[t0, x] + 1``
        * Array indices involving ConditionalDimensions used are turned into
          integer-division expressions.
          Example: ``u[t_sub, x] = u[time, x] >>> u[time / 4, x] = u[time, x]``
    """
    # Lower SteppingDimensions
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
            rule = lambda i: i.function.is_TimeFunction and i.function._time_size == k
            replacer = lambda i: xreplace_indices(i, mapper, rule)
            iet = XSubs(replacer=replacer).visit(iet)

    # Lower ConditionalDimensions
    cdims = [d for d in FindSymbols('free-symbols').visit(iet)
             if isinstance(d, ConditionalDimension)]
    mapper = {d: IntDiv(d.index, d.factor) for d in cdims}
    iet = XSubs(mapper).visit(iet)

    return iet


def iet_insert_C_decls(iet, external=None):
    """
    Given an IET, build a new tree with the necessary symbol declarations.
    Declarations are placed as close as possible to the first symbol occurrence.

    Parameters
    ----------
    iet : Node
        The input Iteration/Expression tree.
    external : tuple, optional
        The symbols defined in some outer Callable, which therefore must not
        be re-defined.
    """
    external = external or []

    # Classify and then schedule declarations to stack/heap
    allocator = Allocator()
    mapper = OrderedDict()
    for k, v in MapExpressions().visit(iet).items():
        if k.is_Expression:
            if k.is_scalar_assign:
                # Inline declaration
                mapper[k] = LocalExpression(**k.args)
                continue
            objs = [k.write]
        elif k.is_Call:
            objs = k.arguments

        for i in objs:
            try:
                if i.is_LocalObject:
                    # On the stack
                    site = v[-1] if v else iet
                    allocator.push_stack(site, i)
                elif i.is_Array:
                    if i in external:
                        # The Array is to be defined in some foreign IET
                        continue
                    elif i._mem_stack:
                        # On the stack
                        key = lambda i: not i.is_Parallel
                        site = filter_iterations(v, key=key, stop='asap') or [iet]
                        allocator.push_stack(site[-1], i)
                    else:
                        # On the heap, as a tensor that must be globally accessible
                        allocator.push_heap(i)
            except AttributeError:
                # E.g., a generic SymPy expression
                pass

    # Introduce declarations on the stack
    for k, v in allocator.onstack:
        mapper[k] = tuple(Element(i) for i in v)
    iet = Transformer(mapper, nested=True).visit(iet)

    # Introduce declarations on the heap (if any)
    if allocator.onheap:
        decls, allocs, frees = zip(*allocator.onheap)
        iet = List(header=decls + allocs, body=iet, footer=frees)

    return iet

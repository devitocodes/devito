from collections import OrderedDict

import cgen as c

from devito.cgen_utils import ccode
from devito.ir.iet import (ArrayCast, Expression, Increment, LocalExpression, Element,
                           Iteration, List, Conditional, Section, HaloSpot,
                           ExpressionBundle, MapSections, Transformer, FindNodes,
                           FindSymbols, XSubs, iet_analyze, filter_iterations)
from devito.symbolics import IntDiv, xreplace_indices
from devito.tools import as_mapper, as_tuple
from devito.types import ConditionalDimension

__all__ = ['iet_build', 'iet_insert_decls', 'iet_insert_casts']


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


def iet_insert_casts(iet, parameters):
    """
    Transform the input IET inserting the necessary type casts.
    The type casts are placed at the top of the IET.

    Parameters
    ----------
    iet : Node
        The input Iteration/Expression tree.
    parameters : tuple, optional
        The symbol that might require casting.
    """
    # Make the generated code less verbose: if a non-Array parameter does not
    # appear in any Expression, that is, if the parameter is merely propagated
    # down to another Call, then there's no need to cast it
    exprs = FindNodes(Expression).visit(iet)
    need_cast = {i for i in set().union(*[i.functions for i in exprs]) if i.is_Tensor}
    need_cast.update({i for i in parameters if i.is_Array})

    casts = [ArrayCast(i) for i in parameters if i in need_cast]
    iet = List(body=casts + [iet])
    return iet


def iet_insert_decls(iet, external):
    """
    Transform the input IET inserting the necessary symbol declarations.
    Declarations are placed as close as possible to the first symbol occurrence.

    Parameters
    ----------
    iet : Node
        The input Iteration/Expression tree.
    external : tuple, optional
        The symbols defined in some outer Callable, which therefore must not
        be re-defined.
    """
    iet = as_tuple(iet)

    # Classify and then schedule declarations to stack/heap
    allocator = Allocator()
    for k, v in MapSections().visit(iet).items():
        if k.is_Expression:
            if k.is_scalar_assign:
                # On the stack
                site = v if v else iet
                allocator.push_scalar_on_stack(site[-1], k)
                continue
            objs = [k.write]
        elif k.is_Call:
            objs = k.arguments

        for i in objs:
            try:
                if i.is_LocalObject:
                    # On the stack
                    site = v if v else iet
                    allocator.push_object_on_stack(site[-1], i)
                elif i.is_Array:
                    if i in as_tuple(external):
                        # The Array is defined in some other IET
                        continue
                    elif i._mem_stack:
                        # On the stack
                        key = lambda i: not i.is_Parallel
                        site = filter_iterations(v, key=key) or iet
                        allocator.push_object_on_stack(site[-1], i)
                    else:
                        # On the heap
                        allocator.push_array_on_heap(i)
            except AttributeError:
                # E.g., a generic SymPy expression
                pass

    # Introduce declarations on the stack
    mapper = dict(allocator.onstack)
    iet = Transformer(mapper, nested=True).visit(iet)

    # Introduce declarations on the heap (if any)
    if allocator.onheap:
        decls, allocs, frees = zip(*allocator.onheap)
        iet = List(header=decls + allocs, body=iet, footer=frees)

    return iet


class Allocator(object):

    """
    Support class to generate allocation code in different scopes.
    """

    def __init__(self):
        self.heap = OrderedDict()
        self.stack = OrderedDict()

    def push_object_on_stack(self, scope, obj):
        """Define an Array or a composite type (e.g., a struct) on the stack."""
        handle = self.stack.setdefault(scope, OrderedDict())

        if obj.is_LocalObject:
            handle[obj] = Element(c.Value(obj._C_typename, obj.name))
        else:
            shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
            alignment = "__attribute__((aligned(%d)))" % obj._data_alignment
            value = "%s%s %s" % (obj.name, shape, alignment)
            handle[obj] = Element(c.POD(obj.dtype, value))

    def push_scalar_on_stack(self, scope, expr):
        """Define a Scalar on the stack."""
        handle = self.stack.setdefault(scope, OrderedDict())

        obj = expr.write
        if obj in handle:
            return

        handle[obj] = None  # Placeholder to avoid reallocation
        self.stack[expr] = LocalExpression(**expr.args)

    def push_array_on_heap(self, obj):
        """Define an Array on the heap."""
        if obj in self.heap:
            return

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        self.heap[obj] = (decl, alloc, free)

    @property
    def onstack(self):
        ret = []
        for k, v in self.stack.items():
            try:
                ret.append((k, [i for i in v.values() if i is not None]))
            except AttributeError:
                ret.append((k, v))
        return ret

    @property
    def onheap(self):
        return self.heap.values()

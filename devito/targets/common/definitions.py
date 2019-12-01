"""
Collection of passes for the declaration, allocation, movement and deallocation
of symbols and data.
"""

from collections import OrderedDict

import cgen as c

from devito.ir import (ArrayCast, Element, Expression, List, LocalExpression,
                       FindNodes, MapExprStmts, Transformer)
from devito.symbolics import ccode
from devito.targets.common.rewriters import dle_pass  #TODO: Turn dle_pass into something else
from devito.tools import flatten

__all__ = ['insert_defs', 'insert_casts']


class Allocator(object):

    def __init__(self):
        self.heap = OrderedDict()
        self.stack = OrderedDict()

    def push_object_on_stack(self, scope, obj):
        """Define a LocalObject on the stack."""
        handle = self.stack.setdefault(scope, OrderedDict())
        handle[obj] = Element(c.Value(obj._C_typename, obj.name))

    def push_array_on_stack(self, scope, obj):
        """Define an Array on the stack."""
        handle = self.stack.setdefault(scope, OrderedDict())

        if obj in flatten(self.stack.values()):
            return

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


@dle_pass
def insert_defs(iet):
    """
    Transform the input IET inserting the necessary symbol declarations.
    Declarations are placed as close as possible to the first symbol occurrence.

    Parameters
    ----------
    iet : Callable
        The input Iteration/Expression tree.
    """
    # Classify and then schedule declarations to stack/heap
    allocator = Allocator()
    for k, v in MapExprStmts().visit(iet).items():
        if k.is_Expression:
            if k.is_definition:
                # On the stack
                site = v[-1] if v else iet
                allocator.push_scalar_on_stack(site, k)
                continue
            objs = [k.write]
        elif k.is_Call:
            objs = k.arguments

        for i in objs:
            try:
                if i.is_LocalObject:
                    # On the stack
                    site = v[-1] if v else iet
                    allocator.push_object_on_stack(site, i)
                elif i.is_Array:
                    if i in iet.parameters:
                        # The Array is passed as a Callable argument
                        continue
                    elif i._mem_stack:
                        # On the stack
                        allocator.push_array_on_stack(iet, i)
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
        iet = iet._rebuild(body=List(header=decls+allocs, body=iet.body, footer=frees))

    return iet, {}


@dle_pass
def insert_casts(iet):
    """
    Transform the input IET inserting the necessary type casts.
    The type casts are placed at the top of the IET.

    Parameters
    ----------
    iet : Callable
        The input Iteration/Expression tree.
    """
    # Make the generated code less verbose: if a non-Array parameter does not
    # appear in any Expression, that is, if the parameter is merely propagated
    # down to another Call, then there's no need to cast it
    exprs = FindNodes(Expression).visit(iet)
    need_cast = {i for i in set().union(*[i.functions for i in exprs]) if i.is_Tensor}
    need_cast.update({i for i in iet.parameters if i.is_Array})

    casts = tuple(ArrayCast(i) for i in iet.parameters if i in need_cast)
    iet = iet._rebuild(body=casts + iet.body)

    return iet, {}

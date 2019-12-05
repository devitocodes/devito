"""
Collection of passes for the declaration, allocation, movement and deallocation
of symbols and data.
"""

from collections import OrderedDict

import cgen as c

from devito.ir import (ArrayCast, Element, Expression, List, LocalExpression,
                       FindNodes, MapExprStmts, Transformer)
from devito.symbolics import ccode
from devito.targets.common.engine import target_pass
from devito.tools import flatten

__all__ = ['DataManager']


class Storage(object):

    def __init__(self):
        # Storage with high bandwitdh
        self._high_bw_mem = OrderedDict()

        # Storage with low latency
        self._low_lat_mem = OrderedDict()

    @property
    def _on_low_lat_mem(self):
        ret = []
        for k, v in self._low_lat_mem.items():
            try:
                ret.append((k, [i for i in v.values() if i is not None]))
            except AttributeError:
                ret.append((k, v))
        return ret

    @property
    def _on_high_bw_mem(self):
        return self._high_bw_mem.values()


class DataManager(object):

    def _alloc_object_on_low_lat_mem(self, scope, obj, storage):
        """Allocate a LocalObject in the low latency memory."""
        handle = storage._low_lat_mem.setdefault(scope, OrderedDict())
        handle[obj] = Element(c.Value(obj._C_typename, obj.name))

    def _alloc_array_on_low_lat_mem(self, scope, obj, storage):
        """Allocate an Array in the low latency memory."""
        handle = storage._low_lat_mem.setdefault(scope, OrderedDict())

        if obj in flatten(storage._low_lat_mem.values()):
            return

        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        alignment = "__attribute__((aligned(%d)))" % obj._data_alignment
        value = "%s%s %s" % (obj.name, shape, alignment)
        handle[obj] = Element(c.POD(obj.dtype, value))

    def _alloc_scalar_on_low_lat_mem(self, scope, expr, storage):
        """Allocate a Scalar in the low latency memory."""
        handle = storage._low_lat_mem.setdefault(scope, OrderedDict())

        obj = expr.write
        if obj in handle:
            return

        handle[obj] = None  # Placeholder to avoid reallocation
        storage._low_lat_mem[expr] = LocalExpression(**expr.args)

    def _alloc_array_on_high_bw_mem(self, obj, storage):
        """Allocate an Array in the high bandwidth memory."""
        if obj in storage._high_bw_mem:
            return

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        storage._high_bw_mem[obj] = (decl, alloc, free)

    def _map_function_on_high_bw_mem(self, obj, storage):
        """Place a Function in the high bandwidth memory."""
        return

    @target_pass
    def place_definitions(self, iet):
        """
        Create a new IET with symbols allocated/deallocated in some memory space.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        storage = Storage()

        for k, v in MapExprStmts().visit(iet).items():
            if k.is_Expression:
                if k.is_definition:
                    site = v[-1] if v else iet
                    self._alloc_scalar_on_low_lat_mem(site, k, storage)
                    continue
                objs = [k.write]
            elif k.is_Call:
                objs = k.arguments

            for i in objs:
                try:
                    if i.is_LocalObject:
                        site = v[-1] if v else iet
                        self._alloc_object_on_low_lat_mem(site, i, storage)
                    elif i.is_Array:
                        if i in iet.parameters:
                            # The Array is passed as a Callable argument
                            continue
                        elif i._mem_stack:
                            self._alloc_array_on_low_lat_mem(iet, i, storage)
                        else:
                            self._alloc_array_on_high_bw_mem(i, storage)
                    elif i.is_Function:
                        self._map_function_on_high_bw_mem(i, storage)
                except AttributeError:
                    # E.g., a generic SymPy expression
                    pass

        # Introduce symbol definitions going in the low latency memory
        mapper = dict(storage._on_low_lat_mem)
        iet = Transformer(mapper, nested=True).visit(iet)

        # Introduce symbol definitions going in the high bandwidth memory
        if storage._on_high_bw_mem:
            decls, allocs, frees = zip(*storage._on_high_bw_mem)
            body = List(header=decls + allocs, body=iet.body, footer=frees)
            iet = iet._rebuild(body=body)

        return iet, {}

    @target_pass
    def place_casts(self, iet):
        """
        Create a new IET with the necessary type casts.

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

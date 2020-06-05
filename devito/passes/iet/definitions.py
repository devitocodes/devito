"""
Collection of passes for the declaration, allocation, movement and deallocation
of symbols and data.
"""

from collections import OrderedDict

import cgen as c

from devito.ir import (ArrayCast, Element, List, LocalExpression, FindSymbols,
                       MapExprStmts, Transformer)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import ccode
from devito.tools import as_tuple

__all__ = ['DataManager', 'Storage']


class Storage(object):

    def __init__(self):
        self._high_bw_mem = OrderedDict()
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

        if obj in handle:
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

    def _alloc_array_on_high_bw_mem(self, scope, obj, storage):
        """Allocate an Array in the high bandwidth memory."""
        handle = storage._high_bw_mem.setdefault(scope, OrderedDict())

        if obj in handle:
            return

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        handle[obj] = (decl, alloc, free)

    def _dump_storage(self, iet, storage):
        mapper = {}

        # Introduce symbol definitions going into the low latency memory
        mapper.update(dict(storage._on_low_lat_mem))

        # Introduce symbol definitions going into the high bandwidth memory
        for scope, handle in storage._high_bw_mem.items():
            header = []
            footer = []
            for decl, alloc, free in handle.values():
                if decl is None:
                    header.append(alloc)
                else:
                    header.extend([decl, alloc])
                footer.append(free)
            if header or footer:
                header.append(c.Line())
                footer.insert(0, c.Line())
                body = List(header=header,
                            body=as_tuple(mapper.get(scope)) + scope.body,
                            footer=footer)
                mapper[scope] = scope._rebuild(body=body, **scope.args_frozen)

        processed = Transformer(mapper, nested=True).visit(iet)

        return processed

    @iet_pass
    def place_definitions(self, iet, **kwargs):
        """
        Create a new IET with symbols allocated/deallocated in some memory space.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        storage = Storage()

        already_defined = list(iet.parameters)

        for k, v in MapExprStmts().visit(iet).items():
            if k.is_Expression:
                if k.is_definition:
                    site = v[-1] if v else iet
                    self._alloc_scalar_on_low_lat_mem(site, k, storage)
                    continue
                objs = [k.write]
            elif k.is_Dereference:
                already_defined.append(k.array0)
                objs = [k.array1]
            elif k.is_Call:
                objs = k.arguments

            for i in objs:
                try:
                    if i.is_LocalObject:
                        site = v[-1] if v else iet
                        self._alloc_object_on_low_lat_mem(site, i, storage)
                    elif i.is_Array:
                        if i in already_defined:
                            # The Array is passed as a Callable argument
                            continue

                        site = iet
                        if i._mem_local:
                            # If inside a ParallelRegion, make sure we allocate
                            # inside of it
                            for n in v:
                                if n.is_ParallelBlock:
                                    site = n
                                    break
                        if i._mem_heap:
                            self._alloc_array_on_high_bw_mem(site, i, storage)
                        else:
                            self._alloc_array_on_low_lat_mem(site, i, storage)
                except AttributeError:
                    # E.g., a generic SymPy expression
                    pass

        iet = self._dump_storage(iet, storage)

        return iet, {}

    @iet_pass
    def place_casts(self, iet):
        """
        Create a new IET with the necessary type casts.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        functions = FindSymbols().visit(iet)
        need_cast = {i for i in functions if i.is_Tensor}

        # Make the generated code less verbose by avoiding unnecessary casts
        indexed_names = {i.name for i in FindSymbols('indexeds').visit(iet)}
        need_cast = {i for i in need_cast if i.name in indexed_names or i.is_Array}

        casts = tuple(ArrayCast(i) for i in iet.parameters if i in need_cast)
        iet = iet._rebuild(body=casts + iet.body)

        return iet, {}

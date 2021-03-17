"""
Collection of passes for the declaration, allocation, movement and deallocation
of symbols and data.
"""

from collections import OrderedDict, namedtuple
from functools import singledispatch
from operator import itemgetter

import cgen as c

from devito.ir import (EntryFunction, List, LocalExpression, FindSymbols,
                       MapExprStmts, Transformer)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.misc import is_on_device
from devito.symbolics import ccode
from devito.tools import as_mapper, filter_sorted, flatten
from devito.types import DeviceRM

__all__ = ['DataManager', 'DeviceAwareDataManager', 'Storage']


MetaSite = namedtuple('Definition', 'allocs frees pallocs pfrees')


class Storage(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(Storage, self).__init__(*args, **kwargs)
        self.defined = set()

    def update(self, key, site, **kwargs):
        if key in self.defined:
            return

        try:
            metasite = self[site]
        except KeyError:
            metasite = self.setdefault(site, MetaSite([], [], [], []))

        for k, v in kwargs.items():
            getattr(metasite, k).append(v)

        self.defined.add(key)

    def map(self, key, k, v):
        if key in self.defined:
            return

        self[k] = v
        self.defined.add(key)


class DataManager(object):

    lang = LangBB
    """
    The language used to express data allocations, deletions, and host-device movements.
    """

    def __init__(self, sregistry, *args):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET.
        """
        self.sregistry = sregistry

    def _alloc_object_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate a LocalObject in the low latency memory.
        """
        storage.update(obj, site, allocs=c.Value(obj._C_typename, obj.name))

    def _alloc_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array in the low latency memory.
        """
        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        alignment = self.lang['aligned'](obj._data_alignment)
        value = "%s%s %s" % (obj.name, shape, alignment)

        storage.update(obj, site, allocs=c.POD(obj.dtype, value))

    def _alloc_scalar_on_low_lat_mem(self, site, expr, storage):
        """
        Allocate a Scalar in the low latency memory.
        """
        key = (site, expr.write)  # Ensure a scalar isn't redeclared in the given site
        storage.map(key, expr, LocalExpression(**expr.args))

    def _alloc_array_on_high_bw_mem(self, site, obj, storage, *args):
        """
        Allocate an Array in the high bandwidth memory.
        """
        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        size = "sizeof(%s%s)" % (obj._C_typedata, shape)
        alloc = c.Statement(self.lang['alloc-host'](obj.name, obj._data_alignment, size))

        free = c.Statement(self.lang['free-host'](obj.name))

        storage.update(obj, site, allocs=(decl, alloc), frees=free)

    def _alloc_object_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array of Objects in the low latency memory.
        """
        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        decl = "%s%s" % (obj.name, shape)

        storage.update(obj, site, allocs=c.Value(obj._C_typedata, decl))

    def _alloc_pointed_array_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate the following objects in the high bandwidth memory:

            * The pointer array `obj`;
            * The pointee Array `obj.array`

        If the pointer array is defined over `sregistry.threadid`, that is a thread
        Dimension, then each `obj.array` slice is allocated and freed individually
        by the owner thread.
        """
        # The pointer array
        decl = "**%s" % obj.name
        decl = c.Value(obj._C_typedata, decl)

        size = 'sizeof(%s*)*%s' % (obj._C_typedata, obj.dim.symbolic_size)
        alloc0 = c.Statement(self.lang['alloc-host'](obj.name, obj._data_alignment, size))
        free0 = c.Statement(self.lang['free-host'](obj.name))

        # The pointee Array
        pobj = '%s[%s]' % (obj.name, obj.dim.name)
        shape = "".join("[%s]" % i for i in obj.array.symbolic_shape)
        size = "sizeof(%s%s)" % (obj._C_typedata, shape)
        alloc1 = c.Statement(self.lang['alloc-host'](pobj, obj._data_alignment, size))
        free1 = c.Statement(self.lang['free-host'](pobj))

        if obj.dim is self.sregistry.threadid:
            storage.update(obj, site, allocs=(decl, alloc0), frees=free0,
                           pallocs=(obj.dim, alloc1), pfrees=(obj.dim, free1))
        else:
            storage.update(obj, site, allocs=(decl, alloc0, alloc1), frees=(free0, free1))

    def _dump_storage(self, iet, storage):
        mapper = {}
        for k, v in storage.items():
            # Expr -> LocalExpr ?
            if k.is_Expression:
                mapper[k] = v
                continue

            # allocs/pallocs
            allocs = flatten(v.allocs)
            for tid, body in as_mapper(v.pallocs, itemgetter(0), itemgetter(1)).items():
                header = self.lang.Region._make_header(tid.symbolic_size)
                init = c.Initializer(c.Value(tid._C_typedata, tid.name),
                                     self.lang['thread-num'])
                allocs.append(c.Module((header, c.Block([init] + body))))
            if allocs:
                allocs.append(c.Line())

            # frees/pfrees
            frees = []
            for tid, body in as_mapper(v.pfrees, itemgetter(0), itemgetter(1)).items():
                header = self.lang.Region._make_header(tid.symbolic_size)
                init = c.Initializer(c.Value(tid._C_typedata, tid.name),
                                     self.lang['thread-num'])
                frees.append(c.Module((header, c.Block([init] + body))))
            frees.extend(flatten(v.frees))
            if frees:
                frees.insert(0, c.Line())

            mapper[k] = k._rebuild(body=List(header=allocs, body=k.body, footer=frees),
                                   **k.args_frozen)

        processed = Transformer(mapper, nested=True).visit(iet)

        return processed

    @iet_pass
    def place_definitions(self, iet, **kwargs):
        """
        Create a new IET where all symbols have been declared, allocated, and
        deallocated in one or more memory spaces.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        storage = Storage()

        refmap = FindSymbols().visit(iet).mapper
        placed = list(iet.parameters)

        for k, v in MapExprStmts().visit(iet).items():
            if k.is_LocalExpression:
                placed.append(k.write)
                objs = []
            elif k.is_Expression:
                if k.is_definition:
                    site = v[-1] if v else iet
                    self._alloc_scalar_on_low_lat_mem(site, k, storage)
                    continue
                objs = [k.write]
            elif k.is_Dereference:
                placed.append(k.pointee)
                if k.pointer in placed:
                    objs = []
                else:
                    objs = [k.pointer]
            elif k.is_Call:
                objs = list(k.functions)
                if k.retobj is not None:
                    objs.append(k.retobj.function)
            elif k.is_PointerCast:
                placed.append(k.function)
                objs = []

            for i in objs:
                if i in placed:
                    continue

                try:
                    if i.is_LocalObject:
                        # LocalObject's get placed as close as possible to
                        # their first occurrence
                        site = iet
                        for n in v:
                            if i in refmap[n]:
                                break
                            site = n
                        self._alloc_object_on_low_lat_mem(site, i, storage)
                    elif i.is_Array:
                        # Arrays get placed at the top of the IET
                        if i._mem_heap:
                            self._alloc_array_on_high_bw_mem(iet, i, storage)
                        else:
                            self._alloc_array_on_low_lat_mem(iet, i, storage)
                    elif i.is_ObjectArray:
                        # ObjectArrays get placed at the top of the IET
                        self._alloc_object_array_on_low_lat_mem(iet, i, storage)
                    elif i.is_PointerArray:
                        # PointerArrays get placed at the top of the IET
                        self._alloc_pointed_array_on_high_bw_mem(iet, i, storage)
                except AttributeError:
                    # E.g., a generic SymPy expression
                    pass

        iet = self._dump_storage(iet, storage)

        return iet, {}

    @iet_pass
    def map_onmemspace(self, iet, **kwargs):
        """
        Create a new IET where certain symbols have been mapped to one or more
        extra memory spaces. This may or may not be required depending on the
        underlying architecture.
        """
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
        symbol_names = {i.name for i in FindSymbols('free-symbols').visit(iet)}
        need_cast = {i for i in need_cast if i.name in symbol_names}

        casts = tuple(self.lang.PointerCast(i) for i in iet.parameters if i in need_cast)
        if casts:
            casts = (List(body=casts, footer=c.Line()),)

        iet = iet._rebuild(body=casts + iet.body)

        return iet, {}

    def process(self, graph):
        """
        Apply the `map_on_memspace`, `place_definitions` and `place_casts` passes.
        """
        self.map_onmemspace(graph)
        self.place_definitions(graph)
        self.place_casts(graph)


class DeviceAwareDataManager(DataManager):

    def __init__(self, sregistry, options):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET.
        options : dict
            The optimization options.
            Accepted: ['gpu-fit'].
            * 'gpu-fit': an iterable of `Function`s that are guaranteed to fit
              in the device memory. By default, all `Function`s except saved
              `TimeFunction`'s are assumed to fit in the device memory.
        """
        super().__init__(sregistry)
        self.gpu_fit = options['gpu-fit']

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        _storage = Storage()
        super()._alloc_array_on_high_bw_mem(site, obj, _storage)

        allocs = _storage[site].allocs + [self.lang._map_alloc(obj)]
        frees = [self.lang._map_delete(obj)] + _storage[site].frees
        storage.update(obj, site, allocs=allocs, frees=frees)

    def _map_function_on_high_bw_mem(self, site, obj, storage, devicerm, read_only=False):
        """
        Place a Function in the high bandwidth memory.
        """
        alloc = self.lang._map_to(obj)

        if read_only is False:
            free = c.Collection([self.lang._map_update(obj),
                                 self.lang._map_release(obj, devicerm=devicerm)])
        else:
            free = self.lang._map_delete(obj, devicerm=devicerm)

        storage.update(obj, site, allocs=alloc, frees=free)

    @iet_pass
    def map_onmemspace(self, iet, **kwargs):

        @singledispatch
        def _map_onmemspace(iet):
            return iet, {}

        @_map_onmemspace.register(EntryFunction)
        def _(iet):
            # Special symbol which gives user code control over data deallocations
            devicerm = DeviceRM()

            # Collect written and read-only symbols
            writes = set()
            reads = set()
            for i, v in MapExprStmts().visit(iet).items():
                if not i.is_Expression:
                    # No-op
                    continue
                if not any(isinstance(j, self.lang.DeviceIteration) for j in v):
                    # Not an offloaded Iteration tree
                    continue
                if i.write.is_DiscreteFunction:
                    writes.add(i.write)
                reads.update({r for r in i.reads if r.is_DiscreteFunction})

            # Populate `storage`
            storage = Storage()
            for i in filter_sorted(writes):
                if is_on_device(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage, devicerm)
            for i in filter_sorted(reads - writes):
                if is_on_device(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage, devicerm, True)

            iet = self._dump_storage(iet, storage)

            return iet, {'args': devicerm}

        return _map_onmemspace(iet)

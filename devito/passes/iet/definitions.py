"""
Collection of passes for the declaration, allocation, transfer and deallocation
of symbols and data.
"""

from collections import OrderedDict
from functools import singledispatch
from operator import itemgetter

import numpy as np

from devito.ir import (Block, Call, Definition, DeviceCall, DeviceFunction,
                       DummyExpr, Return, EntryFunction, FindSymbols, MapExprStmts,
                       Transformer, make_callable)
from devito.passes.iet.engine import iet_pass, iet_visit
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.misc import is_on_device
from devito.symbolics import (Byref, DefFunction, FieldFromPointer, IndexedPointer,
                              ListInitializer, SizeOf, VOID, Keyword, ccode)
from devito.tools import as_mapper, as_tuple, filter_sorted, flatten
from devito.types import DeviceRM, Symbol
from devito.types.dense import AliasFunction

__all__ = ['DataManager', 'DeviceAwareDataManager', 'Storage']


class MetaSite(object):

    _items = ('allocs', 'objs', 'frees', 'pallocs', 'pfrees',
              'maps', 'unmaps', 'efuncs')

    def __init__(self):
        for i in self._items:
            setattr(self, i, [])


class Storage(OrderedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.defined = set()

    def update(self, key, site, **kwargs):
        if key in self.defined:
            return

        try:
            metasite = self[site]
        except KeyError:
            metasite = self.setdefault(site, MetaSite())

        for k, v in kwargs.items():
            getattr(metasite, k).append(v)

        self.defined.add(key)

    def map(self, key, site, k, v):
        site = as_tuple(site)
        assert site

        # Is `key` already defined in an outer scope?
        if any((i, key) in self.defined for i in site):
            return

        self[k] = v

        self.defined.add((site[-1], key))


class DataManager(object):

    lang = LangBB
    """
    The language used to express data allocations, deletions, and host-device transfers.
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
        decl = Definition(obj, cargs=obj.cargs)

        if obj._C_init:
            definition = (decl, obj._C_init)
        else:
            definition = (decl)

        frees = obj._C_free

        storage.update(obj, site, objs=definition, frees=frees)

    def _alloc_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array in the low latency memory.
        """
        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        alignment = self.lang['aligned'](obj._data_alignment)
        if obj.initvalue is None:
            initvalue = None
        else:
            initvalue = ListInitializer(obj.initvalue)
        alloc = Definition(obj, shape=shape, qualifier=alignment, initvalue=initvalue)

        storage.update(obj, site, allocs=alloc)

    def _alloc_scalar_on_low_lat_mem(self, site, expr, storage):
        """
        Allocate a Scalar in the low latency memory.
        """
        storage.map(expr.write, site, expr, expr._rebuild(init=True))

    def _alloc_host_array_on_high_bw_mem(self, site, obj, storage, *args):
        """
        Allocate a host Array in the host high bandwidth memory.
        """
        decl = Definition(obj)

        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        nbytes = SizeOf(obj._C_typedata)*obj.size
        alloc = self.lang['host-alloc'](memptr, alignment, nbytes)

        free = self.lang['host-free'](obj._C_symbol)

        storage.update(obj, site, allocs=(decl, alloc), frees=free)

    def _alloc_local_array_on_high_bw_mem(self, site, obj, storage, *args):
        """
        Allocate a local Array in the host high bandwidth memory.
        """
        self._alloc_host_array_on_high_bw_mem(site, obj, storage, *args)

    def _alloc_mapped_array_on_high_bw_mem(self, site, obj, storage, *args):
        """
        Allocate a mapped Array in the host high bandwidth memory.
        """
        decl = Definition(obj)

        # Allocating a mapped Array on the high bandwidth memory requires
        # multiple statements, hence we implement it as a generic Callable
        # to minimize code size, since different arrays will ultimately be
        # able to reuse the same abstract Callable

        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        nbytes = SizeOf(obj._C_typedata)
        alloc0 = self.lang['host-alloc'](memptr, alignment, nbytes)

        nbytes_param = Symbol(name='nbytes', dtype=np.uint64, is_const=True)
        nbytes_arg = SizeOf(obj.indexed._C_typedata)*obj.size

        ffp1 = FieldFromPointer(obj._C_field_data, obj._C_symbol)
        memptr = VOID(Byref(ffp1), '**')
        alloc1 = self.lang['host-alloc'](memptr, alignment, nbytes_param)

        ffp0 = FieldFromPointer(obj._C_field_nbytes, obj._C_symbol)
        init = DummyExpr(ffp0, nbytes_param)

        free0 = self.lang['host-free'](ffp1)

        free1 = self.lang['host-free'](obj._C_symbol)

        ret = Return(obj._C_symbol)

        name = self.sregistry.make_name(prefix='alloc')
        body = (decl, alloc0, alloc1, init, ret)
        efunc0 = make_callable(name, body, retval=obj._C_typename)
        assert len(efunc0.parameters) == 1  # `nbytes_param`
        alloc = Call(name, nbytes_arg, retobj=obj)

        name = self.sregistry.make_name(prefix='free')
        efunc1 = make_callable(name, (free0, free1))
        assert len(efunc1.parameters) == 1  # `obj`
        free = Call(name, obj)

        storage.update(obj, site, allocs=alloc, frees=free, efuncs=(efunc0, efunc1))

    def _alloc_object_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array of Objects in the low latency memory.
        """
        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        decl = Definition(obj, shape=shape)

        storage.update(obj, site, allocs=decl)

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
        decl = Definition(obj)

        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        nbytes = SizeOf(Keyword('%s*' % obj._C_typedata))*obj.dim.symbolic_size
        alloc0 = self.lang['host-alloc'](memptr, alignment, nbytes)

        free0 = self.lang['host-free'](obj._C_symbol)

        # The pointee Array
        pobj = IndexedPointer(obj._C_symbol, obj.dim)
        memptr = VOID(Byref(pobj), '**')
        nbytes = SizeOf(obj._C_typedata)*obj.array.size
        alloc1 = self.lang['host-alloc'](memptr, alignment, nbytes)

        free1 = self.lang['host-free'](pobj)

        # Dump
        if obj.dim is self.sregistry.threadid:
            storage.update(obj, site, allocs=(decl, alloc0), frees=free0,
                           pallocs=(obj.dim, alloc1), pfrees=(obj.dim, free1))
        else:
            storage.update(obj, site, allocs=(decl, alloc0, alloc1), frees=(free0, free1))

    def _inject_definitions(self, iet, storage):
        efuncs = []
        mapper = {}
        for k, v in storage.items():
            # Expr -> LocalExpr ?
            if k.is_Expression:
                mapper[k] = v
                continue

            # objects
            objs = flatten(v.objs)

            # allocs/pallocs
            allocs = flatten(v.allocs)
            for tid, body in as_mapper(v.pallocs, itemgetter(0), itemgetter(1)).items():
                header = self.lang.Region._make_header(tid.symbolic_size)
                init = self.lang['thread-num'](retobj=tid)
                allocs.append(Block(header=header, body=[init] + body))

            # frees/pfrees
            frees = []
            for tid, body in as_mapper(v.pfrees, itemgetter(0), itemgetter(1)).items():
                header = self.lang.Region._make_header(tid.symbolic_size)
                init = self.lang['thread-num'](retobj=tid)
                frees.append(Block(header=header, body=[init] + body))
            frees.extend(flatten(v.frees))

            # efuncs
            efuncs.extend(v.efuncs)

            mapper[k.body] = k.body._rebuild(allocs=allocs, objs=objs, frees=frees)

        processed = Transformer(mapper, nested=True).visit(iet)

        return processed, flatten(efuncs)

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
        # Process inline definitions
        storage = Storage()

        for k, v in MapExprStmts().visit(iet).items():
            if k.is_Expression and k.is_initializable:
                self._alloc_scalar_on_low_lat_mem((iet,) + v, k, storage)

        iet, _ = self._inject_definitions(iet, storage)

        # Process all other definitions, essentially all temporary objects
        # created by the compiler up to this point (Array, LocalObject, etc.)
        storage = Storage()
        defines = FindSymbols('defines-aliases').visit(iet)

        for i in FindSymbols().visit(iet):
            if i in defines:
                continue
            elif i.is_LocalObject:
                self._alloc_object_on_low_lat_mem(iet, i, storage)
            elif i.is_Array:
                if i._mem_heap:
                    if i._mem_host:
                        self._alloc_host_array_on_high_bw_mem(iet, i, storage)
                    elif i._mem_local:
                        self._alloc_local_array_on_high_bw_mem(iet, i, storage)
                    else:
                        self._alloc_mapped_array_on_high_bw_mem(iet, i, storage)
                else:
                    self._alloc_array_on_low_lat_mem(iet, i, storage)
            elif i.is_ObjectArray:
                self._alloc_object_array_on_low_lat_mem(iet, i, storage)
            elif i.is_PointerArray:
                self._alloc_pointed_array_on_high_bw_mem(iet, i, storage)

        iet, efuncs = self._inject_definitions(iet, storage)

        return iet, {'efuncs': efuncs}

    @iet_pass
    def place_casts(self, iet, **kwargs):
        """
        Create a new IET with the necessary type casts.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        # Candidates
        indexeds = FindSymbols('indexeds|indexedbases').visit(iet)

        # Create Function -> n-dimensional array casts
        # E.g. `float (*u)[.] = (float (*)[.]) u_vec->data`
        # NOTE: a cast is needed only if the underlying data object isn't already
        # defined inside the kernel, which happens, for example, when:
        # (i) Dereferencing a PointerArray, e.g., `float (*r0)[.] = (float(*)[.]) pr0[.]`
        # (ii) Declaring a raw pointer, e.g., `float * r0 = NULL; *malloc(&(r0), ...)
        defines = set(FindSymbols('defines').visit(iet))
        bases = sorted({i.base for i in indexeds}, key=lambda i: i.name)
        casts = [self.lang.PointerCast(i.function, obj=i) for i in bases
                 if i not in defines]

        # Incorporate the newly created casts
        if casts:
            iet = iet._rebuild(body=iet.body._rebuild(casts=casts))

        return iet, {}

    def process(self, graph):
        """
        Apply the `place_definitions` and `place_casts` passes.
        """
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

    def _alloc_local_array_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate a local Array in the device high bandwidth memory.
        """
        deviceid = DefFunction(self.lang['device-get'].name)
        doalloc = self.lang['device-alloc']
        dofree = self.lang['device-free']

        nbytes = SizeOf(obj._C_typedata)*obj.size
        init = doalloc(nbytes, deviceid, retobj=obj)

        free = dofree(obj._C_name, deviceid)

        storage.update(obj, site, allocs=init, frees=free)

    def _map_array_on_high_bw_mem(self, site, obj, storage):
        """
        Map an Array already defined in the host memory in to the device high
        bandwidth memory.
        """
        # If Array gets allocated directly in the device memory, there's nothing to map
        if not obj._mem_mapped:
            return

        mmap = self.lang._map_alloc(obj)
        unmap = self.lang._map_delete(obj)

        storage.update(obj, site, maps=mmap, unmaps=unmap)

    def _map_function_on_high_bw_mem(self, site, obj, storage, devicerm, read_only=False):
        """
        Map a Function already defined in the host memory in to the device high
        bandwidth memory.

        Notes
        -----
        In essence, the difference between `_map_function_on_high_bw_mem` and
        `_map_array_on_high_bw_mem` is that the former triggers a data transfer to
        synchronize the host and device copies, while the latter does not.
        """
        mmap = self.lang._map_to(obj)

        if read_only is False:
            unmap = [self.lang._map_update(obj),
                     self.lang._map_release(obj, devicerm=devicerm)]
        else:
            unmap = self.lang._map_delete(obj, devicerm=devicerm)

        storage.update(obj, site, maps=mmap, unmaps=unmap)

    def _dump_transfers(self, iet, storage):
        mapper = {}
        for k, v in storage.items():
            if v.maps or v.unmaps:
                mapper[iet.body] = iet.body._rebuild(maps=flatten(v.maps),
                                                     unmaps=flatten(v.unmaps))

        processed = Transformer(mapper, nested=True).visit(iet)

        return processed

    @iet_visit
    def derive_transfers(self, iet):
        """
        Collect all symbols that cause host-device data transfer, distinguishing
        between reads and writes.
        """

        def needs_transfer(f):
            return (f._mem_mapped and
                    not isinstance(f, AliasFunction) and
                    is_on_device(f, self.gpu_fit))

        writes = set()
        reads = set()
        for i, v in MapExprStmts().visit(iet).items():
            if not any(isinstance(j, self.lang.DeviceIteration) for j in v) and \
               not isinstance(i, DeviceCall) and \
               not isinstance(iet, DeviceFunction):
                # Not an offloaded Iteration tree
                continue

            writes.update({w for w in i.writes if needs_transfer(w)})
            reads.update({f for f in i.functions
                          if needs_transfer(f) and f not in writes})

        return (reads, writes)

    @iet_pass
    def place_transfers(self, iet, **kwargs):
        """
        Create a new IET with host-device data transfers. This requires mapping
        symbols to the suitable memory spaces.
        """

        @singledispatch
        def _place_transfers(iet, mapper):
            return iet, {}

        @_place_transfers.register(EntryFunction)
        def _(iet, mapper):
            try:
                reads, writes = list(zip(*mapper.values()))
            except ValueError:
                return iet, {}
            reads = set(flatten(reads))
            writes = set(flatten(writes))

            # Special symbol which gives user code control over data deallocations
            devicerm = DeviceRM()

            storage = Storage()
            for i in filter_sorted(writes):
                if i.is_Array:
                    self._map_array_on_high_bw_mem(iet, i, storage)
                else:
                    self._map_function_on_high_bw_mem(iet, i, storage, devicerm)
            for i in filter_sorted(reads - writes):
                if i.is_Array:
                    self._map_array_on_high_bw_mem(iet, i, storage)
                else:
                    self._map_function_on_high_bw_mem(iet, i, storage, devicerm, True)

            iet = self._dump_transfers(iet, storage)

            return iet, {}

        return _place_transfers(iet, mapper=kwargs['mapper'])

    def process(self, graph):
        """
        Apply the `place_transfers`, `place_definitions` and `place_casts` passes.
        """
        mapper = self.derive_transfers(graph)
        self.place_transfers(graph, mapper=mapper)
        super().process(graph)

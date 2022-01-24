"""
Collection of passes for the declaration, allocation, transfer and deallocation
of symbols and data.
"""

from collections import OrderedDict, namedtuple
from functools import singledispatch
from operator import itemgetter

import cgen as c

from devito.ir import (Block, Definition, DeviceFunction, EntryFunction, List,
                       PragmaTransfer, FindSymbols, MapExprStmts, Transformer)
from devito.passes.iet.engine import iet_pass, iet_visit
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.misc import is_on_device
from devito.symbolics import (Byref, DefFunction, IndexedPointer, ListInitializer,
                              SizeOf, VOID, Literal, ccode)
from devito.tools import as_mapper, filter_sorted, flatten, prod
from devito.types import DeviceRM
from devito.types.basic import AbstractFunction

__all__ = ['DataManager', 'DeviceAwareDataManager', 'Storage']


MetaSite = namedtuple('Definition', 'allocs frees pallocs pfrees maps unmaps')


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
            metasite = self.setdefault(site, MetaSite([], [], [], [], [], []))

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
        alloc = Definition(obj)

        storage.update(obj, site, allocs=alloc)

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
        key = (site, expr.write)  # Ensure a scalar isn't redeclared in the given site
        storage.map(key, expr, expr._rebuild(init=True))

    def _alloc_array_on_high_bw_mem(self, site, obj, storage, *args):
        """
        Allocate an Array in the high bandwidth memory.
        """
        decl = Definition(obj)

        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        size = SizeOf(obj._C_typedata)*prod(obj.symbolic_shape)
        alloc = self.lang['host-alloc'](memptr, alignment, size)

        free = self.lang['host-free'](obj._C_symbol)

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
        decl = Definition(obj)

        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        size = SizeOf(Literal('%s*' % obj._C_typedata))*obj.dim.symbolic_size
        alloc0 = self.lang['host-alloc'](memptr, alignment, size)

        free0 = self.lang['host-free'](obj._C_symbol)

        # The pointee Array
        pobj = IndexedPointer(obj._C_symbol, obj.dim)
        memptr = VOID(Byref(pobj), '**')
        size = SizeOf(obj._C_typedata)*prod(obj.array.symbolic_shape)
        alloc1 = self.lang['host-alloc'](memptr, alignment, size)

        free1 = self.lang['host-free'](pobj)

        # Dump
        if obj.dim is self.sregistry.threadid:
            storage.update(obj, site, allocs=(decl, alloc0), frees=free0,
                           pallocs=(obj.dim, alloc1), pfrees=(obj.dim, free1))
        else:
            storage.update(obj, site, allocs=(decl, alloc0, alloc1), frees=(free0, free1))

    def _dump_definitions(self, iet, storage):
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
                init = self.lang['thread-num'](retobj=tid)
                allocs.append(Block(header=header, body=[init] + body))

            # frees/pfrees
            frees = []
            for tid, body in as_mapper(v.pfrees, itemgetter(0), itemgetter(1)).items():
                header = self.lang.Region._make_header(tid.symbolic_size)
                init = self.lang['thread-num'](retobj=tid)
                frees.append(Block(header=header, body=[init] + body))
            frees.extend(flatten(v.frees))

            if k is iet:
                mapper[k.body] = k.body._rebuild(allocs=allocs, frees=frees)
            else:
                mapper[k] = k._rebuild(body=List(header=allocs, footer=frees))

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

        # All objects for which a definition is already available
        placed = set(iet.parameters)
        placed.update({i.function for i in placed})

        for k, v in MapExprStmts().visit(iet).items():
            if k.is_Expression:
                if k.is_initializable:
                    site = v[-1] if v else iet
                    self._alloc_scalar_on_low_lat_mem(site, k, storage)
                    continue
                objs = [k.write]
            elif k.is_Dereference:
                placed.add(k.pointee)
                if k.pointer in placed:
                    objs = []
                else:
                    objs = [k.pointer]
            elif k.is_Call:
                objs = list(k.functions)
                if k.retobj is not None:
                    placed.add(k.retobj.function)
            elif k.is_PointerCast:
                placed.add(k.function)
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

        iet = self._dump_definitions(iet, storage)

        return iet, {}

    @iet_visit
    def derive_transfers(self, iet):
        """
        Collect all symbols that cause host-device data transfer, distinguishing
        between reads and writes.
        """
        return ([], [])

    @iet_pass
    def place_transfers(self, iet, **kwargs):
        """
        Create a new IET with host-device data transfers. This requires mapping
        symbols to the suitable memory spaces.
        """
        return iet, {}

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

        # A cast is needed only if the underlying Function object isn't already
        # defined inside the kernel, which happens in:
        # (i) Dereference of PointerArray, e.g., `float (*r0)[.] = (float(*)[.]) pr0[.]`
        # (ii) Linearized mallocs, e.g., `float * r0 = NULL; *malloc(&(r0), ...)
        defines = set(FindSymbols('defines').visit(iet))
        defines = {i._C_name for i in defines}
        needs_cast = lambda f: f.name not in defines and f._C_name != f.name

        # Create Function -> n-dimensional array casts
        # E.g. `float (*u)[.] = (float (*)[.]) u_vec->data`
        functions = sorted({i.function for i in indexeds}, key=lambda i: i.name)
        casts = [self.lang.PointerCast(f) for f in functions if needs_cast(f)]

        # Incorporate the newly created casts
        if casts:
            iet = iet._rebuild(body=iet.body._rebuild(casts=casts))

        return iet, {}

    def process(self, graph):
        """
        Apply the `place_transfers`, `place_definitions` and `place_casts` passes.
        """
        mapper = self.derive_transfers(graph)
        self.place_transfers(graph, mapper=mapper)
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
        if obj._mem_mapped:
            super()._alloc_array_on_high_bw_mem(site, obj, storage)
        else:
            # E.g., use `acc_malloc` or `omp_target_alloc` -- the Array only resides
            # on the device as it never needs to be accessed on the host
            assert obj._mem_local

            deviceid = DefFunction(self.lang['device-get'].name)
            doalloc = self.lang['device-alloc']
            dofree = self.lang['device-free']

            size = SizeOf(obj._C_typedata)*prod(obj.symbolic_shape)
            init = doalloc(size, deviceid, retobj=obj)

            free = dofree(obj._C_name, deviceid)

            storage.update(obj, site, allocs=init, frees=free)

    def _map_array_on_high_bw_mem(self, site, obj, storage):
        """
        Map an Array already defined in the host memory in to the device high
        bandwidth memory.
        """
        # If Array gets allocated directly in the device memory, there's nothing to map
        if obj._mem_local:
            return

        mmap = PragmaTransfer(self.lang._map_alloc, obj)
        unmap = PragmaTransfer(self.lang._map_delete, obj)

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
        mmap = PragmaTransfer(self.lang._map_to, obj)

        if read_only is False:
            unmap = [PragmaTransfer(self.lang._map_update, obj),
                     PragmaTransfer(self.lang._map_release, obj, devicerm=devicerm)]
        else:
            unmap = PragmaTransfer(self.lang._map_delete, obj, devicerm=devicerm)

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

        def needs_transfer(f):
            return (isinstance(f, AbstractFunction) and
                    is_on_device(f, self.gpu_fit) and
                    f._mem_mapped)

        writes = set()
        reads = set()
        for i, v in MapExprStmts().visit(iet).items():
            if not any(isinstance(j, self.lang.DeviceIteration) for j in v) and \
               not isinstance(iet, DeviceFunction):
                # Not an offloaded Iteration tree
                continue

            writes.update({w for w in i.writes if needs_transfer(w)})
            reads.update({f for f in i.functions
                          if needs_transfer(f) and f not in writes})

        return (reads, writes)

    @iet_pass
    def place_transfers(self, iet, **kwargs):

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

            return iet, {'args': devicerm}

        return _place_transfers(iet, mapper=kwargs['mapper'])

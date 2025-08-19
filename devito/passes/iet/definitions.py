"""
Collection of passes for the declaration, allocation, transfer and deallocation
of symbols and data.
"""

from collections import OrderedDict
from ctypes import c_uint64
from functools import singledispatch
from operator import itemgetter

import numpy as np

from devito.ir import (
    Block, Call, Definition, DummyExpr, Iteration, List, Return, EntryFunction,
    FindNodes, FindSymbols, MapExprStmts, Transformer, make_callable
)
from devito.passes import is_gpu_create
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import (
    Byref, DefFunction, FieldFromPointer, IndexedPointer, ListInitializer,
    SizeOf, VOID, pow_to_mul, unevaluate, as_long
)
from devito.tools import as_mapper, as_list, as_tuple, filter_sorted, flatten
from devito.types import (
    Array, ComponentAccess, CustomDimension, Dimension, DeviceMap, DeviceRM,
    Eq, Symbol, size_t
)

__all__ = ['DataManager', 'DeviceAwareDataManager', 'Storage']


class MetaSite:

    _items = ('standalones', 'allocs', 'stacks', 'objs', 'frees', 'pallocs',
              'pfrees', 'maps', 'unmaps', 'efuncs')

    def __init__(self):
        for i in self._items:
            setattr(self, i, [])


class Storage(OrderedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.defined = set()
        self.includes = set()

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

    def include(self, v):
        if v:
            self.includes.add(v)


class DataManager:

    langbb = LangBB
    """
    The language used to express data allocations, deletions, and host-device transfers.
    """

    def __init__(self, rcompile=None, sregistry=None, platform=None,
                 options=None, **kwargs):
        self.rcompile = rcompile
        self.sregistry = sregistry
        self.platform = platform

    def _alloc_object_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate a LocalObject in the low latency memory.
        """
        decl = Definition(obj)

        if obj._C_init:
            definition = (decl, obj._C_init)
        else:
            definition = (decl)

        frees = obj._C_free

        if obj.free_symbols - {obj}:
            storage.update(obj, site, objs=definition, frees=frees)
        else:
            storage.update(obj, site, standalones=definition, frees=frees)

    def _alloc_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array in the low latency memory.
        """
        alloc = Definition(obj)

        storage.update(obj, site, stacks=alloc)

    def _alloc_array_on_global_mem(self, site, obj, storage):
        """
        Allocate an Array in the global memory.
        """
        # Is dynamic initialization required?
        try:
            if all(i.is_Number for i in obj.initvalue):
                return
        except AttributeError:
            return

        # Create input array
        name = '%s_init' % obj.name
        initvalue = np.array([unevaluate(pow_to_mul(i)) for i in obj.initvalue])
        src = Array(name=name, dtype=obj.dtype, dimensions=obj.dimensions,
                    space='host', scope='stack', initvalue=initvalue)

        # Copy input array into global array
        name = self.sregistry.make_name(prefix='init_global')
        nbytes = SizeOf(obj._C_typedata)*as_long(obj.size)
        body = [Definition(src),
                self.langbb['alloc-global-symbol'](obj.indexed, src.indexed, nbytes)]
        efunc = make_callable(name, body)
        alloc = Call(name, efunc.parameters)

        storage.update(obj, site, allocs=alloc, efuncs=efunc)
        storage.include(self.langbb['header-memcpy'])

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
        nbytes = SizeOf(obj._C_typedata)*as_long(obj.size)
        alloc = self.langbb['host-alloc'](memptr, alignment, nbytes)

        free = self.langbb['host-free'](obj._C_symbol)

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

        sizeof_dtypeN = SizeOf(obj.indexed._C_typedata)
        sizeof_dtype1 = SizeOf(obj.c0.indexed._C_typedata)

        # NOTE: the `arity` is calculated such as `sizeof(float3)/sizeof(float)`
        # for portability reasons (since we don't know the size of compound
        # types a priori)
        arity_param = Symbol(name='arity', dtype=size_t)
        arity_arg = sizeof_dtypeN / sizeof_dtype1
        ndims_param = Symbol(name='ndims', dtype=size_t)
        ndims_arg = obj.ndim
        shape_param = Array(name=f'{obj.name}_shape', scope='rvalue',
                            dtype=np.int32 if obj.is_regular else np.uint64,
                            dimensions=(Dimension(name='d'),))
        shape_arg = ListInitializer(obj.c0.symbolic_shape, dtype=shape_param.dtype)

        ffp0 = FieldFromPointer(obj._C_field_data, obj._C_symbol)
        ffp1 = FieldFromPointer(obj._C_field_shape, obj._C_symbol)
        ffp2 = FieldFromPointer(obj._C_field_size, obj._C_symbol)
        ffp3 = FieldFromPointer(obj._C_field_nbytes, obj._C_symbol)
        ffp4 = FieldFromPointer(obj._C_field_arity, obj._C_symbol)

        # Allocate the Array struct
        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        nbytes = SizeOf(obj._C_typedata)
        alloc0 = self.langbb['host-alloc'](memptr, alignment, nbytes)

        # Allocate the shape array
        memptr = VOID(Byref(ffp1), '**')
        nbytes = SizeOf(c_uint64)*ndims_param
        alloc1 = self.langbb['host-alloc'](memptr, alignment, nbytes)

        # Initialize the Array metadata
        dim, = shape_param.dimensions
        init = [DummyExpr(ffp2, 1)]
        init.append(Iteration(
            List(body=(
                DummyExpr(IndexedPointer(ffp1, dim), shape_param[dim]),
                DummyExpr(ffp2, ffp2*shape_param[dim])
            )),
            dim,
            ndims_param - 1,
        ))
        init.append(DummyExpr(ffp3, ffp2*arity_param*sizeof_dtype1))
        init.append(DummyExpr(ffp4, arity_param))

        # Allocate the underlying host data
        memptr = VOID(Byref(ffp0), '**')
        alloc2 = self.langbb['host-alloc-pin'](memptr, alignment, ffp3)

        # Free all of the allocated data
        frees = [self.langbb['host-free-pin'](ffp0),
                 self.langbb['host-free'](ffp1),
                 self.langbb['host-free'](obj._C_symbol)]

        # Allocate the underlying device data, if required by the backend
        alloc_dmap, free_dmap = self._make_dmap_allocfree(obj, ffp3)

        ret = Return(obj._C_symbol)

        # Wrap everything in a Callable so that we can reuse the same code
        # for equivalent Array structs
        name = self.sregistry.make_name(prefix='alloc')
        body = (decl, alloc0, alloc1, *init, alloc2, *as_tuple(alloc_dmap), ret)
        efunc0 = make_callable(name, body, retval=obj)
        args = list(efunc0.parameters)
        args[args.index(arity_param)] = arity_arg
        args[args.index(ndims_param)] = ndims_arg
        args[args.index(shape_param)] = shape_arg
        alloc = Call(name, args, retobj=obj)

        # Same story for the frees
        name = self.sregistry.make_name(prefix='free')
        frees = as_tuple(free_dmap) + as_tuple(frees)
        efunc1 = make_callable(name, frees)
        free = Call(name, efunc1.parameters)

        storage.update(obj, site, allocs=alloc, frees=free, efuncs=(efunc0, efunc1))
        storage.include(self.langbb['header-array'])

    def _alloc_bundle_struct_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate a Bundle struct in the host high bandwidth memory.
        """
        decl = Definition(obj)

        sizeof_dtypeN = SizeOf(obj.indexed._C_typedata)
        sizeof_dtype1 = SizeOf(obj.c0.indexed._C_typedata)

        # NOTE: the `arity` is calculated such as `sizeof(float3)/sizeof(float)`
        # for portability reasons (since we don't know the size of compound
        # types a priori)
        arity_param = Symbol(name='arity', dtype=size_t)
        arity_arg = sizeof_dtypeN / sizeof_dtype1
        ndims_param = Symbol(name='ndims', dtype=size_t)
        ndims_arg = obj.ndim
        shape_param = Array(name=f'{obj.name}_shape', scope='rvalue',
                            dtype=np.int32 if obj.is_regular else np.uint64,
                            dimensions=(Dimension(name='d'),))
        shape_arg = ListInitializer(obj.c0.symbolic_shape, dtype=shape_param.dtype)

        ffp1 = FieldFromPointer(obj._C_field_shape, obj._C_symbol)
        ffp2 = FieldFromPointer(obj._C_field_size, obj._C_symbol)
        ffp3 = FieldFromPointer(obj._C_field_nbytes, obj._C_symbol)
        ffp4 = FieldFromPointer(obj._C_field_arity, obj._C_symbol)

        # Allocate the Bundle struct
        memptr = VOID(Byref(obj._C_symbol), '**')
        alignment = obj._data_alignment
        nbytes = SizeOf(obj._C_typedata)
        alloc0 = self.langbb['host-alloc'](memptr, alignment, nbytes)

        # Allocate the shape array
        memptr = VOID(Byref(ffp1), '**')
        nbytes = SizeOf(c_uint64)*obj.ndim
        alloc1 = self.langbb['host-alloc'](memptr, alignment, nbytes)

        # Initialize the Bundle metadata
        dim, = shape_param.dimensions
        init = [DummyExpr(ffp2, 1)]
        init.append(Iteration(
            List(body=(
                DummyExpr(IndexedPointer(ffp1, dim), shape_param[dim]),
                DummyExpr(ffp2, ffp2*shape_param[dim])
            )),
            dim,
            ndims_param - 1,
        ))
        init.append(DummyExpr(ffp3, ffp2*arity_param*sizeof_dtype1))
        init.append(DummyExpr(ffp4, arity_param))

        # Free all of the allocated data
        frees = [self.langbb['host-free'](ffp1),
                 self.langbb['host-free'](obj._C_symbol)]

        ret = Return(obj._C_symbol)

        # Wrap everything in a Callable so that we can reuse the same code
        # for equivalent Bundle structs
        name = self.sregistry.make_name(prefix='alloc')
        body = (decl, alloc0, alloc1, *init, ret)
        efunc0 = make_callable(name, body, retval=obj)
        args = list(efunc0.parameters)
        args[args.index(arity_param)] = arity_arg
        args[args.index(ndims_param)] = ndims_arg
        args[args.index(shape_param)] = shape_arg
        alloc = Call(name, args, retobj=obj)

        # Same story for the frees
        name = self.sregistry.make_name(prefix='free')
        efunc1 = make_callable(name, frees)
        free = Call(name, efunc1.parameters)

        storage.update(obj, site, allocs=alloc, frees=free, efuncs=(efunc0, efunc1))
        storage.include(self.langbb['header-array'])

    def _alloc_object_array_on_low_lat_mem(self, site, obj, storage):
        """
        Allocate an Array of Objects in the low latency memory.
        """
        decl = Definition(obj)

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
        nbytes = SizeOf(obj._C_typedata, stars='*')*as_long(obj.dim.symbolic_size)
        alloc0 = self.langbb['host-alloc'](memptr, alignment, nbytes)

        free0 = self.langbb['host-free'](obj._C_symbol)

        # The pointee Array
        pobj = IndexedPointer(obj._C_symbol, obj.dim)
        memptr = VOID(Byref(pobj), '**')
        nbytes = SizeOf(obj._C_typedata)*as_long(obj.array.size)
        alloc1 = self.langbb['host-alloc'](memptr, alignment, nbytes)

        free1 = self.langbb['host-free'](pobj)

        # Dump
        if obj.dim is self.sregistry.threadid:
            storage.update(obj, site, allocs=(decl, alloc0), frees=free0,
                           pallocs=(obj.dim, alloc1), pfrees=(obj.dim, free1))
        else:
            storage.update(obj, site, allocs=(decl, alloc0, alloc1), frees=(free0, free1))

    def _make_dmap_allocfree(self, obj, nbytes_param):
        """
        Construct IETs to allocate and free the `dmap` field of a mapped Array.
        """
        return None, None

    def _inject_definitions(self, iet, storage):
        efuncs = []
        mapper = {}
        for k, v in storage.items():
            # Expr -> LocalExpr ?
            if k.is_Expression:
                mapper[k] = v
                continue

            assert k.is_Callable
            cbody = k.body

            # objects
            standalones = as_list(cbody.standalones) + flatten(v.standalones)
            objs = as_list(cbody.objs) + flatten(v.objs)

            # allocs/pallocs
            allocs = as_list(cbody.allocs) + flatten(v.allocs)
            stacks = as_list(cbody.stacks) + flatten(v.stacks)
            for tid, body in as_mapper(v.pallocs, itemgetter(0), itemgetter(1)).items():
                header = self.langbb.Region._make_header(tid.symbolic_size)
                init = self.langbb['thread-num'](retobj=tid)
                allocs.append(Block(header=header, body=[init] + body))

            # frees/pfrees
            frees = []
            for tid, body in as_mapper(v.pfrees, itemgetter(0), itemgetter(1)).items():
                header = self.langbb.Region._make_header(tid.symbolic_size)
                init = self.langbb['thread-num'](retobj=tid)
                frees.append(Block(header=header, body=[init] + body))
            frees.extend(as_list(cbody.frees) + flatten(v.frees))

            # maps/unmaps
            maps = as_list(cbody.maps) + flatten(v.maps)
            unmaps = as_list(cbody.unmaps) + flatten(v.unmaps)

            # efuncs
            efuncs.extend(v.efuncs)

            mapper[cbody] = cbody._rebuild(
                standalones=standalones, allocs=allocs, stacks=stacks,
                maps=maps, objs=objs, unmaps=unmaps, frees=frees
            )

        processed = Transformer(mapper, nested=True).visit(iet)

        return processed, flatten(efuncs)

    @iet_pass
    def place_definitions(self, iet, globs=None, **kwargs):
        """
        Create a new IET where all symbols have been declared, allocated, and
        deallocated in one or more memory spaces.

        Parameters
        ----------
        iet : Callable
            The input Iteration/Expression tree.
        """
        storage = Storage()

        # Process inline definitions
        for k, v in MapExprStmts().visit(iet).items():
            if k.is_Expression and k.is_initializable:
                self._alloc_scalar_on_low_lat_mem((iet,) + v, k, storage)

        iet, _ = self._inject_definitions(iet, storage)

        # Process all other definitions, essentially all temporary objects
        # created by the compiler up to this point (Array, LocalObject, etc.)
        storage = Storage()
        defines = FindSymbols('defines-aliases|globals').visit(iet)

        for i in FindSymbols().visit(iet):
            if i in defines:
                continue

            elif i.is_LocalObject:
                self._alloc_object_on_low_lat_mem(iet, i, storage)

            elif i.is_Bundle:
                if i._mem_heap:
                    if i.is_transient:
                        self._alloc_bundle_struct_on_high_bw_mem(iet, i, storage)
                    elif i._mem_local:
                        self._alloc_local_array_on_high_bw_mem(iet, i, storage)
                    elif i._mem_mapped:
                        self._alloc_mapped_array_on_high_bw_mem(iet, i, storage)
                elif i._mem_stack:
                    self._alloc_array_on_low_lat_mem(iet, i, storage)

            elif i.is_Array:
                if i._mem_heap:
                    if i._mem_host:
                        self._alloc_host_array_on_high_bw_mem(iet, i, storage)
                    elif i._mem_local:
                        self._alloc_local_array_on_high_bw_mem(iet, i, storage)
                    else:
                        self._alloc_mapped_array_on_high_bw_mem(iet, i, storage)
                elif i._mem_stack:
                    self._alloc_array_on_low_lat_mem(iet, i, storage)
                elif globs is not None:
                    # Track, to be handled by the EntryFunction being a global obj!
                    globs.add(i)

            elif i.is_ObjectArray:
                self._alloc_object_array_on_low_lat_mem(iet, i, storage)

            elif i.is_PointerArray:
                self._alloc_pointed_array_on_high_bw_mem(iet, i, storage)

        # Handle postponed global objects
        if isinstance(iet, EntryFunction) and globs:
            for i in sorted(globs, key=lambda f: f.name):
                self._alloc_array_on_global_mem(iet, i, storage)

        iet, efuncs = self._inject_definitions(iet, storage)

        return iet, {'efuncs': efuncs,
                     'globals': as_tuple(globs),
                     'includes': as_tuple(sorted(storage.includes))}

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
        defines = set(FindSymbols('defines|globals').visit(iet))
        bases = sorted({i.base for i in indexeds}, key=lambda i: i.name)

        # Some objects don't distinguish their _C_symbol because they are known,
        # by construction, not to require it, thus making the generated code
        # cleaner. These objects don't need a cast
        bases = [i for i in bases if i.name != i.function._C_name]

        # Create and attach the type casts
        casts = tuple(self.langbb.PointerCast(i.function, obj=i) for i in bases
                      if i not in defines)
        if casts:
            iet = iet._rebuild(body=iet.body._rebuild(casts=casts + iet.body.casts))

        return iet, {}

    def process(self, graph):
        """
        Apply the `place_definitions` and `place_casts` passes.
        """
        self.place_definitions(graph, globs=set())
        self.place_casts(graph)


class DeviceAwareDataManager(DataManager):

    def __init__(self, options=None, **kwargs):
        self.gpu_fit = options['gpu-fit']
        self.gpu_create = options['gpu-create']
        self.pmode = options.get('place-transfers')

        super().__init__(**kwargs)

    def _alloc_local_array_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate a local Array in the device high bandwidth memory.
        """
        deviceid = DefFunction(self.langbb['device-get'].name)
        doalloc = self.langbb['device-alloc']
        dofree = self.langbb['device-free']

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

        mmap = self.langbb._map_alloc(obj)
        unmap = self.langbb._map_delete(obj)

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
        if read_only is False:
            if is_gpu_create(obj, self.gpu_create):
                mmap = self.langbb._map_alloc(obj)

                efuncs, init = make_zero_init(obj, self.rcompile, self.sregistry)

                mmap = (mmap, init)
            else:
                mmap = self.langbb._map_to(obj)
                efuncs = ()

            # Copy back to host memory, release device memory
            unmap = (self.langbb._map_update(obj),
                     self.langbb._map_release(obj, devicerm=devicerm))
        else:
            mmap = self.langbb._map_to(obj)
            efuncs = ()
            unmap = self.langbb._map_delete(obj, devicerm=devicerm)

        storage.update(obj, site, maps=mmap, unmaps=unmap, efuncs=efuncs)

    @iet_pass
    def place_transfers(self, iet, data_movs=None, **kwargs):
        """
        Create a new IET with host-device data transfers. This requires mapping
        symbols to the suitable memory spaces.
        """
        if not self.pmode:
            return iet, {}

        @singledispatch
        def _place_transfers(iet, data_movs):
            return iet, {}

        @_place_transfers.register(EntryFunction)
        def _(iet, data_movs):
            reads, writes = data_movs

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

            iet, efuncs = self._inject_definitions(iet, storage)

            return iet, {'efuncs': efuncs}

        return _place_transfers(iet, data_movs=data_movs)

    @iet_pass
    def place_devptr(self, iet, **kwargs):
        """
        Transform `iet` such that device pointers are used in DeviceCalls.
        """
        defines = FindSymbols('defines').visit(iet)
        dmaps = [i for i in FindSymbols('basics').visit(iet)
                 if isinstance(i, DeviceMap) and i not in defines]

        maps = [self.langbb.PointerCast(i.function, obj=i) for i in dmaps]
        body = iet.body._rebuild(maps=iet.body.maps + tuple(maps))
        iet = iet._rebuild(body=body)

        return iet, {}

    @iet_pass
    def place_bundling(self, iet, **kwargs):
        """
        Transform `iet` adding snippets to pack and unpack Bundles.
        """
        return iet, {}

    def process(self, graph):
        """
        Apply the `place_transfers`, `place_definitions` and `place_casts` passes.
        """
        self.place_transfers(graph, data_movs=graph.data_movs)
        self.place_definitions(graph, globs=set())
        self.place_devptr(graph)
        self.place_bundling(graph, writes_input=graph.writes_input)
        self.place_casts(graph)


def make_zero_init(obj, rcompile, sregistry):
    cdims = []
    for d, (h0, h1), s in zip(obj.dimensions, obj._size_halo, obj.symbolic_shape):
        if d.is_NonlinearDerived:
            assert h0 == h1 == 0
            m = 0
            M = s - 1
        else:
            m = d.symbolic_min - h0
            M = d.symbolic_max + h1
        cdims.append(CustomDimension(name=d.name, parent=d,
                                     symbolic_min=m, symbolic_max=M))

    if obj.is_Bundle:
        eqns = [Eq(ComponentAccess(obj[cdims], i), 0) for i in range(obj.ncomp)]
    else:
        eqns = [Eq(obj[cdims], 0)]

    irs, byproduct = rcompile(eqns)

    init = irs.iet.body.body[0]

    name = sregistry.make_name(prefix='init')
    efunc = make_callable(name, init)
    init = Call(name, efunc.parameters)

    efuncs = [efunc]

    # Also the called device kernels, if any
    calls = [i.name for i in FindNodes(Call).visit(efunc)]
    efuncs.extend([i.root for i in byproduct.funcs if i.root.name in calls])

    return efuncs, init

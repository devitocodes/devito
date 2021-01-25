"""
Collection of passes for the declaration, allocation, movement and deallocation
of symbols and data.
"""

from collections import OrderedDict, namedtuple
from operator import itemgetter

import cgen as c

from devito.ir import (List, LocalExpression, PointerCast, FindSymbols,
                       MapExprStmts, Transformer)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.openmp import Ompizer
from devito.symbolics import ccode
from devito.tools import as_mapper, flatten

__all__ = ['DataManager', 'Storage']


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

    _Parallelizer = Ompizer

    def __init__(self, sregistry):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET (e.g., `sregistry.threadid`, that is the thread
            Dimension, used by the DataManager for parallel memory allocation).
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
        alignment = "__attribute__((aligned(%d)))" % obj._data_alignment
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
        alloc = "posix_memalign((void**)&%s, %d, sizeof(%s%s))"
        alloc = alloc % (obj.name, obj._data_alignment, obj._C_typedata, shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

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

        If the pointer array is defined over `sregistry.threadid`, that it a thread
        Dimension, then each `obj.array` slice is allocated and freed individually
        by the logically-owning thread.
        """
        # The pointer array
        decl = "**%s" % obj.name
        decl = c.Value(obj._C_typedata, decl)

        alloc0 = "posix_memalign((void**)&%s, %d, sizeof(%s*)*%s)"
        alloc0 = alloc0 % (obj.name, obj._data_alignment, obj._C_typedata,
                           obj.dim.symbolic_size)
        alloc0 = c.Statement(alloc0)

        free0 = c.Statement('free(%s)' % obj.name)

        # The pointee Array
        shape = "".join("[%s]" % i for i in obj.array.symbolic_shape)
        alloc1 = "posix_memalign((void**)&%s[%s], %d, sizeof(%s%s))"
        alloc1 = alloc1 % (obj.name, obj.dim.name, obj._data_alignment, obj._C_typedata,
                           shape)
        alloc1 = c.Statement(alloc1)

        free1 = c.Statement('free(%s[%s])' % (obj.name, obj.dim.name))

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
                header = self._Parallelizer._Region._make_header(tid.symbolic_size)
                init = self._Parallelizer._make_tid(tid)
                allocs.append(c.Module((header, c.Block([init] + body))))
            if allocs:
                allocs.append(c.Line())

            # frees/pfrees
            frees = []
            for tid, body in as_mapper(v.pfrees, itemgetter(0), itemgetter(1)).items():
                header = self._Parallelizer._Region._make_header(tid.symbolic_size)
                init = self._Parallelizer._make_tid(tid)
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
                placed.append(k.array)
                if k.parray in placed:
                    objs = []
                else:
                    objs = [k.parray]
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
                        # their first appearence
                        site = iet
                        for n in v:
                            if i in refmap[n]:
                                break
                            site = n
                        self._alloc_object_on_low_lat_mem(site, i, storage)
                    elif i.is_Array:
                        # Array's get placed as far as possible from their
                        # first appearence
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
                    elif i.is_ObjectArray:
                        # ObjectArray's get placed at the top of the IET
                        self._alloc_object_array_on_low_lat_mem(iet, i, storage)
                    elif i.is_PointerArray:
                        # PointerArray's get placed at the top of the IET
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

        casts = tuple(PointerCast(i) for i in iet.parameters if i in need_cast)
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

from functools import singledispatch

import cgen as c

from devito.core.gpu import DeviceNoopOperator, DeviceOperator, DeviceCustomOperator
from devito.ir.iet import EntryFunction, MapExprStmts
from devito.passes.iet import DataManager, DeviceOmpizer, Storage, iet_pass, is_on_device  #TODO: TO BE MOVED...
from devito.tools import filter_sorted
from devito.types import DeviceRM

__all__ = ['DeviceNoopOmpOperator', 'DeviceAdvOmpOperator', 'DeviceCustomOmpOperator']


class DeviceOmpDataManager(DataManager):

    def __init__(self, parallelizer, sregistry, options):
        """
        Parameters
        ----------
        parallelizer : Parallelizer
            Used to implement data movement between host and device as well as
            data mapping, allocation, and deallocation on the device.
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET (e.g., `sregistry.threadid`, that is the thread
            Dimension, used by the DataManager for parallel memory allocation).
        options : dict
            The optimization options.
            Accepted: ['gpu-fit'].
            * 'gpu-fit': an iterable of `Function`s that are guaranteed to fit
              in the device memory. By default, all `Function`s except saved
              `TimeFunction`'s are assumed to fit in the device memory.
        """
        super().__init__(parallelizer, sregistry)
        self.gpu_fit = options['gpu-fit']

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        _storage = Storage()
        super()._alloc_array_on_high_bw_mem(site, obj, _storage)

        allocs = _storage[site].allocs + [self.parallelizer._map_alloc(obj)]
        frees = [self.parallelizer._map_delete(obj)] + _storage[site].frees
        storage.update(obj, site, allocs=allocs, frees=frees)

    def _map_function_on_high_bw_mem(self, site, obj, storage, devicerm, read_only=False):
        """
        Place a Function in the high bandwidth memory.
        """
        alloc = self.parallelizer._map_to(obj)

        if read_only is False:
            free = c.Collection([self.parallelizer._map_update(obj),
                                 self.parallelizer._map_release(obj, devicerm=devicerm)])
        else:
            free = self.parallelizer._map_delete(obj, devicerm=devicerm)

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
                if not any(isinstance(j, self.parallelizer._Iteration) for j in v):
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


# Operators


class DeviceOmpOperatorMixin(object):

    _Parallelizer = DeviceOmpizer
    _DataManager = DeviceOmpDataManager

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        oo = kwargs['options']

        oo.pop('openmp', None)  # It may or may not have been provided
        kwargs = super()._normalize_kwargs(**kwargs)
        oo['openmp'] = True

        return kwargs


class DeviceNoopOmpOperator(DeviceOmpOperatorMixin, DeviceNoopOperator):
    pass


class DeviceAdvOmpOperator(DeviceOmpOperatorMixin, DeviceOperator):
    pass


class DeviceCustomOmpOperator(DeviceOmpOperatorMixin, DeviceCustomOperator):

    _known_passes = DeviceCustomOperator._known_passes + ('openmp',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openmp'] = mapper['parallel']
        return mapper

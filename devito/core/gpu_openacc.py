import cgen as c

from devito.core.gpu import DeviceNoopOperator, DeviceOperator, DeviceCustomOperator
from devito.core.gpu_openmp import DeviceOmpDataManager
from devito.passes.iet import DeviceAccizer
from devito.tools import prod

__all__ = ['DeviceNoopAccOperator', 'DeviceAdvAccOperator', 'DeviceCustomAccOperator']

#TODO: SQUASH gpu.py gpu_openmp.py gpu_openacc.py  ?? just like cpu.py after all...


class DeviceAccDataManager(DeviceOmpDataManager):

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate an Array in the high bandwidth memory.
        """
        if obj._mem_mapped:
            # posix_memalign + copy-to-device
            super()._alloc_array_on_high_bw_mem(site, obj, storage)
        else:
            # acc_malloc -- the Array only resides on the device, ie, it never
            # needs to be accessed on the host
            assert obj._mem_default
            size_trunkated = "".join("[%s]" % i for i in obj.symbolic_shape[1:])
            decl = c.Value(obj._C_typedata, "(*%s)%s" % (obj.name, size_trunkated))
            cast = "(%s (*)%s)" % (obj._C_typedata, size_trunkated)
            size_full = prod(obj.symbolic_shape)
            alloc = "%s acc_malloc(sizeof(%s[%s]))" % (cast, obj._C_typedata, size_full)
            init = c.Initializer(decl, alloc)

            free = c.Statement('acc_free(%s)' % obj.name)

            storage.update(obj, site, allocs=init, frees=free)


# Operators


class DeviceAccOperatorMixin(object):

    _Parallelizer = DeviceAccizer
    _DataManager = DeviceAccDataManager

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        oo = kwargs['options']
        oo.pop('openmp', None)

        kwargs = super()._normalize_kwargs(**kwargs)
        oo['openacc'] = True

        return kwargs


class DeviceNoopAccOperator(DeviceAccOperatorMixin, DeviceNoopOperator):
    pass


class DeviceAdvAccOperator(DeviceAccOperatorMixin, DeviceOperator):
    pass


class DeviceCustomAccOperator(DeviceAccOperatorMixin, DeviceCustomOperator):

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openacc'] = mapper['parallel']
        return mapper

    _known_passes = DeviceCustomOperator._known_passes + ('openacc',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

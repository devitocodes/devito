from devito.core.gpu import DeviceNoopOperator, DeviceAdvOperator, DeviceCustomOperator
from devito.passes.iet import DeviceAccDataManager, DeviceAccizer

__all__ = ['DeviceNoopAccOperator', 'DeviceAdvAccOperator', 'DeviceCustomAccOperator']

#TODO: SQUASH gpu.py gpu_openmp.py gpu_openacc.py  ?? just like cpu.py after all...


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


class DeviceAdvAccOperator(DeviceAccOperatorMixin, DeviceAdvOperator):
    pass


class DeviceCustomAccOperator(DeviceAccOperatorMixin, DeviceCustomOperator):

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openacc'] = mapper['parallel']
        return mapper

    _known_passes = DeviceCustomOperator._known_passes + ('openacc',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

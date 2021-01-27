from devito.core.gpu import DeviceNoopOperator, DeviceAdvOperator, DeviceCustomOperator
from devito.passes.iet import DeviceOmpDataManager, DeviceOmpizer

__all__ = ['DeviceNoopOmpOperator', 'DeviceAdvOmpOperator', 'DeviceCustomOmpOperator']


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


class DeviceAdvOmpOperator(DeviceOmpOperatorMixin, DeviceAdvOperator):
    pass


class DeviceCustomOmpOperator(DeviceOmpOperatorMixin, DeviceCustomOperator):

    _known_passes = DeviceCustomOperator._known_passes + ('openmp',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openmp'] = mapper['parallel']
        return mapper

from collections import namedtuple
from ctypes import POINTER, Structure
from functools import cached_property

from devito.tools import EnrichedTuple, Tag

# Additional Function-related APIs

__all__ = [
    'CELL',
    'NODE',
    'Buffer',
    'DeviceLayer',
    'DimensionTuple',
    'DiskLayer',
    'HierarchyLayer',
    'HostLayer',
    'IgnoreDimSort',
    'Offset',
    'Size',
    'device_layer',
    'disk_layer',
    'host_layer',
]


class Buffer(Tag):

    def __init__(self, value):
        super().__init__('Buffer', value)


class Stagger(Tag):
    """Stagger region."""
    pass

NODE = Stagger('node')  # noqa
CELL = Stagger('cell')


Size = namedtuple('Size', 'left right')
Offset = namedtuple('Offset', 'left right')


class DimensionTuple(EnrichedTuple):

    def __getitem_hook__(self, dim):
        for d in self.getters:
            if d._defines & dim._defines:
                return self.getters[d]
        raise KeyError


class Staggering(DimensionTuple):

    @cached_property
    def on_node(self):
        return not self or all(s == 0 for s in self)

    @property
    def _ref(self):
        if not self:
            return None
        elif self.on_node:
            return NODE
        else:
            return tuple(d for d, s in zip(self.getters, self, strict=True) if s == 1)


class IgnoreDimSort(tuple):
    """A tuple subclass used to wrap the implicit_dims to indicate
    that the topological sort of other dimensions should not occur."""
    pass


class CtypesFactory:

    cache = {}

    @classmethod
    def generate(cls, pname, pfields):
        key = (pname, tuple(pfields))
        try:
            return cls.cache[key]
        except KeyError:
            dtype = POINTER(type(pname, (Structure,), {'_fields_': pfields}))
            return cls.cache.setdefault(key, dtype)


class HierarchyLayer:

    """
    Represent a generic layer of the node storage hierarchy (e.g., disk, host).
    """

    def __init__(self, suffix=''):
        self.suffix = suffix

    def __repr__(self):
        return f"Layer<{self.suffix}>"

    def __eq__(self, other):
        return (isinstance(other, HierarchyLayer) and
                self.suffix == other.suffix)

    def __hash__(self):
        return hash(self.suffix)


class HostLayer(HierarchyLayer):
    pass


class DeviceLayer(HierarchyLayer):
    pass


class DiskLayer(HierarchyLayer):
    pass


host_layer = HostLayer('host')
device_layer = DeviceLayer('device')
disk_layer = DiskLayer('disk')

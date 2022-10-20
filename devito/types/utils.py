from ctypes import POINTER, Structure

from devito.tools import EnrichedTuple, Tag
# Additional Function-related APIs

__all__ = ['Buffer', 'DimensionTuple', 'NODE', 'CELL', 'IgnoreDimSort',
           'HierarchyLayer', 'HostLayer']


class Buffer(Tag):

    def __init__(self, value):
        super(Buffer, self).__init__('Buffer', value)


class Stagger(Tag):
    """Stagger region."""
    pass

NODE = Stagger('node')  # noqa
CELL = Stagger('cell')


class DimensionTuple(EnrichedTuple):

    def __getitem_hook__(self, dim):
        for d in self._getters:
            if d._defines & dim._defines:
                return self._getters[d]
        raise KeyError


class IgnoreDimSort(tuple):
    """A tuple subclass used to wrap the implicit_dims to indicate
    that the topological sort of other dimensions should not occur."""
    pass


class CtypesFactory(object):

    cache = {}

    @classmethod
    def generate(cls, pname, pfields):
        dtype = POINTER(type(pname, (Structure,), {'_fields_': pfields}))
        key = (pname, tuple(pfields))
        return cls.cache.setdefault(key, dtype)


class HierarchyLayer(object):

    """
    Represent a generic layer of the node storage hierarchy (e.g., disk, host).
    """

    def __init__(self, suffix=''):
        self.suffix = suffix

    def __repr__(self):
        return "Layer<%s>" % self.suffix


class HostLayer(HierarchyLayer):
    pass

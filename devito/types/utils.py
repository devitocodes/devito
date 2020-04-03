from devito.tools import EnrichedTuple, Tag
# Additional Function-related APIs

__all__ = ['Buffer', 'DimensionTuple', 'NODE', 'CELL']


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
        for d in dim._defines:
            if d in self._getters:
                return self._getters[d]
        raise KeyError

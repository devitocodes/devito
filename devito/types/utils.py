import sympy
from devito.tools import Tag
# Additional Function-related APIs

__all__ = ['Buffer', 'NODE', 'CELL']


class Buffer(Tag):

    def __init__(self, value):
        super(Buffer, self).__init__('Buffer', value)


class Stagger(Tag):
    """Stagger region."""
    pass

NODE = Stagger('node')  # noqa
CELL = Stagger('cell')

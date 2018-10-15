from devito.ops.utils import namespace
import devito.types as types

__all__ = ['OPSGridObject']

class OPSGridObject(types.LocalObject):

    is_OPSGridObject = True

    dtype = namespace['type-ops_block']

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return '{} {} = {}'.format(self.dtype, self.name, str(self.value))
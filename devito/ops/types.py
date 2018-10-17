from devito.ops.utils import namespace
import devito.types as types

__all__ = ['OPSDeclObject']


class OPSDeclObject(types.LocalObject):

    is_OPSGridObject = True

    def __init__(self, dtype, name, value):
        self.dtype = dtype
        self.name = name
        self.value = value

    # Temporary: for debugging purposes only
    def __str__(self):
        return '{} {} = {}{}'.format(self.dtype, 
                                   self.name, 
                                   type(self.value), 
                                   str(self.value.args))

from ctypes import c_int, c_double, c_void_p

from devito.types import CompositeObject, LocalObject, Symbol

__all__ = ['Timer', 'VoidPointer', 'VolatileInt', 'c_volatile_int',
           'c_volatile_int_p']


class Timer(CompositeObject):

    def __init__(self, name, sections):
        super().__init__(name, 'profiler', [(i, c_double) for i in sections])

    def reset(self):
        for i in self.fields:
            setattr(self.value._obj, i, 0.0)
        return self.value

    @property
    def total(self):
        return sum(getattr(self.value._obj, i) for i in self.fields)

    @property
    def sections(self):
        return self.fields

    def _arg_values(self, args=None, **kwargs):
        values = super()._arg_values(args=args, **kwargs)

        # Reset timer
        for i in self.fields:
            setattr(values[self.name]._obj, i, 0.0)

        return values

    # Pickling support
    _pickle_args = ['name', 'sections']


class VoidPointer(LocalObject):

    dtype = type('void*', (c_void_p,), {})

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']


class VolatileInt(Symbol):

    @property
    def _C_typedata(self):
        return 'volatile int'

    _C_typename = _C_typedata

    @property
    def _C_ctype(self):
        return c_volatile_int


# ctypes subtypes

class c_volatile_int(c_int):
    pass


class c_volatile_int_p(c_void_p):
    pass

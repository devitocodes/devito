from ctypes import c_double

from devito.types import CompositeObject

__all__ = ['Timer']


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

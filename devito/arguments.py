from cached_property import cached_property
import cgen as c
import numpy as np


class RuntimeArgProvider(object):

    @property
    def rtargs(self):
        raise NotImplemented()


class RuntimeArgument(object):

    def __init__(self, name, source, default_value=None):
        self.name = name
        self.source = source
        self._value = self._default_value = default_value

    @property
    def value(self):
        return self._value

    @property
    def ready(self):
        return self._value is not None

    @property
    def decl(self):
        raise NotImplemented()

    def reset(self):
        self._value = self._default_value

    def verify(self, kwargs):
        raise NotImplemented()


class ScalarArgument(RuntimeArgument):

    def __init__(self, name, source, reducer, default_value=None):
        super(ScalarArgument, self).__init__(name, source, default_value)
        self.reducer = reducer

    @property
    def decl(self):
        return c.Value('const int', self.name)

    @property
    def dtype(self):
        return self.source.dtype

    def verify(self, value):
        # Assuming self._value was initialised as appropriate for the reducer
        if value is not None:
            if self._value is not None:
                self._value = self.reducer(self._value, value)
            else:
                self._value = value
        return self._value is not None


class TensorArgument(RuntimeArgument):

    def __init__(self, name, source, dtype):
        super(TensorArgument, self).__init__(name, source)
        self.dtype = dtype
        self._value = self._default_value = self.source

    @property
    def value(self):
        if isinstance(self._value, np.ndarray):
            return self._value
        else:
            return self._value.data

    @property
    def decl(self):
        return c.Value(c.dtype_to_ctype(self.dtype), '*restrict %s_vec' % self.name)

    @property
    def ccast(self):
        alignment = "__attribute__((aligned(64)))"
        shape = ''.join(["[%s]" % i.ccode for i in self.source.indices[1:]])

        cast = c.Initializer(c.POD(self.dtype,
                                   '(*restrict %s)%s %s' % (self.name, shape, alignment)),
                             '(%s (*)%s) %s' % (c.dtype_to_ctype(self.dtype),
                                                shape, '%s_vec' % self.name))
        return cast

    def verify(self, value):
        if value is None:
            value = self._value

        verify = self.source.shape == value.shape

        verify = verify and all([d.verify(v) for d, v in zip(self.source.indices,
                                                             value.shape)])
        if verify:
            self._value = value

        return self._value is not None and verify


class DimensionArgProvider(RuntimeArgProvider):
    reducer = max
    _default_value = None

    def __init__(self, *args, **kwargs):
        super(DimensionArgProvider, self).__init__(*args, **kwargs)
        self._value = self._default_value

    def reset(self):
        self._value = self._default_value

    @property
    def value(self):
        return self._value

    @cached_property
    def rtargs(self):
        # TODO: Create proper Argument objects - with good init values
        if self.size is not None:
            return []
        else:
            size = ScalarArgument("%s_size" % self.name, self, max)
            return [size]

    # TODO: Do I need a verify on a dimension?
    def verify(self, value):
        verify = True

        if value is None and self._value is not None:
            return verify

        if value is not None and value == self._value:
            return verify

        if self.size is not None:
            # Assuming the only people calling my verify are symbolic data, they need to
            # be bigger than my size if I have a hard-coded size
            verify = (value >= self.size)
        else:
            # Assuming self._value was initialised as maxint
            if value is not None and self._value is not None:
                value = self.reducer(self._value, value)
            if hasattr(self, 'parent'):
                verify = verify and self.parent.verify(value)
                # If I don't know my value, ask my parent
                if value is None:
                    value = self.parent.value

            # Derived dimensions could be linked through constraints
            # At this point, a constraint needs to be added that enforces
            # dim_e - dim_s < SOME_MAX
            # Also need a default constraint that dim_e > dim_s (or vice-versa)
            verify = verify and all([a.verify(v) for a, v in zip(self.rtargs, (value,))])
            if verify:
                self._value = value
        return verify


class SymbolicDataArgProvider(RuntimeArgProvider):

    @cached_property
    def rtargs(self):
        # TODO: Create proper Argument objects - with good init values
        return [TensorArgument(self.name, self, self.dtype)]


class ScalarFunctionArgProv(RuntimeArgProvider):

    @cached_property
    def rtargs(self):
        return [ScalarArgument(self.name, self, self.dtype)]


class TensorFunctionArgProv(RuntimeArgProvider):

    @cached_property
    def rtargs(self):
        return [TensorArgument(self.name, self, self.dtype)]

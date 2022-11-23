from ctypes import byref

import sympy

from devito.tools import Pickable, as_tuple, sympy_mutex
from devito.types.args import ArgProvider
from devito.types.caching import Uncached
from devito.types.basic import Basic
from devito.types.utils import CtypesFactory

__all__ = ['Object', 'LocalObject', 'CompositeObject']


class AbstractObject(Basic, sympy.Basic, Pickable):

    """
    Base class for objects with derived type.

    The hierarchy is structured as follows

                         AbstractObject
                                |
                 ---------------------------------
                 |                               |
              Object                       LocalObject
                 |
          CompositeObject

    Warnings
    --------
    AbstractObjects are created and managed directly by Devito.
    """

    is_AbstractObject = True

    __rargs__ = ('name', 'dtype')

    def __new__(cls, *args, **kwargs):
        with sympy_mutex:
            obj = sympy.Basic.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def __init__(self, name, dtype):
        self.name = name
        self._dtype = dtype

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def _hashable_content(self):
        return (self.name, self.dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def free_symbols(self):
        return {self}

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_ctype(self):
        return self.dtype

    @property
    def base(self):
        return self

    @property
    def function(self):
        return self

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Object(AbstractObject, ArgProvider, Uncached):

    """
    Object with derived type defined in Python.
    """

    is_Object = True

    def __init__(self, name, dtype, value=None):
        super(Object, self).__init__(name, dtype)
        self.value = value

    __hash__ = Uncached.__hash__

    @property
    def _mem_external(self):
        return True

    @property
    def _arg_names(self):
        return (self.name,)

    def _arg_defaults(self):
        if callable(self.value):
            return {self.name: self.value()}
        else:
            return {self.name: self.value}

    def _arg_values(self, **kwargs):
        """
        Produce runtime values for this Object after evaluating user input.

        Parameters
        ----------
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        if self.name in kwargs:
            obj = kwargs.pop(self.name)
            return {self.name: obj._arg_defaults()[obj.name]}
        else:
            return self._arg_defaults()


class CompositeObject(Object):

    """
    Object with composite type (e.g., a C struct) defined in Python.
    """

    __rargs__ = ('name', 'pname', 'pfields')

    def __init__(self, name, pname, pfields, value=None):
        dtype = CtypesFactory.generate(pname, pfields)
        value = self.__value_setup__(dtype, value)
        super(CompositeObject, self).__init__(name, dtype, value)

    def __value_setup__(self, dtype, value):
        return value or byref(dtype._type_())

    @property
    def pfields(self):
        return tuple(self.dtype._type_._fields_)

    @property
    def pname(self):
        return self.dtype._type_.__name__

    @property
    def fields(self):
        return [i for i, _ in self.pfields]


class LocalObject(AbstractObject):

    """
    Object with derived type defined inside an Operator.
    """

    is_LocalObject = True

    dtype = None
    """
    LocalObjects encode their dtype as a class attribute.
    """

    __rargs__ = ('name',)
    __rkwargs__ = ('cargs', 'liveness')

    def __init__(self, name, cargs=None, **kwargs):
        self.name = name
        self.cargs = as_tuple(cargs)

        self._liveness = kwargs.get('liveness', 'lazy')
        assert self._liveness in ['eager', 'lazy']

    def _hashable_content(self):
        return super()._hashable_content() + self.cargs + (self.liveness,)

    @property
    def liveness(self):
        return self._liveness

    @property
    def free_symbols(self):
        return super().free_symbols | set(self.cargs)

    @property
    def _C_init(self):
        """
        A symbolic initializer for the LocalObject, injected in the generated code.

        Notes
        -----
        To be overridden by subclasses, ignored otherwise.
        """
        return None

    @property
    def _C_free(self):
        """
        A symbolic destructor for the LocalObject, injected in the generated code.

        Notes
        -----
        To be overridden by subclasses, ignored otherwise.
        """
        return None

    @property
    def _mem_internal_eager(self):
        return self._liveness == 'eager'

    @property
    def _mem_internal_lazy(self):
        return self._liveness == 'lazy'

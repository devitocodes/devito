import sympy

import devito.types.basic as basic

from devito.tools import dtype_to_cstr


class OpsAccessible(basic.Symbol):
    """
    OPS accessible symbol

    Parameters
    ----------
    name : str
        Name of the symbol.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    """

    def __new__(cls, name, dtype, read_only, *args, **kwargs):
        obj = basic.Symbol.__new__(cls, name, dtype, *args, **kwargs)
        obj.__init__(name, dtype, read_only, *args, **kwargs)
        return obj

    def __init__(self, name, dtype, read_only, *args, **kwargs):
        self.read_only = read_only
        super().__init__(name, dtype, *args, **kwargs)

    def __call__(self, indexes):
        return OpsAccess(self, indexes)

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_typename(self):
        return '%sACC<%s>' % (
            'const ' if self.read_only else '',
            dtype_to_cstr(self.dtype)
        )

    @property
    def _C_typedata(self):
        return 'ACC<%s>' % dtype_to_cstr(self.dtype)


class OpsAccess(basic.Basic):
    """
    OPS access

    Parameters
    ----------
    base : OpsAccessible
        Symbol to access
    indexes: [int]
        Indexes to access
    """

    def __init__(self, base, indexes):
        self.base = base
        self.indexes = indexes

    @property
    def _C_name(self):
        return "%s(%s)" % (
            self.base._C_name,
            ",".join([str(i) for i in self.indexes])
        )

    @property
    def _C_typename(self):
        return self.base._C_typename

    @property
    def _C_typedata(self):
        return self.base._C_typedata

    def __repr__(self):
        return "%s(%s)" % (
            self.base.name,
            ", ".join([str(i) for i in self.indexes])
        )

    def __str__(self):
        return self.__repr__()

    def _sympy_(self):
        return sympy.Function(self.base.name)(*self.indexes)

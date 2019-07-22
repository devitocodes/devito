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
    is_Scalar = True

    def __new__(cls, name, dtype, read_only, *args, **kwargs):
        obj = basic.Symbol.__new__(cls, name, dtype, *args, **kwargs)
        obj.__init__(name, dtype, read_only, *args, **kwargs)
        return obj

    def __init__(self, name, dtype, read_only, *args, **kwargs):
        self.read_only = read_only
        super().__init__(name, dtype, *args, **kwargs)

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_typename(self):
        return '%sACC<%s> &' % (
            'const ' if self.read_only else '',
            dtype_to_cstr(self.dtype)
        )

    @property
    def _C_typedata(self):
        return 'ACC<%s>' % dtype_to_cstr(self.dtype)


class OpsAccess(basic.Basic, sympy.Basic):
    """
    OPS access

    Parameters
    ----------
    base : OpsAccessible
        Symbol to access
    indices: [int]
        Indices to access
    """

    def __init__(self, base, indices, *args, **kwargs):
        self.base = base
        self.indices = indices
        super().__init__(*args, **kwargs)

    def _hashable_content(self):
        return (self.base,)

    @property
    def function(self):
        return self.base.function

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def _C_name(self):
        return "%s(%s)" % (
            self.base._C_name,
            ", ".join([str(i) for i in self.indices])
        )

    @property
    def _C_typename(self):
        return self.base._C_typename

    @property
    def _C_typedata(self):
        return self.base._C_typedata

    @property
    def args(self):
        return (self.base,)

    def __str__(self):
        return "%s(%s)" % (
            self.base.name,
            ", ".join([str(i) for i in self.indices])
        )

    def as_coeff_Mul(self):
        return sympy.S.One, self

    def as_coeff_Add(self):
        return sympy.S.Zero, self

    __repr__ = __str__

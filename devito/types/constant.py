import numpy as np

from devito.exceptions import InvalidArgument
from devito.logger import warning
from devito.types.args import ArgProvider
from devito.types.basic import DataSymbol

__all__ = ['Constant']


class Constant(DataSymbol, ArgProvider):

    """
    Symbol representing a constant, scalar value in symbolic equations.
    A Constant carries a scalar value.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.

    Examples
    --------
    >>> from devito import Constant
    >>> c = Constant(name='c')
    >>> c
    c
    >>> c.data
    0.0
    >>> c.data = 4
    >>> c.data
    4.0

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    """

    is_Input = True
    is_Constant = True
    is_Scalar = True

    def __init_finalize__(self, *args, **kwargs):
        self._value = kwargs.get('value', 0)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    @property
    def is_const(self):
        return True

    @property
    def data(self):
        """The value of the data object, as a scalar (int, float, ...)."""
        return self.dtype(self._value)

    @data.setter
    def data(self, val):
        self._value = val

    @property
    def _arg_names(self):
        """Tuple of argument names introduced by this symbol."""
        return (self.name,)

    def _arg_defaults(self, alias=None):
        """A map of default argument values defined by this symbol."""
        key = alias or self
        return {key.name: self.data}

    def _arg_values(self, **kwargs):
        """
        Produce a map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        Parameters
        ----------
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, Constant):
                return new._arg_defaults(alias=self)
            else:
                return {self.name: new}
        else:
            return self._arg_defaults()

    def _arg_check(self, args, intervals):
        """
        Check that ``args`` contains legal runtime values bound to ``self``.
        """
        if self.name not in args:
            raise InvalidArgument("No runtime value for %s" % self.name)
        key = args[self.name]
        try:
            # Might be a plain number, w/o a dtype field
            if key.dtype != self.dtype:
                warning("Data type %s of runtime value `%s` does not match the "
                        "Constant data type %s" % (key.dtype, self.name, self.dtype))
        except AttributeError:
            pass

    _pickle_kwargs = DataSymbol._pickle_kwargs + ['_value']

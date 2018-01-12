import sympy
import numpy as np
from cached_property import cached_property

from devito.types import AbstractSymbol, Scalar, Symbol

__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'SteppingDimension']


class Dimension(AbstractSymbol):

    is_Dimension = True
    is_Space = False
    is_Time = False

    is_Derived = False
    is_Stepping = False
    is_Lowered = False

    """
    Index object that represents a problem dimension and thus defines a
    potential iteration space.

    :param name: Name of the dimension symbol.
    :param reverse: Optional, Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """

    def __new__(cls, name, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        newobj._reverse = kwargs.get('reverse', False)
        newobj._spacing = kwargs.get('spacing', Scalar(name='h_%s' % name))
        return newobj

    def __str__(self):
        return self.name

    @property
    def dtype(self):
        # TODO: Do dimensions really need a dtype?
        return np.int32

    @cached_property
    def symbolic_size(self):
        """The symbolic size of this dimension."""
        return Symbol(name=self.size_name)

    @cached_property
    def symbolic_start(self):
        return Symbol(name=self.start_name)

    @cached_property
    def symbolic_end(self):
        return Symbol(name=self.end_name)

    @property
    def symbolic_extent(self):
        """Return the extent of the loop over this dimension.
        Would be the same as size if using default values """
        return (self.symbolic_end - self.symbolic_start)

    @property
    def limits(self):
        return (self.symbolic_start, self.symbolic_end, 1)

    @property
    def size_name(self):
        return "%s_size" % self.name

    @property
    def start_name(self):
        return "%s_s" % self.name

    @property
    def end_name(self):
        return "%s_e" % self.name

    @property
    def reverse(self):
        return self._reverse

    @property
    def spacing(self):
        return self._spacing

    @reverse.setter
    def reverse(self, val):
        # TODO: this is an outrageous hack. TimeFunctions are updating this value
        # at construction time. This is a symptom we need local and global dimensions
        self._reverse = val

    @property
    def base(self):
        return self

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() +\
            (self.reverse, self.spacing)


class SpaceDimension(Dimension):

    is_Space = True

    """
    Dimension symbol to represent a space dimension that defines the
    extent of physical grid. :class:`SpaceDimensions` create dedicated
    shortcut notations for spatial derivatives on :class:`Function`
    symbols.
    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class TimeDimension(Dimension):

    is_Time = True

    """
    Dimension symbol to represent a dimension that defines the extent
    of time. As time might be used in different contexts, all derived
    time dimensions should inherit from :class:`TimeDimension`.
    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class SteppingDimension(Dimension):

    is_Derived = True
    is_Stepping = True

    """
    Dimension symbol that defines the stepping direction of an
    :class:`Operator` and implies modulo buffered iteration. This is most
    commonly use to represent a timestepping dimension.
    :param parent: Parent dimension over which to loop in modulo fashion.
    """

    def __new__(cls, name, parent, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        assert isinstance(parent, Dimension)
        newobj._modulo = kwargs.get('modulo', 2)
        newobj._parent = parent
        # Inherit time/space identifiers
        newobj.is_Time = parent.is_Time
        newobj.is_Space = parent.is_Space
        return newobj

    @property
    def parent(self):
        return self._parent

    @property
    def modulo(self):
        return self._modulo

    @modulo.setter
    def modulo(self, val):
        # TODO: this is an outrageous hack. TimeFunctions are updating this value
        # at construction time. This is a symptom we need local and global dimensions
        self._modulo = val

    @property
    def reverse(self):
        return self.parent.reverse

    @property
    def spacing(self):
        return self.parent.spacing

    def _hashable_content(self):
        return (self.parent._hashable_content(), self.modulo)


class LoweredDimension(Dimension):

    is_Lowered = True

    """
    Dimension symbol representing a modulo iteration created when
    resolving a :class:`SteppingDimension`.

    :param origin: The expression mapped to this dimension.
    """

    def __new__(cls, name, origin, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        newobj._origin = origin
        return newobj

    @property
    def origin(self):
        return self._origin

    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.origin,)

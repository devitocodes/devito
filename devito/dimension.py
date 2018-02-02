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
    is_Sub = False
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
        return Scalar(name=self.size_name, dtype=np.int32)

    @cached_property
    def symbolic_start(self):
        """
        The symbol defining the iteration start for this dimension.
        """
        return Scalar(name=self.start_name, dtype=np.int32)

    @cached_property
    def symbolic_end(self):
        """
        The symbol defining the iteration end for this dimension.
        """
        return Scalar(name=self.end_name, dtype=np.int32)

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
        # at construction time.
        self._reverse = val

    @property
    def base(self):
        return self

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() +\
            (self.reverse, self.spacing)

    def argument_defaults(self, size=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param size: Optional, known size as provided by data-carrying symbols
        """
        return {self.start_name: 0, self.end_name: size, self.size_name: size}

    def argument_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        values = {}

        if self.start_name in kwargs:
            values[self.start_name] = kwargs.pop(self.start_name)

        if self.end_name in kwargs:
            values[self.end_name] = kwargs.pop(self.end_name)

        # Let the dimension name be an alias for `dim_e`
        if self.name in kwargs:
            values[self.end_name] = kwargs.pop(self.name)

        return values


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


class DerivedDimension(Dimension):

    is_Derived = True

    """
    Dimension symbol derived from a ``parent`` Dimension.

    :param name: Name of the dimension symbol.
    :param parent: Parent dimension from which the ``SubDimension`` is
                   created.
    """

    def __new__(cls, name, parent, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        assert isinstance(parent, Dimension)
        newobj._parent = parent
        # Inherit time/space identifiers
        newobj.is_Time = parent.is_Time
        newobj.is_Space = parent.is_Space
        return newobj

    @property
    def parent(self):
        return self._parent

    @property
    def reverse(self):
        return self.parent.reverse

    @property
    def spacing(self):
        return self.parent.spacing

    def _hashable_content(self):
        return (self.parent._hashable_content(),)


class SubDimension(DerivedDimension):

    is_Sub = True

    """
    Dimension symbol representing a sub-region of a ``parent`` Dimension.

    :param name: Name of the dimension symbol.
    :param parent: Parent dimension from which the SubDimension is created.
    :param lower: Lower offset from the ``parent`` dimension.
    :param upper: Upper offset from the ``parent`` dimension.
    """

    def __new__(cls, name, parent, lower, upper, **kwargs):
        newobj = DerivedDimension.__new__(cls, name, parent)
        newobj._lower = lower
        newobj._upper = upper
        return newobj

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    def _hashable_content(self):
        return (self.parent._hashable_content(), self.lower, self.upper)

    def argument_defaults(self, parent_defaults):
        """
        Returns a map of default argument values defined by this symbol.

        :param parent_defaults: Default values for the parent dimensions.
        """
        args = {}

        if self.parent.start_name in parent_defaults:
            args[self.start_name] = parent_defaults[self.parent.start_name] + self.lower

        if self.parent.end_name in parent_defaults:
            args[self.end_name] = parent_defaults[self.parent.end_name] + self.upper

        if self.parent.size_name in parent_defaults:
            args[self.size_name] = parent_defaults[self.parent.size_name] -\
                (self.lower + self.upper)

        return args


class SteppingDimension(DerivedDimension):

    is_Stepping = True

    """
    Dimension symbol that defines the stepping direction of an
    :class:`Operator` and implies modulo buffered iteration. This is most
    commonly use to represent a timestepping dimension.

    :param name: Name of the dimension symbol.
    :param parent: Parent dimension over which to loop in modulo fashion.
    """

    @property
    def symbolic_start(self):
        """
        The symbol defining the iteration start for this dimension.

        note ::

        Internally we always define symbolic iteration ranges in terms
        of the parent variable.
        """
        return self.parent.symbolic_start

    @property
    def symbolic_end(self):
        """
        The symbol defining the iteration end for this dimension.

        note ::

        Internally we always define symbolic iteration ranges in terms
        of the parent variable.
        """
        return self.parent.symbolic_end

    def argument_defaults(self, size=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param size: Optional, known size as provided by data-carrying symbols

        note ::

        A :class:`SteppingDimension` neither knows its size nor its
        iteration end point. So all we can provide is a starting point.
        """
        return {self.parent.start_name: 0}

    def argument_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        values = self.parent.argument_values(**kwargs)

        if self.start_name in kwargs:
            values[self.parent.start_name] = kwargs.pop(self.start_name)

        if self.end_name in kwargs:
            values[self.parent.end_name] = kwargs.pop(self.end_name)

        # Let the dimension name be an alias for `dim_e`
        if self.name in kwargs:
            values[self.parent.end_name] = kwargs.pop(self.name)

        return values


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

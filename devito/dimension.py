import sympy
import numpy as np
from cached_property import cached_property

from devito.exceptions import InvalidArgument
from devito.types import AbstractSymbol, Scalar, Symbol

__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'DefaultDimension',
           'SteppingDimension', 'SubDimension', 'ConditionalDimension', 'dimensions']


class Dimension(AbstractSymbol):

    is_Dimension = True
    is_Space = False
    is_Time = False

    is_Default = False
    is_Derived = False
    is_NonlinearDerived = False
    is_Sub = False
    is_Conditional = False
    is_Stepping = False

    is_Lowered = False

    """
    Index object that represents a problem dimension and thus defines a
    potential iteration space.

    :param name: Name of the dimension symbol.
    :param spacing: Optional, symbol for the spacing along this dimension.
    """

    def __new__(cls, name, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
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
        return Scalar(name=self.min_name, dtype=np.int32)

    @cached_property
    def symbolic_end(self):
        """
        The symbol defining the iteration end for this dimension.
        """
        return Scalar(name=self.max_name, dtype=np.int32)

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
    def ext_name(self):
        return "%s_n" % self.name

    @property
    def min_name(self):
        return "%s_m" % self.name

    @property
    def max_name(self):
        return "%s_M" % self.name

    @property
    def spacing(self):
        return self._spacing

    @property
    def base(self):
        return self

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() + (self.spacing,)

    @property
    def _defines(self):
        return {self}

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this dimension."""
        return (self.name, self.size_name, self.ext_name, self.max_name, self.min_name)

    def _arg_defaults(self, start=None, size=None, alias=None):
        """
        Returns a map of default argument values defined by this dimension.

        :param start: (Optional) known starting point as provided by
                      data-carrying symbols.
        :param size: (Optional) known size as provided by data-carrying symbols.
        :param alias: (Optional) name under which to store values.
        """
        dim = alias or self
        return {dim.min_name: start or 0, dim.max_name: size, dim.size_name: size}

    def _arg_values(self, args, interval, **kwargs):
        """
        Returns a map of argument values after evaluating user input. If no
        user input is provided, get a known value in ``args`` and adjust it
        so that no out-of-bounds memory accesses will be performeed. The
        adjustment exploits the information in ``interval``, a :class:`Interval`
        describing the data space of this dimension. If there is no known value
        in ``args``, use a default value.

        :param args: Dictionary of known argument values.
        :param interval: A :class:`Interval` for ``self``.
        :param kwargs: Dictionary of user-provided argument overrides.
        """
        defaults = self._arg_defaults()
        values = {}

        # Min value
        if self.min_name in kwargs:
            # User-override
            values[self.min_name] = kwargs.pop(self.min_name)
        else:
            # Adjust known/default value to avoid OOB accesses
            values[self.min_name] = args.get(self.min_name, defaults[self.min_name])
            try:
                values[self.min_name] -= min(interval.lower, 0)
            except (AttributeError, TypeError):
                pass

        # Max value
        if self.max_name in kwargs:
            # User-override
            values[self.max_name] = kwargs.pop(self.max_name)
        elif self.name in kwargs:
            # Let `dim.name` to be an alias for `dim.max_name`
            values[self.max_name] = kwargs.pop(self.name)
        elif self.ext_name in kwargs:
            # Extent is used to derive max value
            values[self.max_name] = values[self.min_name] + kwargs[self.ext_name] - 1
        else:
            # Adjust known/default value to avoid OOB accesses
            values[self.max_name] = args.get(self.max_name, defaults[self.max_name])
            try:
                values[self.max_name] -= (1 + max(interval.upper, 0))
            except (AttributeError, TypeError):
                pass

        return values

    def _arg_check(self, args, size, interval):
        """
        :raises InvalidArgument: If any of the ``self``-related runtime arguments
                                 in ``args`` will cause an out-of-bounds access.
        """
        if self.min_name not in args:
            raise InvalidArgument("No runtime value for %s" % self.min_name)
        if interval.is_Defined and args[self.min_name] + interval.lower < 0:
            raise InvalidArgument("OOB detected due to %s=%d" % (self.min_name,
                                                                 args[self.min_name]))

        if self.max_name not in args:
            raise InvalidArgument("No runtime value for %s" % self.max_name)
        if interval.is_Defined and args[self.max_name] + interval.upper >= size:
            raise InvalidArgument("OOB detected due to %s=%d" % (self.max_name,
                                                                 args[self.max_name]))

        if args[self.max_name] < args[self.min_name]:
            raise InvalidArgument("Illegal max=%s < min=%s"
                                  % (args[self.max_name], args[self.min_name]))


class SpaceDimension(Dimension):

    is_Space = True

    """
    Dimension symbol to represent a space dimension that defines the
    extent of physical grid. :class:`SpaceDimensions` create dedicated
    shortcut notations for spatial derivatives on :class:`Function`
    symbols.

    :param name: Name of the dimension symbol.
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class TimeDimension(Dimension):

    is_Time = True

    """
    Dimension symbol to represent a dimension that defines the extent
    of time. As time might be used in different contexts, all derived
    time dimensions should inherit from :class:`TimeDimension`.

    :param name: Name of the dimension symbol.
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class DefaultDimension(Dimension):
    is_Default = True

    def __new__(cls, name, **kwargs):
        newobj = sympy.Symbol.__new__(cls, name)
        newobj._spacing = kwargs.get('spacing', Scalar(name='h_%s' % name))
        newobj._default_value = kwargs.get('default_value', None)
        return newobj

    def _arg_defaults(self, start=None, size=None, alias=None):
        dim = alias or self
        size = size or dim._default_value
        return {dim.min_name: start or 0, dim.max_name: size, dim.size_name: size}


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
    def spacing(self):
        return self.parent.spacing

    def _hashable_content(self):
        return (self.parent._hashable_content(),)

    @property
    def _defines(self):
        return {self} | self.parent._defines

    @property
    def _arg_names(self):
        return self.parent._arg_names


class SubDimension(DerivedDimension):

    is_Sub = True

    """
    Dimension symbol representing a contiguous sub-region of a given
    ``parent`` Dimension.

    :param name: Name of the dimension symbol.
    :param parent: Parent dimension from which the SubDimension is created.
    :param lower: Lower offset from the ``parent`` dimension.
    :param upper: Upper offset from the ``parent`` dimension.
    """

    def __new__(cls, name, parent, lower, upper, **kwargs):
        newobj = DerivedDimension.__new__(cls, name, parent, **kwargs)
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

    def _arg_values(self, args, interval, **kwargs):
        values = {}
        values[self.min_name] = args[self.parent.min_name] + self.lower
        values[self.max_name] = args[self.parent.max_name] + self.upper
        values[self.size_name] = args[self.parent.size_name] - (self.lower + self.upper)
        return values


class ConditionalDimension(DerivedDimension):

    is_NonlinearDerived = True
    is_Conditional = True

    """
    Dimension symbol representing a sub-region of a given ``parent`` Dimension.
    Unlike a :class:`SubDimension`, a ConditionalDimension does not represent
    a contiguous region. The iterations touched by a ConditionalDimension
    are expressible in two different ways: ::

        * ``factor``: an integer indicating the size of the increment.
        * ``condition``: an arbitrary SymPy expression depending on ``parent``.
                         All iterations for which the expression evaluates to
                         True are part of the ``SubDimension`` region.
    """

    def __new__(cls, name, parent, **kwargs):
        newobj = DerivedDimension.__new__(cls, name, parent, **kwargs)
        newobj._factor = kwargs.get('factor')
        newobj._condition = kwargs.get('condition')
        return newobj

    @property
    def factor(self):
        return self._factor

    @property
    def condition(self):
        return self._condition

    def _hashable_content(self):
        return (self.parent._hashable_content(), self.factor, self.condition)

    def _arg_defaults(self, **kwargs):
        """
        A :class:`ConditionalDimension` provides no arguments, so this
        method returns an empty dict.
        """
        return {}

    def _arg_values(self, *args, **kwargs):
        """
        A :class:`ConditionalDimension` provides no arguments, so there are
        no argument values to be derived.
        """
        return {}

    def _arg_check(self, *args):
        """
        A :class:`ConditionalDimension` provides no arguments, so there are
        no checks to be performed.
        """
        return


class SteppingDimension(DerivedDimension):

    is_NonlinearDerived = True
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

    @property
    def _arg_names(self):
        return (self.min_name, self.max_name, self.name) + self.parent._arg_names

    def _arg_defaults(self, start=None, **kwargs):
        """
        Returns a map of default argument values defined by this dimension.

        :param start: Optional, known starting point as provided by
                      data-carrying symbols.

        note ::

        A :class:`SteppingDimension` neither knows its size nor its
        iteration end point. So all we can provide is a starting point.
        """
        return {self.parent.min_name: start}

    def _arg_values(self, *args, **kwargs):
        values = {}

        if self.min_name in kwargs:
            values[self.parent.min_name] = kwargs.pop(self.min_name)

        if self.max_name in kwargs:
            values[self.parent.max_name] = kwargs.pop(self.max_name)

        # Let the dimension name be an alias for `dim_e`
        if self.name in kwargs:
            values[self.parent.max_name] = kwargs.pop(self.name)

        return values

    def _arg_check(self, *args):
        return


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


def dimensions(names):
    """
    Shortcut for: ::

        dimensions('i j k') -> [Dimension('i'), Dimension('j'), Dimension('k')]
    """
    assert type(names) == str
    return tuple(Dimension(i) for i in names.split())

import sympy
import numpy as np
from cached_property import cached_property

from devito.types import AbstractSymbol, Scalar, Symbol

__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'SteppingDimension',
           'SubDimension', 'ConditionalDimension', 'dimensions']


class Dimension(AbstractSymbol):

    is_Dimension = True
    is_Space = False
    is_Time = False

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
    def spacing(self):
        return self._spacing

    @property
    def base(self):
        return self

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() + (self.spacing,)

    def _arg_defaults(self, start=None, size=None, alias=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param start: (Optional) known starting point as provided by
                      data-carrying symbols.
        :param size: (Optional) known size as provided by data-carrying symbols.
        :param alias: (Optional) name under which to store values.
        """
        dim = alias or self
        return {dim.start_name: start or 0, dim.end_name: size, dim.size_name: size}

    def _arg_infers(self, args, interval=None, direction=None):
        """
        Returns a map of "better" default argument values, reading this symbols'
        argument values in ``args`` and adjusting them if an interval or a direction
        are provided.

        :param args: Dictionary of known argument values.
        :param interval: (Optional) a :class:`Interval` for ``self``.
        :param direction: (Optional) a :class:`IterationDirection` for ``self``.
        """
        inferred = {}

        if interval is None or direction is None:
            return inferred

        if self.start_name in args:
            inferred[self.start_name] = args[self.start_name] - min(interval.lower, 0)

        if self.end_name in args:
            inferred[self.end_name] = args[self.end_name] - 1

        return inferred

    def _arg_values(self, **kwargs):
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

    def _arg_infers(self, args, interval):
        inferred = {}

        if self.parent.start_name in args:
            inferred[self.start_name] = args[self.parent.start_name] + self.lower

        if self.parent.end_name in args:
            inferred[self.end_name] = args[self.parent.end_name] + self.upper

        if self.parent.size_name in args:
            inferred[self.size_name] = args[self.parent.size_name] -\
                (self.lower + self.upper)

        return inferred


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

    def _arg_defaults(self, start=None, size=None, **kwargs):
        """
        Returns a map of default argument values defined by this symbol.

        :param start: Optional, known starting point as provided by
                      data-carrying symbols.
        :param size: Optional, known size as provided by data-carrying symbols.

        note ::

        A :class:`SteppingDimension` neither knows its size nor its
        iteration end point. So all we can provide is a starting point.
        """
        return {self.parent.start_name: start}

    def _arg_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        values = self.parent._arg_values(**kwargs)

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


def dimensions(names):
    """
    Shortcut for: ::

        dimensions('i j k') -> [Dimension('i'), Dimension('j'), Dimension('k')]
    """
    assert type(names) == str
    return tuple(Dimension(i) for i in names.split())

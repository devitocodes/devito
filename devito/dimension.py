import sympy
from sympy.core.cache import cacheit
import numpy as np
from cached_property import cached_property

from devito.exceptions import InvalidArgument
from devito.types import AbstractSymbol, Scalar, Symbol
from devito.logger import debug

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
    A Dimension is a symbol representing a problem dimension and thus defining a
    potential iteration space.

    :param name: Name of the dimension symbol.
    :param spacing: Optional, symbol for the spacing along this dimension.
    """

    def __new__(cls, name, spacing=None):
        return Dimension.__xnew_cached_(cls, name, spacing)

    def __new_stage2__(cls, name, spacing=None):
        newobj = sympy.Symbol.__xnew__(cls, name)
        newobj._spacing = spacing or Scalar(name='h_%s' % name)
        return newobj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

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

        # Allow the specific case of max=min-1, which disables the loop
        if args[self.max_name] < args[self.min_name]-1:
            raise InvalidArgument("Illegal max=%s < min=%s"
                                  % (args[self.max_name], args[self.min_name]))
        elif args[self.max_name] == args[self.min_name]-1:
            debug("%s=%d and %s=%d might cause no iterations along Dimension %s",
                  self.min_name, args[self.min_name],
                  self.max_name, args[self.max_name], self.name)


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

    """
    Dimension symbol to represent a dimension that has a statically-known size.

    .. note::

        A DefaultDimension carries a value, so it has a mutable state. Hence, it
        is not cached.
    """

    def __new__(cls, name, spacing=None, default_value=None):
        newobj = Dimension.__xnew__(cls, name)
        newobj._default_value = default_value or 0
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
    :param parent: The parent Dimension.
    """

    def __new__(cls, name, parent):
        return DerivedDimension.__xnew_cached_(cls, name, parent)

    def __new_stage2__(cls, name, parent):
        assert isinstance(parent, Dimension)
        newobj = sympy.Symbol.__xnew__(cls, name)
        newobj._parent = parent
        # Inherit time/space identifiers
        newobj.is_Time = parent.is_Time
        newobj.is_Space = parent.is_Space
        return newobj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def parent(self):
        return self._parent

    @property
    def spacing(self):
        return self.parent.spacing

    def _hashable_content(self):
        return (self.name, self.parent._hashable_content())

    @property
    def _defines(self):
        return {self} | self.parent._defines

    @property
    def _arg_names(self):
        return self.parent._arg_names

    def _arg_check(self, *args):
        """
        A :class:`DerivedDimension` performs no runtime checks.
        """
        return


class SubDimension(DerivedDimension):

    is_Sub = True

    """
    Dimension symbol representing a contiguous sub-region of a given
    ``parent`` Dimension.

    :param name: Name of the dimension symbol.
    :param parent: Parent dimension from which the SubDimension is created.
    :param lower: Symbolic expression to provide the lower bound
    :param upper: Symbolic expression to provide the upper bound
    """

    def __new__(cls, name, parent, lower, upper, size):
        return SubDimension.__xnew_cached_(cls, name, parent, lower, upper, size)

    def __new_stage2__(cls, name, parent, lower, upper, size):
        newobj = DerivedDimension.__xnew__(cls, name, parent)
        newobj._interval = sympy.Interval(lower, upper)
        newobj._size = size
        return newobj

    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @classmethod
    def left(cls, name, parent, thickness):
        return cls(name, parent,
                   lower=parent.symbolic_start,
                   upper=parent.symbolic_start+thickness-1,
                   size=thickness)

    @classmethod
    def right(cls, name, parent, thickness):
        return cls(name, parent,
                   lower=parent.symbolic_end-thickness+1,
                   upper=parent.symbolic_end,
                   size=thickness)

    @classmethod
    def middle(cls, name, parent, thickness_left, thickness_right):
        return cls(name, parent,
                   lower=parent.symbolic_start+thickness_left,
                   upper=parent.symbolic_end-thickness_right,
                   size=parent.symbolic_size-thickness_left-thickness_right)

    @property
    def symbolic_start(self):
        return self._interval.left

    @property
    def symbolic_end(self):
        return self._interval.right

    @property
    def symbolic_size(self):
        return self._size

    def offset_lower(self):
        # The lower extreme of the subdimension can be related to either the
        # start or end of the parent dimension
        try:
            val = self.symbolic_start - self.parent.symbolic_start
            return int(val), self.parent.symbolic_start
        except TypeError:
            val = self.symbolic_start - self.parent.symbolic_end
            return int(val), self.parent.symbolic_end

    def offset_upper(self):
        # The upper extreme of the subdimension can be related to either the
        # start or end of the parent dimension
        try:
            val = self.symbolic_end - self.parent.symbolic_start
            return int(val), self.parent.symbolic_start
        except TypeError:
            val = self.symbolic_end - self.parent.symbolic_end
            return int(val), self.parent.symbolic_end

    def _hashable_content(self):
        return super(SubDimension, self)._hashable_content() + (self._interval,
                                                                self._size)

    def _arg_defaults(self, **kwargs):
        """
        A :class:`SubDimension` provides no arguments, so this method returns
        an empty dict.
        """
        return {}

    def _arg_values(self, *args, **kwargs):
        """
        A :class:`SubDimension` provides no arguments, so there are
        no argument values to be derived.
        """
        return {}


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

    ConditionalDimension needs runtime arguments. The generated C code will require
    the size of the dimension to initialize the arrays as e.g:

        .. code-block:: python
           x = grid.dimension[0]
           x1 = ConditionalDimension(name='x1', parent=x, factor=2)
           u1 = TimeFunction(name='u1', dimensions=(x1,), size=grid.shape[0]/factor)
           # The generated code will look like
           float (*restrict u1)[x1_size + 1] =

    """

    def __new__(cls, name, parent, factor=None, condition=None):
        return ConditionalDimension.__xnew_cached_(cls, name, parent, factor, condition)

    def __new_stage2__(cls, name, parent, factor, condition):
        newobj = DerivedDimension.__xnew__(cls, name, parent)
        newobj._factor = factor
        newobj._condition = condition
        return newobj

    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def spacing(self):
        return self.factor * self.parent.spacing

    @property
    def factor(self):
        return self._factor

    @property
    def condition(self):
        return self._condition

    def _hashable_content(self):
        return super(ConditionalDimension, self)._hashable_content() + (self.factor,
                                                                        self.condition)


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
        """
        The argument values provided by a :class:`SteppingDimension` are those
        of its parent, as it acts as an alias.
        """
        values = {}

        if self.min_name in kwargs:
            values[self.parent.min_name] = kwargs.pop(self.min_name)

        if self.max_name in kwargs:
            values[self.parent.max_name] = kwargs.pop(self.max_name)

        # Let the dimension name be an alias for `dim_e`
        if self.name in kwargs:
            values[self.parent.max_name] = kwargs.pop(self.name)

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

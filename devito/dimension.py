import sympy
from sympy.core.cache import cacheit
import numpy as np
from cached_property import cached_property

from devito.exceptions import InvalidArgument
from devito.types import AbstractSymbol, Scalar
from devito.logger import debug
from devito.tools import Pickable

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

    @property
    def root(self):
        return self

    @property
    def _properties(self):
        return (self.spacing,)

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() + self._properties

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

    # Pickling support
    _pickle_args = ['name']
    _pickle_kwargs = ['spacing']
    __reduce_ex__ = Pickable.__reduce_ex__


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

    _keymap = {}
    """Map all seen instance `_properties` to a unique number. This is used
    to create unique Dimension names."""

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

    @classmethod
    def _gensuffix(cls, key):
        return cls._keymap.setdefault(key, len(cls._keymap))

    @classmethod
    def _genname(cls, prefix, key):
        return "%s%d" % (prefix, cls._gensuffix(key))

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        return self._parent.root

    @property
    def spacing(self):
        return self.parent.spacing

    @property
    def _properties(self):
        return ()

    def _hashable_content(self):
        return (self.name, self.parent._hashable_content()) + self._properties

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

    # Pickling support
    _pickle_args = Dimension._pickle_args + ['parent']
    _pickle_kwargs = []


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

    @property
    def _properties(self):
        return (self._interval, self._size)

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

    # Pickling support
    _pickle_args = DerivedDimension._pickle_args +\
        ['symbolic_start', 'symbolic_end', 'symbolic_size']
    _pickle_kwargs = []


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

    @property
    def _properties(self):
        return (self._factor, self._condition)

    # Pickling support
    _pickle_kwargs = DerivedDimension._pickle_kwargs + ['factor', 'condition']


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


class ModuloDimension(DerivedDimension):

    """
    Dimension symbol representing a non-contiguous sub-region of a given
    ``parent`` Dimension, which cyclically produces a finite range of values,
    such as ``0, 1, 2, 0, 1, 2, 0, ...``.

    :param parent: Parent dimension from which the ModuloDimension is created.
    :param offset: An integer representing an offset from the parent dimension.
    :param modulo: The extent of the range.
    :param name: (Optional) force a name for this Dimension.
    """

    def __new__(cls, parent, offset, modulo, name=None):
        return ModuloDimension.__xnew_cached_(cls, parent, offset, modulo, name)

    def __new_stage2__(cls, parent, offset, modulo, name):
        if name is None:
            name = cls._genname(parent.name, (offset, modulo))
        newobj = DerivedDimension.__xnew__(cls, name, parent)
        newobj._offset = offset
        newobj._modulo = modulo
        return newobj

    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def offset(self):
        return self._offset

    @property
    def modulo(self):
        return self._modulo

    @property
    def origin(self):
        return self.parent + self.offset

    @cached_property
    def symbolic_start(self):
        return (self.root + self.offset) % self.modulo

    symbolic_incr = symbolic_start

    @property
    def _properties(self):
        return (self._offset, self._modulo)

    def _arg_defaults(self, **kwargs):
        """
        A :class:`ModuloDimension` provides no arguments, so this method
        returns an empty dict.
        """
        return {}

    def _arg_values(self, *args, **kwargs):
        """
        A :class:`ModuloDimension` provides no arguments, so there are
        no argument values to be derived.
        """
        return {}

    # Pickling support
    _pickle_args = ['parent', 'offset', 'modulo']
    _pickle_kwargs = ['name']


class IncrDimension(DerivedDimension):

    """
    Dimension symbol representing a non-contiguous sub-region of a given
    ``parent`` Dimension, with one point every ``step`` points. Thus, if
    ``step == k``, the dimension represents the sequence ``start, start + k,
    start + 2*k, ...``.

    :param parent: Parent dimension from which the IncrDimension is created.
    :param start: An integer representing the starting point of the sequence.
    :param step: The distance between two consecutive points.
    :param name: (Optional) force a name for this Dimension.
    """

    def __new__(cls, parent, start, step, name=None):
        return IncrDimension.__xnew_cached_(cls, parent, start, step, name)

    def __new_stage2__(cls, parent, start, step, name):
        if name is None:
            name = cls._genname(parent.name, (start, step))
        newobj = DerivedDimension.__xnew__(cls, name, parent)
        newobj._start = start
        newobj._step = step
        return newobj

    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def step(self):
        return self._step

    @cached_property
    def symbolic_start(self):
        return self._start

    @property
    def symbolic_incr(self):
        return self + self.step

    @property
    def _properties(self):
        return (self._start, self._step)

    def _arg_defaults(self, **kwargs):
        """
        A :class:`IncrDimension` provides no arguments, so this method
        returns an empty dict.
        """
        return {}

    def _arg_values(self, *args, **kwargs):
        """
        A :class:`IncrDimension` provides no arguments, so there are
        no argument values to be derived.
        """
        return {}

    # Pickling support
    _pickle_args = ['parent', 'symbolic_start', 'step']
    _pickle_kwargs = ['name']


def dimensions(names):
    """
    Shortcut for: ::

        dimensions('i j k') -> [Dimension('i'), Dimension('j'), Dimension('k')]
    """
    assert type(names) == str
    return tuple(Dimension(i) for i in names.split())

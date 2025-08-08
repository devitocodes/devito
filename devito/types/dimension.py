from collections import namedtuple
from functools import cached_property
import math

import sympy
from sympy.core.decorators import call_highest_priority
import numpy as np

from devito.data import LEFT, RIGHT
from devito.deprecations import deprecations
from devito.exceptions import InvalidArgument
from devito.logger import debug
from devito.tools import Pickable, is_integer, is_number, memoized_meth
from devito.types.args import ArgProvider
from devito.types.basic import Symbol, DataSymbol, Scalar
from devito.types.constant import Constant


__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'DefaultDimension',
           'CustomDimension', 'SteppingDimension', 'SubDimension',
           'MultiSubDimension', 'ConditionalDimension', 'ModuloDimension',
           'IncrDimension', 'BlockDimension', 'StencilDimension',
           'VirtualDimension', 'Spacing', 'dimensions']


SubDimensionThickness = namedtuple('SubDimensionThickness', 'left right')


class Dimension(ArgProvider):

    """
    Symbol defining an iteration space.

    A Dimension represents a problem dimension. It is typically used to index
    into Functions, but it can also appear in the middle of a symbolic expression
    just like any other symbol.

    Dimension is the root of a hierarchy of classes, which looks as follows (only
    the classes exposed to the level of the user API are shown)::

                                       Dimension
                                           |
                              ---------------------------
                              |                         |
                       DerivedDimension            DefaultDimension
                              |
                   ---------------------
                   |                   |
              SubDimension   ConditionalDimension

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : symbol, optional, default=h_name
        A symbol to represent the physical spacing along this Dimension.

    Examples
    --------
    Dimensions are automatically created when a Grid is instantiated.

    >>> from devito import Grid
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> type(x)
    <class 'devito.types.dimension.SpaceDimension'>
    >>> time = grid.time_dim
    >>> type(time)
    <class 'devito.types.dimension.TimeDimension'>
    >>> t = grid.stepping_dim
    >>> type(t)
    <class 'devito.types.dimension.SteppingDimension'>

    Alternatively, one can create Dimensions explicitly

    >>> from devito import Dimension
    >>> i = Dimension(name='i')

    Or, when many "free" Dimensions are needed, with the shortcut

    >>> from devito import dimensions
    >>> i, j, k = dimensions('i j k')

    A Dimension can be used to build a Function as well as within symbolic
    expressions, as both array index ("indexed notation") and free symbol.

    >>> from devito import Function
    >>> f = Function(name='f', shape=(4, 4), dimensions=(i, j))
    >>> f + f
    2*f(i, j)
    >>> f[i + 1, j] + f[i, j + 1]
    f[i, j + 1] + f[i + 1, j]
    >>> f*i
    i*f(i, j)
    """

    is_Dimension = True
    is_Space = False
    is_Time = False

    is_Default = False
    is_Custom = False
    is_Derived = False
    is_NonlinearDerived = False
    is_AbstractSub = False
    is_Sub = False
    is_MultiSub = False
    is_Conditional = False
    is_Stepping = False
    is_Stencil = False
    is_SubIterator = False
    is_Modulo = False
    is_Incr = False
    is_Block = False
    is_Virtual = False

    # Prioritize self's __add__ and __sub__ to construct AffineIndexAccessFunction
    _op_priority = sympy.Expr._op_priority + 1.

    __rargs__ = ('name',)
    __rkwargs__ = ('spacing',)

    def __new__(cls, *args, **kwargs):
        """
        Equivalent to ``BasicDimension(*args, **kwargs)``.

        Notes
        -----
        This is only necessary for backwards compatibility, as originally
        there was no BasicDimension (i.e., Dimension was just the top class).
        """
        if cls is Dimension:
            return BasicDimension(*args, **kwargs)
        else:
            return BasicDimension.__new__(cls, *args, **kwargs)

    @classmethod
    def class_key(cls):
        """
        Overrides sympy.Symbol.class_key such that Dimensions always
        preceed other symbols when printed (e.g. x + h_x, not h_x + x).
        """
        a, b, c = super().class_key()
        return a, b - 1, c

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        # Unlike other Symbols, Dimensions can only be integers
        return np.int32

    def __str__(self):
        return self.name

    def _hashable_content(self):
        return tuple(getattr(self, i) for i in self.__rargs__ + self.__rkwargs__)

    @property
    def spacing(self):
        """Symbol representing the physical spacing along the Dimension."""
        return self._spacing

    @cached_property
    def symbolic_size(self):
        """Symbolic size of the Dimension."""
        return Scalar(name=self.size_name, dtype=np.int32, is_const=True)

    @cached_property
    def symbolic_min(self):
        """Symbol defining the minimum point of the Dimension."""
        return Scalar(name=self.min_name, dtype=np.int32, is_const=True)

    @cached_property
    def symbolic_max(self):
        """Symbol defining the maximum point of the Dimension."""
        return Scalar(name=self.max_name, dtype=np.int32, is_const=True)

    @property
    def symbolic_incr(self):
        """The increment value while iterating over the Dimension."""
        return sympy.S.One

    @cached_property
    def size_name(self):
        return f"{self.name}_size"

    @cached_property
    def min_name(self):
        return f"{self.name}_m"

    @cached_property
    def max_name(self):
        return f"{self.name}_M"

    @property
    def indirect(self):
        return False

    @property
    def index(self):
        return self

    @property
    def is_const(self):
        return False

    @property
    def root(self):
        return self

    @cached_property
    def bound_symbols(self):
        candidates = [self.symbolic_min, self.symbolic_max, self.symbolic_size,
                      self.symbolic_incr]
        return frozenset(i for i in candidates if not i.is_Number)

    @property
    def _maybe_distributed(self):
        """Could it be a distributed Dimension?"""
        return True

    @cached_property
    def _defines(self):
        return frozenset({self})

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return AffineIndexAccessFunction(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return AffineIndexAccessFunction(self, other)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return AffineIndexAccessFunction(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return AffineIndexAccessFunction(other, -self)

    @property
    def _arg_names(self):
        """Tuple of argument names introduced by the Dimension."""
        return (self.name, self.size_name, self.max_name, self.min_name)

    def _arg_defaults(self, _min=None, size=None, alias=None):
        """
        A map of default argument values defined by the Dimension.

        Parameters
        ----------
        _min : int, optional
            Minimum point as provided by data-carrying objects.
        size : int, optional
            Size as provided by data-carrying symbols.
        alias : Dimension, optional
            To get the min/max/size names under which to store values. Use
            self's if None.
        """
        dim = alias or self
        return {dim.min_name: _min or 0,
                dim.size_name: size,
                dim.max_name: size if size is None else size-1}

    def _arg_values(self, interval, grid=None, args=None, **kwargs):
        """
        Produce a map of argument values after evaluating user input. If no user
        input is provided, get a known value in ``args`` and adjust it so that no
        out-of-bounds memory accesses will be performeed. The adjustment exploits
        the information in ``interval``, an Interval describing the Dimension data
        space. If no value is available in ``args``, use a default value.

        Parameters
        ----------
        interval : Interval
            Description of the Dimension data space.
        grid : Grid, optional
            Used for spacing overriding and MPI execution; if ``self`` is a distributed
            Dimension, then ``grid`` is used to translate user input into rank-local
            indices.
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        # Fetch user input and convert into rank-local values
        glb_minv = kwargs.pop(self.min_name, None)
        glb_maxv = kwargs.pop(self.max_name, kwargs.pop(self.name, None))
        if grid is not None and grid.is_distributed(self):
            loc_minv, loc_maxv = grid.distributor.glb_to_loc(self, (glb_minv, glb_maxv))
        else:
            loc_minv, loc_maxv = glb_minv, glb_maxv

        # If no user-override provided, use a suitable default value
        defaults = self._arg_defaults()
        if glb_minv is None:
            loc_minv = args.get(self.min_name, defaults[self.min_name])
            try:
                loc_minv -= min(interval.lower, 0)
            except (AttributeError, TypeError):
                pass
        if glb_maxv is None:
            loc_maxv = args.get(self.max_name, defaults[self.max_name])
            try:
                loc_maxv -= max(interval.upper, 0)
            except (AttributeError, TypeError):
                pass

        # Some `args` may still be DerivedDimenions' defaults. These, in turn,
        # may represent sets of legal values. If that's the case, here we just
        # pick one. Note that we sort for determinism
        try:
            loc_minv = loc_minv.stop
        except AttributeError:
            try:
                loc_minv = sorted(loc_minv).pop(0)
            except TypeError:
                pass
        try:
            loc_maxv = loc_maxv.stop
        except AttributeError:
            try:
                loc_maxv = sorted(loc_maxv).pop(0)
            except TypeError:
                pass

        return {self.min_name: loc_minv, self.max_name: loc_maxv}

    def _arg_check(self, args, size, interval):
        """
        Raises
        ------
        InvalidArgument
            If any of the ``self``-related runtime arguments in ``args``
            will cause an out-of-bounds access.
        """
        if self.min_name not in args:
            raise InvalidArgument(f"No runtime value for {self.min_name}")
        if interval.is_Defined and args[self.min_name] + interval.lower < 0:
            raise InvalidArgument(f"OOB detected due to "
                                  f"{self.min_name}={args[self.min_name]}")

        if self.max_name not in args:
            raise InvalidArgument(f"No runtime value for {self.max_name}")
        if interval.is_Defined:
            if is_integer(interval.upper):
                upper = interval.upper
            else:
                # Autopadding causes non-integer upper limit
                from devito.symbolics import normalize_args
                upper = interval.upper.subs(normalize_args(args))
            if args[self.max_name] + upper >= size:
                raise InvalidArgument(f"OOB detected due to "
                                      f"{self.max_name}={args[self.max_name]}")

        # Allow the specific case of max=min-1, which disables the loop
        if args[self.max_name] < args[self.min_name]-1:
            raise InvalidArgument("Illegal %s=%d < %s=%d"
                                  % (self.max_name, args[self.max_name],
                                     self.min_name, args[self.min_name]))
        elif args[self.max_name] == args[self.min_name]-1:
            debug("%s=%d and %s=%d might cause no iterations along Dimension %s",
                  self.min_name, args[self.min_name],
                  self.max_name, args[self.max_name], self.name)

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__
    __getnewargs_ex__ = Pickable.__getnewargs_ex__


class Spacing(Scalar):
    pass


class BasicDimension(Dimension, Symbol):

    __doc__ = Dimension.__doc__

    def __new__(cls, *args, **kwargs):
        return Symbol.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, name, spacing=None, **kwargs):
        self._spacing = spacing or Spacing(name=f'h_{name}', is_const=True)

    def __eq__(self, other):
        # Being of type Cached, Dimensions are by construction unique. But unlike
        # Symbols, equality is much stricter -- we consider any two Dimensions
        # equal iff they are the very same object. This has several advantages.
        # First of all, it makes it much more difficult to trick the compiler
        # to generate buggy code (e.g., using two different "x" Dimensions that
        # actually represent the same iteration space). Secondly, comparison
        # is much cheaper, since we avoid having to go through all of the
        # __rargs__/__rkwargs__, and there can be quite a few depending on the
        # specific Dimension type
        return self is other

    __hash__ = Symbol.__hash__


class DefaultDimension(Dimension, DataSymbol):

    """
    Symbol defining an iteration space with statically-known size.

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : Symbol, optional
        A symbol to represent the physical spacing along this Dimension.
    default_value : float, optional
        Default value associated with the Dimension.

    Notes
    -----
    A DefaultDimension carries a value, so it has a mutable state. Hence, it is
    not cached.
    """

    is_Default = True

    def __new__(cls, *args, **kwargs):
        return DataSymbol.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, name, spacing=None, default_value=None, **kwargs):
        self._spacing = spacing or Spacing(name=f'h_{name}', is_const=True)
        self._default_value = default_value or 0

    @cached_property
    def symbolic_size(self):
        return sympy.Number(self._default_value)

    def _arg_defaults(self, _min=None, size=None, alias=None):
        dim = alias or self
        size = size or dim._default_value
        return {dim.min_name: _min or 0, dim.size_name: size,
                dim.max_name: size if size is None else size-1}


class SpaceDimension(BasicDimension):

    """
    Symbol defining an iteration space.

    This symbol represents a space dimension that defines the extent of
    a physical grid.

    A SpaceDimension creates dedicated shortcut notations for spatial
    derivatives on Functions.

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : symbol, optional
        A symbol to represent the physical spacing along this Dimension.
    """

    is_Space = True


class TimeDimension(BasicDimension):

    """
    Symbol defining an iteration space.

    This symbol represents a time dimension that defines the extent of time.

    A TimeDimension create dedicated shortcut notations for time derivatives
    on Functions.

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : symbol, optional
        A symbol to represent the physical spacing along this Dimension.
    """

    is_Time = True


class DerivedDimension(BasicDimension):

    """
    Symbol defining an iteration space derived from a ``parent`` Dimension.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    """

    is_Derived = True

    __rargs__ = Dimension.__rargs__ + ('parent',)
    __rkwargs__ = ()

    def __init_finalize__(self, name, parent):
        assert isinstance(parent, Dimension)
        self._parent = parent
        # Inherit time/space identifiers
        self.is_Time = parent.is_Time
        self.is_Space = parent.is_Space

    @property
    def parent(self):
        return self._parent

    @property
    def index(self):
        return self if self.indirect else self.parent

    @property
    def root(self):
        return self._parent.root

    @property
    def spacing(self):
        return self.parent.spacing

    @cached_property
    def _defines(self):
        return frozenset({self}) | self.parent._defines

    @property
    def _arg_names(self):
        return self.parent._arg_names

    def _arg_check(self, *args, **kwargs):
        """A DerivedDimension performs no runtime checks."""
        return


# ***
# The Dimensions below are exposed in the user API. They can only be created by
# the user

class Thickness(DataSymbol):
    """A DataSymbol to represent a thickness of a SubDimension"""

    __rkwargs__ = DataSymbol.__rkwargs__ + ('root', 'side', 'local', 'value')

    def __new__(cls, *args, root=None, side=None, local=False, **kwargs):
        newobj = super().__new__(cls, *args, **kwargs)
        newobj._root = root
        newobj._side = side
        newobj._local = local

        return newobj

    def __init_finalize__(self, *args, **kwargs):
        self._value = kwargs.pop('value', None)

        kwargs.setdefault('is_const', True)
        super().__init_finalize__(*args, **kwargs)

    @property
    def root(self):
        return self._root

    @property
    def side(self):
        return self._side

    @property
    def local(self):
        return self._local

    @property
    def value(self):
        return self._value

    def _arg_values(self, grid=None, **kwargs):
        # Allow override of thickness values to disable BCs
        # However, arguments from the user are considered global
        # So overriding the thickness to a nonzero value should not cause
        # boundaries to exist between ranks where they did not before
        rtkn = kwargs.get(self.name, self.value)
        if grid is not None and grid.is_distributed(self.root):
            # Get local thickness
            if self.local:
                # Dimension is of type `left`/`right` - compute the offset
                # and then add 1 to get the appropriate thickness
                if self.value is not None:
                    tkn = grid.distributor.glb_to_loc(self.root, rtkn-1, self.side)
                    tkn = tkn+1 if tkn is not None else 0
                else:
                    tkn = 0
            else:
                # Dimension is of type `middle`
                tkn = grid.distributor.glb_to_loc(self.root, rtkn, self.side) or 0
        else:
            tkn = rtkn or 0

        return {self.name: tkn}


class AbstractSubDimension(DerivedDimension):

    """
    Symbol defining a convex iteration sub-space derived from a `parent`
    Dimension.

    Notes
    -----
    This is just the abstract base class for various types of SubDimensions.
    """

    is_AbstractSub = True

    __rargs__ = DerivedDimension.__rargs__ + ('thickness',)
    __rkwargs__ = ()

    _thickness_type = Symbol

    def __init_finalize__(self, name, parent, thickness, **kwargs):
        super().__init_finalize__(name, parent)
        thickness = thickness or (None, None)
        if any(isinstance(tkn, self._thickness_type) for tkn in thickness):
            self._thickness = SubDimensionThickness(*thickness)
        else:
            self._thickness = self._symbolic_thickness(thickness=thickness)

    @cached_property
    def _interval(self):
        left = self.parent.symbolic_min + self.ltkn
        right = self.parent.symbolic_max - self.rtkn
        return sympy.Interval(left, right)

    @memoized_meth
    def _symbolic_thickness(self, **kwargs):
        kwargs = {'dtype': np.int32, 'is_const': True, 'nonnegative': True}

        names = ["%s_%stkn" % (self.parent.name, s) for s in ('l', 'r')]
        return SubDimensionThickness(*[Symbol(name=n, **kwargs) for n in names])

    @cached_property
    def symbolic_min(self):
        return self._interval.left

    @cached_property
    def symbolic_max(self):
        return self._interval.right

    @cached_property
    def symbolic_size(self):
        # The size must be given as a function of the parent's symbols
        return self.symbolic_max - self.symbolic_min + 1

    @property
    def thickness(self):
        return self._thickness

    tkns = thickness  # Shortcut for thickness

    @property
    def ltkn(self):
        # Shortcut for the left thickness symbol
        return self.thickness.left

    @property
    def rtkn(self):
        # Shortcut for the right thickness symbol
        return self.thickness.right

    def __hash__(self):
        return id(self)


class SubDimension(AbstractSubDimension):

    """
    Symbol defining a convex iteration sub-space derived from a ``parent``
    Dimension.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    left : expr-like
        Symbolic expression providing the left (lower) bound of the
        SubDimension.
    right : expr-like
        Symbolic expression providing the right (upper) bound of the
        SubDimension.
    thickness : 2-tuple of 2-tuples
        The thickness of the left and right regions, respectively.
    local : bool
        True if, in case of domain decomposition, the SubDimension is
        guaranteed not to span more than one domain, False otherwise.

    Examples
    --------
    SubDimensions should *not* be created directly in user code; SubDomains
    should be used instead. Exceptions are rare.

    To create a SubDimension, one should use the shortcut methods ``left``,
    ``right``, ``middle``. For example, to create a SubDimension that spans
    the entire space of the parent Dimension except for the two extremes:

    >>> from devito import Dimension, SubDimension
    >>> x = Dimension('x')
    >>> xi = SubDimension.middle('xi', x, 1, 1)

    For a SubDimension that only spans the three leftmost points of its
    parent Dimension, instead:

    >>> xl = SubDimension.left('xl', x, 3)

    SubDimensions created via the ``left`` and ``right`` shortcuts are, by default,
    local (i.e., non-distributed) Dimensions, as they are assumed to fit entirely
    within a single domain. This is the most typical use case (e.g., to set up
    boundary conditions). To drop this assumption, pass ``local=False``.
    """

    is_Sub = True

    __rargs__ = AbstractSubDimension.__rargs__ + ('local',)

    _thickness_type = Thickness

    def __init_finalize__(self, name, parent, thickness, local,
                          **kwargs):
        self._local = local
        super().__init_finalize__(name, parent, thickness)

    @classmethod
    def left(cls, name, parent, thickness, local=True):
        return cls(name, parent, thickness=(thickness, None), local=local)

    @classmethod
    def right(cls, name, parent, thickness, local=True):
        return cls(name, parent, thickness=(None, thickness), local=local)

    @classmethod
    def middle(cls, name, parent, thickness_left, thickness_right, local=False):
        return cls(name, parent, thickness=(thickness_left, thickness_right), local=local)

    @memoized_meth
    def _symbolic_thickness(self, thickness=None):
        kwargs = {'dtype': np.int32, 'is_const': True, 'nonnegative': True,
                  'root': self.root, 'local': self.local}

        names = ["%s_%stkn" % (self.parent.name, s) for s in ('l', 'r')]
        sides = [LEFT, RIGHT]
        return SubDimensionThickness(*[Thickness(name=n, side=s, value=t, **kwargs)
                                       for n, s, t in zip(names, sides, thickness)])

    @cached_property
    def _interval(self):
        if self.thickness.right.value is None:  # Left SubDimension
            left = self.parent.symbolic_min
            right = self.parent.symbolic_min + self.ltkn - 1
        elif self.thickness.left.value is None:  # Right SubDimension
            left = self.parent.symbolic_max - self.rtkn + 1
            right = self.parent.symbolic_max
        else:  # Middle SubDimension
            return super()._interval

        return sympy.Interval(left, right)

    @property
    def local(self):
        return self._local

    @property
    def is_left(self):
        return self.thickness.right.value is None

    @property
    def is_right(self):
        return self.thickness.left.value is None

    @property
    def is_middle(self):
        return not self.is_left and not self.is_right

    @cached_property
    def bound_symbols(self):
        # Add thickness symbols
        return frozenset().union(*[i.free_symbols for i in super().bound_symbols])

    @property
    def _maybe_distributed(self):
        return not self.local

    @property
    def _arg_names(self):
        return tuple(k.name for k in self.thickness) + self.parent._arg_names

    def _arg_defaults(self, grid=None, **kwargs):
        return {}

    def _arg_values(self, interval, grid=None, **kwargs):
        # SubDimension thicknesses at runtime are calculated by the thicknesses
        # themselves
        return {}


class MultiSubDimension(AbstractSubDimension):

    """
    A special Dimension to be used in MultiSubDomains.
    """

    is_MultiSub = True

    __rkwargs__ = ('functions', 'bounds_indices', 'implicit_dimension')

    def __init_finalize__(self, name, parent, thickness, functions=None,
                          bounds_indices=None, implicit_dimension=None):

        super().__init_finalize__(name, parent, thickness)
        self.functions = functions
        self.bounds_indices = bounds_indices
        self.implicit_dimension = implicit_dimension

    @cached_property
    def bound_symbols(self):
        return self.parent.bound_symbols


class SubsamplingFactor(Scalar):
    pass


class ConditionalDimension(DerivedDimension):

    """
    Symbol defining a non-convex iteration sub-space derived from a ``parent``
    Dimension, implemented by the compiler generating conditional "if-then" code
    within the parent Dimension's iteration space.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    factor : int, optional, default=None
        The number of iterations between two executions of the if-branch. If None
        (default), ``condition`` must be provided.
    condition : expr-like, optional, default=None
        An arbitrary SymPy expression, typically involving the ``parent``
        Dimension. When it evaluates to True, the if-branch is executed. If None
        (default), ``factor`` must be provided.
    indirect : bool, optional, default=False
        If True, use `self`, rather than the parent Dimension, to
        index into arrays. A typical use case is when arrays are accessed
        indirectly via the ``condition`` expression.

    Examples
    --------
    Among the other things, ConditionalDimensions are indicated to implement
    Function subsampling. In the following example, an Operator evaluates the
    Function ``g`` and saves its content into ``f`` every ``factor=4`` iterations.

    >>> from devito import Dimension, ConditionalDimension, Function, Eq, Operator
    >>> size, factor = 16, 4
    >>> i = Dimension(name='i')
    >>> ci = ConditionalDimension(name='ci', parent=i, factor=factor)
    >>> g = Function(name='g', shape=(size,), dimensions=(i,))
    >>> f = Function(name='f', shape=(int(size/factor),), dimensions=(ci,))
    >>> op = Operator([Eq(g, 1), Eq(f, g)])

    The Operator generates the following for-loop (pseudocode)

    .. code-block:: C

        for (int i = i_m; i <= i_M; i += 1) {
          g[i] = 1;
          if (i%4 == 0) {
            f[i / 4] = g[i];
          }
        }

    Another typical use case is when one needs to constrain the execution of
    loop iterations so that certain conditions are honoured. The following
    artificial example uses ConditionalDimension to guard against out-of-bounds
    accesses in indirectly accessed arrays.

    >>> from sympy import And
    >>> ci = ConditionalDimension(name='ci', parent=i,
    ...                           condition=And(g[i] > 0, g[i] < 4, evaluate=False))
    >>> f = Function(name='f', shape=(int(size/factor),), dimensions=(ci,))
    >>> op = Operator(Eq(f[g[i]], f[g[i]] + 1))

    The Operator generates the following for-loop (pseudocode)

    .. code-block:: C

        for (int i = i_m; i <= i_M; i += 1) {
          if (g[i] > 0 && g[i] < 4) {
            f[g[i]] = f[g[i]] + 1;
          }
        }

    """

    is_NonlinearDerived = True
    is_Conditional = True

    __rkwargs__ = DerivedDimension.__rkwargs__ + \
        ('factor', 'condition', 'indirect')

    def __init_finalize__(self, name, parent=None, factor=None, condition=None,
                          indirect=False, **kwargs):
        # `parent=None` degenerates to a ConditionalDimension outside of
        # any iteration space
        if parent is None:
            parent = BOTTOM

        super().__init_finalize__(name, parent)

        # Process subsampling factor
        if factor is None:
            self._factor = None
        elif is_number(factor):
            self._factor = int(factor)
        elif factor.is_Constant:
            deprecations.constant_factor_warn
            self._factor = factor
        else:
            raise ValueError("factor must be an integer")

        self._condition = condition
        self._indirect = indirect

    @property
    def uses_symbolic_factor(self):
        return self._factor is not None

    @property
    def factor_data(self):
        if isinstance(self.factor, Constant):
            return self.factor.data
        elif self.factor is not None:
            return self.factor
        else:
            return 1

    @property
    def spacing(self):
        return self.factor_data * self.parent.spacing

    @property
    def factor(self):
        return self._factor

    @cached_property
    def symbolic_factor(self):
        if not self.uses_symbolic_factor:
            return None
        elif isinstance(self.factor, Constant):
            return self.factor
        else:
            return SubsamplingFactor(
                name=f'{self.name}f', dtype=np.int32, is_const=True
            )

    @property
    def condition(self):
        return self._condition

    @property
    def indirect(self):
        return self._indirect

    @cached_property
    def free_symbols(self):
        retval = set(super().free_symbols)
        if self.condition is not None:
            retval |= self.condition.free_symbols
        try:
            retval |= self.factor.free_symbols
        except AttributeError:
            pass
        return retval

    def _arg_values(self, interval, grid=None, args=None, **kwargs):
        if not self.uses_symbolic_factor:
            return {}

        args = args or {}
        fname = self.symbolic_factor.name
        fact = kwargs.get(fname, args.get(fname, self.factor_data))
        if isinstance(fact, Constant):
            fact = fact.data

        toint = lambda x: math.ceil(x / fact)
        vals = {}
        try:
            vals[self.min_name] = toint(kwargs.get(self.parent.min_name))
        except (KeyError, TypeError):
            pass

        try:
            vals[self.max_name] = toint(kwargs.get(self.parent.max_name))
        except (KeyError, TypeError):
            pass

        vals[self.symbolic_factor.name] = fact

        return vals

    def _arg_defaults(self, _min=None, size=None, alias=None):
        defaults = super()._arg_defaults(_min=_min, size=size, alias=alias)

        # We can also add the parent's default endpoint. Note that exactly
        # `factor` endpoints are legal, so we return them all. It's then
        # up to the caller to decide which one to pick upon reduction
        dim = alias or self
        if dim.uses_symbolic_factor:
            factor = defaults[dim.symbolic_factor.name] = self.factor_data
            defaults[dim.parent.max_name] = range(0, factor*size - 1)

        return defaults


# ***
# The Dimensions below are for internal use only. They are created by the compiler
# during the construction of an Operator


class ModuloDimension(DerivedDimension):

    """
    Dimension symbol representing a non-contiguous sub-region of a given
    ``parent`` Dimension, which cyclically produces a finite range of values,
    such as ``0, 1, 2, 0, 1, 2, 0, ...``.

    When ``modulo=None``, the ModuloDimension degenerates and keeps generating
    the same number such as ``2, 2, 2, 2, 2, ...`` (the actual value depends
    on ``incr``).

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The Dimension from which the ModuloDimension is derived.
    offset : expr-like, optional
        Min value offset. Defaults to 0.
    modulo : int, optional
        The divisor value.
    incr : expr-like, optional
        The iterator increment value. Defaults to ``offset % modulo``.
    origin : expr-like, optional
        The expression -- typically a function of the parent Dimension -- the
        ModuloDimension represents.

    Notes
    -----
    This type should not be instantiated directly in user code; if in need for
    modulo buffered iteration, use SteppingDimension instead.

    About `origin` -- the ModuloDimensions `t0, t1, t2, ...` are generated by the
    compiler to implement modulo iteration along a TimeDimension. For example,
    `t0`'s `origin` may be `t + 0` (where `t` is a SteppingDimension), `t1`'s
    `origin` will then be `t + 1` and so on.
    """

    is_NonlinearDerived = True
    is_Modulo = True
    is_SubIterator = True

    __rkwargs__ = ('offset', 'modulo', 'incr', 'origin')

    def __init_finalize__(self, name, parent, offset=None, modulo=None, incr=None,
                          origin=None, **kwargs):
        super().__init_finalize__(name, parent)

        # Sanity check
        assert modulo is not None or incr is not None

        self._offset = offset or 0
        self._modulo = modulo
        self._incr = incr
        self._origin = origin

    @property
    def offset(self):
        return self._offset

    @property
    def modulo(self):
        return self._modulo

    @property
    def incr(self):
        return self._incr

    @property
    def origin(self):
        return self._origin

    @cached_property
    def symbolic_size(self):
        try:
            return sympy.Number(self.modulo)
        except (TypeError, ValueError):
            pass
        try:
            return sympy.Number(self.incr)
        except (TypeError, ValueError):
            return self.incr

    @cached_property
    def symbolic_min(self):
        if self.modulo is not None:
            return self.offset % self.modulo
        # Make sure we return a symbolic object as this point `offset` may well
        # be a pure Python number
        try:
            return sympy.Number(self.offset)
        except (TypeError, ValueError):
            return self.offset

    @cached_property
    def symbolic_incr(self):
        if self._incr is not None:
            incr = self._incr
        else:
            incr = self.offset
        if self.modulo is not None:
            incr = incr % self.modulo
        # Make sure we return a symbolic object as this point `incr` may well
        # be a pure Python number
        try:
            return sympy.Number(incr)
        except (TypeError, ValueError):
            return incr

    @cached_property
    def bound_symbols(self):
        return set(self.parent.bound_symbols)

    def _arg_defaults(self, **kwargs):
        return {}

    def _arg_values(self, *args, **kwargs):
        return {}

    # Override SymPy arithmetic operators to exploit properties of modular arithmetic

    def __add__(self, other):
        # Exploit compatibility with addition:
        # `a1 ≡ b1 (mod n) and a2 ≡ b2 (mod n)` => `a1 + a2 ≡ b1 + b2 (mod n)`
        try:
            if self.modulo == other.modulo:
                return self.origin + other.origin
        except (AttributeError, TypeError, sympy.SympifyError):
            pass
        return super().__add__(other)

    def __sub__(self, other):
        # Exploit compatibility with subtraction:
        # `a1 ≡ b1 (mod n) and a2 ≡ b2 (mod n)` => `a1 – a2 ≡ b1 – b2 (mod n)`
        try:
            if self.modulo == other.modulo:
                return self.origin - other.origin
        except (AttributeError, TypeError, sympy.SympifyError):
            pass
        return super().__sub__(other)


class AbstractIncrDimension(DerivedDimension):

    """
    Dimension symbol representing a non-contiguous sub-region of a given
    ``parent`` Dimension, with one point every ``step`` points. Thus, if
    ``step == k``, the dimension represents the sequence ``min, min + k,
    min + 2*k, ...``.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The Dimension from which the IncrDimension is derived.
    _min : expr-like
        The minimum point of the Dimension.
    _max : expr-like
        The maximum point of the Dimension.
    step : expr-like, optional
        The distance between two consecutive points. Defaults to the
        symbolic size.
    size : expr-like, optional
        The symbolic size of the Dimension. Defaults to `_max-_min+1`.

    Notes
    -----
    This type should not be instantiated directly in user code.
    """

    is_Incr = True

    __rargs__ = ('name', 'parent', 'symbolic_min', 'symbolic_max')
    __rkwargs__ = ('step', 'size')

    def __init_finalize__(self, name, parent, _min, _max, step=None, size=None, **kwargs):
        super().__init_finalize__(name, parent)
        self._min = _min
        self._max = _max
        self._step = step
        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def _depth(self):
        """
        The depth of `self` in the hierarchy of IncrDimensions.
        """
        return len([i for i in self._defines if i.is_Incr])

    @cached_property
    def step(self):
        if self._step is not None:
            return self._step
        else:
            return Scalar(name=self.size_name, dtype=np.int32, is_const=True)

    @cached_property
    def symbolic_size(self):
        if self.size is not None:
            # Make sure we return a symbolic object as the provided size might
            # be for example a pure int
            try:
                return sympy.Number(self.size)
            except (TypeError, ValueError):
                return self._size
        else:
            # The size must be given as a function of the parent's symbols
            return self.symbolic_max - self.symbolic_min + 1

    @cached_property
    def symbolic_min(self):
        # Make sure we return a symbolic object as the provided min might
        # be for example a pure int
        try:
            return sympy.Number(self._min)
        except (TypeError, ValueError):
            return self._min

    @cached_property
    def symbolic_max(self):
        # Make sure we return a symbolic object as the provided max might
        # be for example a pure int
        try:
            return sympy.Number(self._max)
        except (TypeError, ValueError):
            return self._max

    @cached_property
    def symbolic_incr(self):
        try:
            return sympy.Number(self.step)
        except (TypeError, ValueError):
            return self.step

    @cached_property
    def bound_symbols(self):
        ret = set(self.parent.bound_symbols)
        if self.symbolic_incr.is_Symbol:
            ret.add(self.symbolic_incr)
        return frozenset(ret)


class IncrDimension(AbstractIncrDimension):

    """
    A concrete implementation of an AbstractIncrDimension.

    Notes
    -----
    This type should not be instantiated directly in user code.
    """

    is_SubIterator = True


class BlockDimension(AbstractIncrDimension):

    """
    Dimension symbol for lowering TILABLE Dimensions.
    """

    is_Block = True
    is_PerfKnob = True

    @cached_property
    def _arg_names(self):
        try:
            return (self.step.name,)
        except AttributeError:
            # `step` not a Symbol
            return ()

    def _arg_defaults(self, **kwargs):
        try:
            return {self.step.name: 16}
        except AttributeError:
            # `step` not a Symbol
            return {}

    def _arg_values(self, interval, grid=None, args=None, **kwargs):
        try:
            name = self.step.name
        except AttributeError:
            # `step` not a Symbol
            return {}

        if name in kwargs:
            return {name: kwargs.pop(name)}
        elif isinstance(self.parent, BlockDimension):
            # `self` is a BlockDimension within an outer BlockDimension, but
            # no value supplied -> the sub-block will span the entire block
            return {name: args[self.parent.step.name]}
        else:
            # TODO": Check the args for space order and apply heuristics (e.g.,
            # `2*space_order`?) for even better block sizes
            value = self._arg_defaults()[name]
            if value <= args[self.root.max_name] - args[self.root.min_name] + 1:
                return {name: value}
            else:
                # Avoid OOB (will end up here only in case of tiny iteration spaces)
                return {name: 1}

    def _arg_check(self, args, *_args):
        try:
            name = self.step.name
        except AttributeError:
            # `step` not a Symbol
            return

        value = args[name]
        if isinstance(self.parent, BlockDimension):
            # sub-BlockDimensions must be perfect divisors of their parent
            parent_value = args[self.parent.step.name]
            if parent_value % value > 0:
                raise InvalidArgument("Illegal block size `%s=%d`: sub-block sizes "
                                      "must divide the parent block size evenly (`%s=%d`)"
                                      % (name, value, self.parent.step.name,
                                         parent_value))
        else:
            if value < 0:
                raise InvalidArgument("Illegal block size `%s=%d`: it should be > 0"
                                      % (name, value))
            if value > args[self.root.max_name] - args[self.root.min_name] + 1:
                # Avoid OOB
                raise InvalidArgument("Illegal block size `%s=%d`: it's greater than the "
                                      "iteration range and it will cause an OOB access"
                                      % (name, value))


class CustomDimension(BasicDimension):

    """
    Dimension defining an iteration space with known size. Unlike a
    DefaultDimension, a CustomDimension:

        * Provides more freedom -- the symbolic_{min,max,size} can be set at will;
        * It provides no runtime argument values.

    Notes
    -----
    This type should not be instantiated directly in user code.
    """

    is_Custom = True

    __rkwargs__ = ('symbolic_min', 'symbolic_max', 'symbolic_size', 'parent',
                   'local')

    def __init_finalize__(self, name, symbolic_min=None, symbolic_max=None,
                          symbolic_size=None, parent=None, local=True, **kwargs):
        self._symbolic_min = symbolic_min
        self._symbolic_max = symbolic_max
        self._symbolic_size = symbolic_size
        self._parent = parent or BOTTOM
        self._local = local
        super().__init_finalize__(name)

    @property
    def is_Derived(self):
        return self._parent is not None

    @property
    def is_NonlinearDerived(self):
        return self.is_Derived and self.parent.is_NonlinearDerived

    @property
    def parent(self):
        return self._parent

    @property
    def index(self):
        return self.parent or self

    @property
    def root(self):
        if self.is_Derived:
            return self.parent.root
        else:
            return self

    @property
    def spacing(self):
        if self.is_Derived:
            return self.parent.spacing
        else:
            return self._spacing

    @property
    def local(self):
        return self._local

    @property
    def bound_symbols(self):
        ret = {self.symbolic_min, self.symbolic_max, self.symbolic_size}
        if self.is_Derived:
            ret.update(self.parent.bound_symbols)
        return frozenset(i for i in ret if i.is_Symbol)

    @property
    def _maybe_distributed(self):
        return not self.local

    @cached_property
    def _defines(self):
        ret = frozenset({self})
        if self.is_Derived:
            ret |= self.parent._defines
        return ret

    @cached_property
    def symbolic_min(self):
        try:
            return sympy.Number(self._symbolic_min)
        except (TypeError, ValueError):
            pass
        if self._symbolic_min is None:
            return super().symbolic_min
        else:
            return self._symbolic_min

    @cached_property
    def symbolic_max(self):
        try:
            return sympy.Number(self._symbolic_max)
        except (TypeError, ValueError):
            pass
        if self._symbolic_max is None:
            return super().symbolic_max
        else:
            return self._symbolic_max

    @cached_property
    def symbolic_size(self):
        try:
            return sympy.Number(self._symbolic_size)
        except (TypeError, ValueError):
            pass
        if self._symbolic_size is None:
            v = self.symbolic_max - self.symbolic_min + 1
            if v.is_Number:
                return v
            else:
                return super().symbolic_size
        else:
            return self._symbolic_size

    def _arg_defaults(self, **kwargs):
        return {}

    def _arg_values(self, *args, **kwargs):
        return {}

    def _arg_check(self, *args):
        """A CustomDimension performs no runtime checks."""
        return


class DynamicDimensionMixin:

    """
    A mixin to create Dimensions producing non-const Symbols.
    """

    @cached_property
    def symbolic_size(self):
        return Scalar(name=self.size_name, dtype=np.int32)

    @cached_property
    def symbolic_min(self):
        return Scalar(name=self.min_name, dtype=np.int32)

    @cached_property
    def symbolic_max(self):
        return Scalar(name=self.max_name, dtype=np.int32)


class DynamicDimension(DynamicDimensionMixin, BasicDimension):
    pass


class DynamicSubDimension(DynamicDimensionMixin, SubDimension):

    @classmethod
    def _symbolic_thickness(cls, name):
        return (Scalar(name="%s_ltkn" % name, dtype=np.int32, nonnegative=True),
                Scalar(name="%s_rtkn" % name, dtype=np.int32, nonnegative=True))


class StencilDimension(BasicDimension):

    """
    Dimension symbol representing the points of a stencil.

    Parameters
    ----------
    name : str
        Name of the dimension.
    _min : expr-like
        The minimum point of the stencil.
    _max : expr-like
        The maximum point of the stencil.
    spacing : expr-like, optional
        The space between two stencil points.
    """

    is_Stencil = True

    __rargs__ = BasicDimension.__rargs__ + ('_min', '_max')
    __rkwargs__ = BasicDimension.__rkwargs__ + ('step',)

    def __init_finalize__(self, name, _min, _max, spacing=1, step=1,
                          **kwargs):
        self._spacing = sympy.sympify(spacing)

        if not is_integer(_min):
            raise ValueError("Expected integer `min` (got %s)" % _min)
        if not is_integer(_max):
            raise ValueError("Expected integer `max` (got %s)" % _max)
        if not is_integer(self._spacing):
            raise ValueError("Expected integer `spacing` (got %s)" % self._spacing)
        if not is_integer(step):
            raise ValueError("Expected integer `step` (got %s)" % step)

        self._min = _min
        self._max = _max
        self._step = step

        self._size = _max - _min + 1

        if self._size < 1:
            raise ValueError("Expected size greater than 0 (got %s)" % self._size)

    @property
    def step(self):
        return self._step

    @property
    def backward(self):
        return self.step < 0

    @cached_property
    def symbolic_size(self):
        return sympy.Number(self._size)

    @cached_property
    def symbolic_min(self):
        return sympy.Number(self._min)

    @cached_property
    def symbolic_max(self):
        return sympy.Number(self._max)

    @property
    def range(self):
        return range(self._min, self._max + 1)

    def transpose(self):
        return StencilDimension(self.name, -self._max, -self._min, step=-1)

    @property
    def _arg_names(self):
        return ()

    def _arg_defaults(self, **kwargs):
        return {}

    def _arg_values(self, *args, **kwargs):
        return {}


class VirtualDimension(CustomDimension):

    """
    Dimension symbol representing a mock iteration space, which as such
    is eventually ditched by the compiler.

    Mock iteration spaces are used for compilation purposes only, typically
    to bind objects such as Guards and Syncs to a specific point in the
    program flow.

    Examples
    --------
    To generate nested conditionals within the same loop nest, one may use
    VirtualDimensions to represent the different branches of the conditionals.

        .. code-block:: C

        for (int i = i_m; i <= i_M; i += 1)
          if (i < 10)
            if (i < 5)
              do A(i);
            if (i >= 5)
              do B(i);

    The above code can be obtained by using one VirtualDimension for the
    `i < 5` conditional and another VirtualDimension for the `i >= 5` conditional.
    """

    is_Virtual = True

    __rkwargs__ = ('parent',)

    def __init_finalize__(self, name, parent=None):
        super().__init_finalize__(name, parent=parent,
                                  symbolic_min=sympy.S.Zero,
                                  symbolic_max=sympy.S.Zero)


# ***
# The Dimensions below are created by Devito and may eventually be
# accessed in user code to e.g. construct or manipulate Eqs


class SteppingDimension(DerivedDimension):

    """
    Symbol defining a convex iteration sub-space derived from a ``parent``
    Dimension, which cyclically produces a finite range of values, such
    as ``0, 1, 2, 0, 1, 2, 0, ...`` (also referred to as "modulo buffered
    iteration").

    SteppingDimension is most commonly used to represent a time-stepping Dimension.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    """

    is_NonlinearDerived = True
    is_Stepping = True
    is_SubIterator = True

    @property
    def symbolic_min(self):
        return self.parent.symbolic_min

    @property
    def symbolic_max(self):
        return self.parent.symbolic_max

    @property
    def _arg_names(self):
        return (self.min_name, self.max_name, self.name) + self.parent._arg_names

    def _arg_defaults(self, _min=None, **kwargs):
        """
        A map of default argument values defined by this dimension.

        Parameters
        ----------
        _min : int, optional
            Minimum point as provided by data-carrying objects.

        Notes
        -----
        A SteppingDimension does not know its max point and therefore
        does not have a size argument.
        """
        return {self.parent.min_name: _min}

    def _arg_values(self, *args, **kwargs):
        """
        The argument values provided by a SteppingDimension are those
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


# *** Utils


class IndexAccessFunction(sympy.Add):

    """
    A IndexAccessFunction is an expression used to index into a Function.
    """

    # Prioritize self's __add__ and __sub__ to construct AffineIndexAccessFunction
    _op_priority = sympy.Add._op_priority + 1.

    def __eq__(self, other):
        return super().__eq__(other)

    __hash__ = sympy.Add.__hash__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self.func(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self.func(self, other)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self.func(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return self.func(other, -self)

    def __mod__(self, other):
        return sympy.Mod(sympy.Add(*self.args), other)


class AffineIndexAccessFunction(IndexAccessFunction):
    """
    An AffineIndexAccessFunction is the sum of three operands:

        * the "main" Dimension (in practice a SpaceDimension or a TimeDimension);
        * an offset (number or symbolic expression, of dtype integer).
        * one or more StencilDimensions;

    Examples
    --------
    The AffineIndexAccessFunction `x + sd + 3`, with `sd in [-2, 2]`, represents
    the index access functions `[x + 1, x + 2, x + 3, x + 4, x + 5]`
    """

    def __new__(cls, *args, **kwargs):
        # `args` may contain arbitrarily complicated expressions, so first of all
        # we let SymPy simplify it, then we process the args and see if the
        # resulting expression is indeed an AffineIndexAccessFunction
        add = sympy.Add(*args, **kwargs)
        if not isinstance(add, sympy.Add):
            # E.g., reduced to a Symbol
            return add

        d = 0
        sds = []
        ofs_items = []
        for a in add.args:
            if isinstance(a, StencilDimension):
                sds.append(a)
            elif isinstance(a, Dimension):
                d = cls._separate_dims(d, a, ofs_items)
                if d is None:
                    return add
            elif isinstance(a, AffineIndexAccessFunction):
                if sds and a.sds:
                    return add
                d = cls._separate_dims(d, a.d, ofs_items)
                if d is None:
                    return add
                sds = list(a.sds or sds)
                ofs_items.append(a.ofs)
            else:
                ofs_items.append(a)

        ofs = sympy.Add(*[i for i in ofs_items if i is not None])
        if not all(is_integer(i) or i.is_Symbol for i in ofs.free_symbols):
            return add

        sds = tuple(sds)

        obj = IndexAccessFunction.__new__(cls, d, ofs, *sds)

        if isinstance(obj, AffineIndexAccessFunction):
            obj.d = d
            obj.ofs = ofs
            obj.sds = sds
        else:
            # E.g., SymPy simplified it to Zero or something else
            pass

        return obj

    @classmethod
    def _separate_dims(cls, d0, d1, ofs_items):
        if d0 == 0 and d1 == 0:
            return 0
        elif d0 == 0 and isinstance(d1, Dimension):
            return d1
        elif d1 == 0 and isinstance(d0, Dimension):
            return d0
        elif isinstance(d0, Dimension) and isinstance(d1, AbstractIncrDimension):
            # E.g., `time + x0_blk0` after skewing
            ofs_items.append(d1)
            return d0
        elif isinstance(d1, Dimension) and isinstance(d0, AbstractIncrDimension):
            # E.g., `time + x0_blk0` after skewing
            ofs_items.append(d0)
            return d1
        else:
            return None


def dimensions(names, n=1):
    if n > 1:
        return tuple(Dimension('%s%s' % (names, i)) for i in range(n))
    else:
        assert type(names) is str
        return tuple(Dimension(i) for i in names.split())


BOTTOM = Dimension(name='⊥')

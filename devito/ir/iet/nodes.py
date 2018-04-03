"""The Iteration/Expression Tree (IET) hierarchy."""

from __future__ import absolute_import

import abc
import inspect
from cached_property import cached_property
from collections import Iterable, OrderedDict, namedtuple

import cgen as c

from devito.cgen_utils import ccode
from devito.ir.equations import ClusterizedEq
from devito.ir.iet import (IterationProperty, SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC,
                           VECTOR, ELEMENTAL, REMAINDER, WRAPPABLE, tagger, ntags)
from devito.ir.support import Forward, detect_io
from devito.dimension import Dimension
from devito.symbolics import as_symbol
from devito.tools import as_tuple, filter_ordered, filter_sorted, flatten
from devito.types import AbstractFunction, Symbol, Indexed

__all__ = ['Node', 'Block', 'Denormals', 'Expression', 'Element', 'Callable',
           'Call', 'Conditional', 'Iteration', 'List', 'LocalExpression', 'TimedList',
           'UnboundedIndex', 'MetaCall', 'ArrayCast', 'PointerCast']


class Node(object):

    __metaclass__ = abc.ABCMeta

    is_Node = True
    is_Block = False
    is_Iteration = False
    is_IterationFold = False
    is_Expression = False
    is_Callable = False
    is_Call = False
    is_List = False
    is_Element = False

    """
    :attr:`_traversable`. The traversable fields of the Node; that is, fields
    walked over by a :class:`Visitor`. All arguments in __init__ whose name
    appears in this list are treated as traversable fields.
    """
    _traversable = []

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames = inspect.getargspec(cls.__init__).args
        obj._args = {k: v for k, v in zip(argnames[1:], args)}
        obj._args.update(kwargs.items())
        obj._args.update({k: None for k in argnames[1:] if k not in obj._args})
        return obj

    def _rebuild(self, *args, **kwargs):
        """Reconstruct self. None of the embedded Sympy expressions are rebuilt."""
        handle = self._args.copy()  # Original constructor arguments
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict([(k, v) for k, v in zip(argnames, args)]))
        handle.update(kwargs)
        return type(self)(**handle)

    @property
    def ccode(self):
        """Generate C code.

        This is a shorthand for

            .. code-block:: python

              from devito.ir.iet import CGen
              CGen().visit(self)
        """
        from devito.ir.iet.visitors import CGen
        return CGen().visit(self)

    @property
    def view(self):
        """
        Generate a representation of the Iteration/Expression tree rooted in ``self``.
        """
        from devito.ir.iet.visitors import printAST
        return printAST(self)

    @property
    def children(self):
        """Return the traversable children."""
        return tuple(getattr(self, i) for i in self._traversable)

    @property
    def args(self):
        """Arguments used to construct the Node."""
        return self._args.copy()

    @property
    def args_frozen(self):
        """Arguments used to construct the Node that cannot be traversed."""
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    def __str__(self):
        return str(self.ccode)

    @abc.abstractproperty
    def functions(self):
        """
        Return all :class:`AbstractFunction` objects used by this :class:`Node`.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def free_symbols(self):
        """
        Return all :class:`Symbol` objects used by this :class:`Node`.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def defines(self):
        """
        Return all :class:`Symbol` objects defined by this :class:`Node`.
        """
        raise NotImplementedError()


class Block(Node):

    """A sequence of nodes, wrapped in a block {...}."""

    is_Block = True

    _traversable = ['body']

    def __init__(self, header=None, body=None, footer=None):
        self.header = as_tuple(header)
        self.body = as_tuple(body)
        self.footer = as_tuple(footer)

    def __repr__(self):
        return "<%s (%d, %d, %d)>" % (self.__class__.__name__, len(self.header),
                                      len(self.body), len(self.footer))


class List(Block):

    """A sequence of nodes."""

    is_List = True


class Element(Node):

    """A generic node in an Iteration/Expression tree. Can be a comment,
    a statement, ..."""

    is_Element = True

    def __init__(self, element):
        assert isinstance(element, (c.Comment, c.Statement, c.Value, c.Initializer,
                                    c.Pragma, c.Line, c.Assign, c.POD))
        self.element = element

    def __repr__(self):
        return "Element::\n\t%s" % (self.element)


class Call(Node):

    """A function call."""

    is_Call = True

    def __init__(self, name, params=None):
        self.name = name
        self.params = as_tuple(params)

    def __repr__(self):
        return "Call::\n\t%s(...)" % self.name

    @property
    def functions(self):
        """Return all :class:`Symbol` objects used by this :class:`Call`."""
        return tuple(p for p in self.params if isinstance(p, AbstractFunction))

    @cached_property
    def free_symbols(self):
        """Return all :class:`Symbol` objects used by this :class:`Call`."""
        free = tuple(set(flatten(p.free_symbols for p in self.params)))
        # HACK: Filter dimensions to avoid them on popping onto outer parameters
        free = tuple(s for s in free if not isinstance(s, Dimension))
        return free

    @property
    def defines(self):
        """Return all :class:`Symbol` objects defined by this :class:`Call`."""
        return ()


class Expression(Node):

    """A node encapsulating a SymPy equation."""

    is_Expression = True

    def __init__(self, expr):
        assert isinstance(expr, ClusterizedEq)
        assert isinstance(expr.lhs, (Symbol, Indexed))
        self.expr = expr

        self._functions = tuple(filter_ordered(flatten(detect_io(expr, relax=True))))

        self.dimensions = flatten(i.indices for i in self.functions if i.is_Indexed)
        self.dimensions = filter_ordered(self.dimensions)

    def __repr__(self):
        return "<%s::%s>" % (self.__class__.__name__,
                             filter_ordered([f.func for f in self.functions]))

    def substitute(self, substitutions):
        """Apply substitutions to the expression.

        :param substitutions: Dict containing the substitutions to apply to
                              the stored expression.
        """
        self.expr = self.expr.xreplace(substitutions)

    @property
    def dtype(self):
        return self.expr.dtype

    @property
    def output(self):
        """
        Return the symbol written by this Expression.
        """
        return self.expr.lhs

    @property
    def functions(self):
        return self._functions

    @property
    def defines(self):
        """
        Return any symbols an :class:`Expression` may define.
        """
        return (self.write, ) if self.write.is_Scalar else ()

    @property
    def write(self):
        """
        Return the function written by this Expression.
        """
        return self.expr.lhs.base.function

    @property
    def is_scalar(self):
        """
        Return True if a scalar expression, False otherwise.
        """
        return self.expr.lhs.is_Symbol

    @property
    def is_tensor(self):
        """
        Return True if a tensor expression, False otherwise.
        """
        return not self.is_scalar

    @property
    def is_increment(self):
        """
        Return True if the write is actually an associative and commutative increment.
        """
        return self.expr.is_Increment

    @property
    def shape(self):
        """
        Return the shape of the written LHS.
        """
        return () if self.is_scalar else self.expr.lhs.shape

    @property
    def free_symbols(self):
        """Return all :class:`Symbol` objects used by this :class:`Expression`."""
        return tuple(self.expr.free_symbols)


class Iteration(Node):
    """Implement a for-loop over nodes.

    :param nodes: Single or list of :class:`Node` objects defining the loop body.
    :param dimension: :class:`Dimension` object over which to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    :param index: Symbol to be used as iteration variable.
    :param offsets: A 2-tuple of start and end offsets to honour in the loop.
    :param direction: The :class:`IterationDirection` of the Iteration. Defaults
                      to ``Forward``.
    :param properties: A bag of :class:`IterationProperty` objects, decorating
                       the Iteration (sequential, parallel, vectorizable, ...).
    :param pragmas: A bag of pragmas attached to this Iteration.
    :param uindices: a bag of UnboundedIndex objects, representing free iteration
                     variables (i.e., the Iteration end point is independent of
                     any of these UnboundedIndex).
    """

    is_Iteration = True

    _traversable = ['nodes']

    def __init__(self, nodes, dimension, limits, index=None, offsets=None,
                 direction=None, properties=None, pragmas=None, uindices=None):
        # Ensure we deal with a list of Expression objects internally
        self.nodes = as_tuple(nodes)

        self.dim = dimension
        self.index = index or self.dim.name
        self.direction = direction or Forward

        # Generate loop limits
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = tuple(limits)
        else:
            self.limits = (0, limits, 1)

        # Record offsets to later adjust loop limits accordingly
        self.offsets = (0, 0) if offsets is None else as_tuple(offsets)
        assert len(self.offsets) == 2

        # Track this Iteration's properties, pragmas and unbounded indices
        properties = as_tuple(properties)
        assert (i in IterationProperty._KNOWN for i in properties)
        self.properties = as_tuple(filter_sorted(properties))
        self.pragmas = as_tuple(pragmas)
        self.uindices = as_tuple(uindices)
        assert all(isinstance(i, UnboundedIndex) for i in self.uindices)

    def __repr__(self):
        properties = ""
        if self.properties:
            properties = [str(i) for i in self.properties]
            properties = "WithProperties[%s]::" % ",".join(properties)
        index = self.index
        if self.uindices:
            index += '[%s]' % ','.join(ccode(i.index) for i in self.uindices)
        return "<%sIteration %s; %s>" % (properties, index, self.limits)

    @property
    def defines(self):
        """
        Return any symbols defined in the :class:`Iteration` header.
        """
        dims = (self.dim, self.dim.parent) if self.dim.is_Derived else (self.dim,)
        return dims + tuple(i.name for i in self.uindices)

    @property
    def is_Linear(self):
        return len(self.uindices) == 0

    @property
    def is_Sequential(self):
        return SEQUENTIAL in self.properties

    @property
    def is_Parallel(self):
        return PARALLEL in self.properties

    @property
    def is_ParallelAtomic(self):
        return PARALLEL_IF_ATOMIC in self.properties

    @property
    def is_ParallelRelaxed(self):
        return self.is_Parallel or self.is_ParallelAtomic

    @property
    def is_Vectorizable(self):
        return VECTOR in self.properties

    @property
    def is_Elementizable(self):
        return ELEMENTAL in self.properties

    @property
    def is_Wrappable(self):
        return WRAPPABLE in self.properties

    @property
    def is_Remainder(self):
        return REMAINDER in self.properties

    @property
    def tag(self):
        for i in self.properties:
            if i.name == 'tag':
                return i.val
        return None

    def retag(self, tag_value=None):
        """
        Create a new Iteration object which is identical to ``self``, except
        for the tag. If provided, ``tag_value`` is used as new tag; otherwise,
        an internally generated tag is used.
        """
        if self.tag is None:
            return self._rebuild()
        properties = [tagger(tag_value or (ntags() + 1)) if i.name == 'tag' else i
                      for i in self.properties]
        return self._rebuild(properties=properties)

    @property
    def bounds_symbolic(self):
        """Return a 2-tuple representing the symbolic bounds of the object."""
        start = self.limits[0]
        end = self.limits[1]
        try:
            start = as_symbol(start)
        except TypeError:
            # Already a symbolic expression
            pass
        try:
            end = as_symbol(end)
        except TypeError:
            # Already a symbolic expression
            pass
        return (start + as_symbol(self.offsets[0]), end + as_symbol(self.offsets[1]))

    @property
    def extent_symbolic(self):
        """
        Return the symbolic extent of the Iteration.
        """
        return self.bounds_symbolic[1] - self.bounds_symbolic[0] + 1

    @property
    def start_symbolic(self):
        """
        Return the symbolic start of the Iteration.
        """
        return self.bounds_symbolic[0]

    @property
    def end_symbolic(self):
        """
        Return the symbolic end of the Iteration.
        """
        return self.bounds_symbolic[1]

    @property
    def incr_symbolic(self):
        """
        Return the symbolic extent of the Iteration.
        """
        return self.limits[2]

    def bounds(self, start=None, finish=None):
        """Return the start and end points of the Iteration if the limits are
        available (either statically known or provided through ``start``/
        ``finish``). ``None`` is used as a placeholder in the returned 2-tuple
        if a limit is unknown."""
        lower = start if start is not None else self.limits[0]
        upper = finish if finish is not None else self.limits[1]

        lower = lower + self.offsets[0]
        upper = upper + self.offsets[1]

        return (lower, upper)

    def extent(self, start=None, finish=None):
        """Return the number of iterations executed if the limits are known,
        ``None`` otherwise."""
        start, finish = self.bounds(start, finish)
        try:
            return finish - start + 1
        except TypeError:
            return None

    def start(self, start=None):
        """Return the start point of the Iteration if the lower limit is known,
        ``None`` otherwise."""
        return self.bounds(start)[0]

    def end(self, finish=None):
        """Return the end point of the Iteration if the upper limit is known,
        ``None`` otherwise."""
        return self.bounds(finish=finish)[1]

    @property
    def functions(self):
        """
        Return all :class:`Function` objects used in the header of
        this :class:`Iteration`.
        """
        return ()

    @property
    def write(self):
        """Return all :class:`Function` objects written to in this :class:`Iteration`"""
        return []

    @property
    def free_symbols(self):
        """
        Return all :class:`Symbol` objects used in the header of this
        :class:`Iteration`.
        """
        return tuple(self.start_symbolic.free_symbols) \
            + tuple(self.end_symbolic.free_symbols) \
            + tuple(flatten(ui.free_symbols for ui in self.uindices))


class Callable(Node):

    """A node representing a callable function.

    :param name: The name of the callable.
    :param body: A :class:`Node` or an iterable of :class:`Node` objects representing
                 the body of the callable.
    :param retval: The type of the value returned by the callable.
    :param parameters: An iterable of :class:`AbstractFunction`s in input to the
                       callable, or ``None`` if the callable takes no parameter.
    :param prefix: An iterable of qualifiers to prepend to the callable declaration.
                   The default value is ('static', 'inline').
    """

    is_Callable = True

    _traversable = ['body']

    def __init__(self, name, body, retval, parameters=None, prefix=('static', 'inline')):
        self.name = name
        self.body = as_tuple(body)
        self.retval = retval
        self.prefix = prefix
        self.parameters = as_tuple(parameters)

    def __repr__(self):
        parameters = ",".join(['void*' if i.is_Object else c.dtype_to_ctype(i.dtype)
                               for i in self.parameters])
        body = "\n\t".join([str(s) for s in self.body])
        return "Function[%s]<%s; %s>::\n\t%s" % (self.name, self.retval, parameters, body)


class Conditional(Node):

    """
    A node to express if-then-else blocks.

    :param condition: A SymPy expression representing the if condition.
    :param then_body: Single or iterable of :class:`Node` objects defining the
                      body of the 'then' part of the if-then-else.
    :param else_body: (Optional) Single or iterable of :class:`Node` objects
                      defining the body of the 'else' part of the if-then-else.
    """

    is_Conditional = True

    _traversable = ['then_body', 'else_body']

    def __init__(self, condition, then_body, else_body=None):
        self.condition = condition
        self.then_body = as_tuple(then_body)
        self.else_body = as_tuple(else_body)

    def __repr__(self):
        if self.else_body:
            return "<[%s] ? [%s] : [%s]>" %\
                (ccode(self.condition), repr(self.then_body), repr(self.else_body))
        else:
            return "<[%s] ? [%s]" % (ccode(self.condition), repr(self.then_body))

    @property
    def free_symbols(self):
        """
        Return all :class:`Symbol` objects used in the condition of this
        :class:`Conditional`.
        """
        return tuple(self.condition.free_symbols)


# Utilities

class TimedList(List):

    """Wrap a Node with C-level timers."""

    def __init__(self, lname, gname, body):
        """
        Initialize a TimedList object.

        :param lname: Timer name in the local scope.
        :param gname: Name of the global struct tracking all timers.
        :param body: Timed block of code.
        """
        self._name = lname
        # TODO: need omp master pragma to be thread safe
        header = [c.Statement("struct timeval start_%s, end_%s" % (lname, lname)),
                  c.Statement("gettimeofday(&start_%s, NULL)" % lname)]
        footer = [c.Statement("gettimeofday(&end_%s, NULL)" % lname),
                  c.Statement(("%(gn)s->%(ln)s += " +
                               "(double)(end_%(ln)s.tv_sec-start_%(ln)s.tv_sec)+" +
                               "(double)(end_%(ln)s.tv_usec-start_%(ln)s.tv_usec)" +
                               "/1000000") % {'gn': gname, 'ln': lname})]
        super(TimedList, self).__init__(header, body, footer)

    def __repr__(self):
        body = "\n\t".join([str(s) for s in self.body])
        return "%s::\n\t%s" % (self.__class__.__name__, body)

    @property
    def name(self):
        return self._name


class Denormals(List):

    """Macros to make sure denormal numbers are flushed in hardware."""

    def __init__(self, header=None, body=None, footer=None):
        b = [Element(c.Comment('Flush denormal numbers to zero in hardware')),
             Element(c.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)')),
             Element(c.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'))]
        super(Denormals, self).__init__(header, b, footer)

    def __repr__(self):
        return "<DenormalsMacro>"


class ArrayCast(Node):

    """
    A node encapsulating a cast of a raw C pointer to a
    multi-dimensional array.
    """

    def __init__(self, function):
        self.function = function

    @property
    def functions(self):
        """
        Return all :class:`Function` objects used by this :class:`ArrayCast`
        """
        return (self.function,)

    @property
    def defines(self):
        """
        Return the base symbol an :class:`ArrayCast` defines.
        """
        return ()

    @property
    def free_symbols(self):
        """
        Return the symbols required to perform an :class:`ArrayCast`.

        This includes the :class:`AbstractFunction` object that
        defines the data, as well as the dimension sizes.
        """
        sizes = flatten(s.free_symbols for s in self.function.symbolic_shape[1:])
        return (self.function, ) + as_tuple(sizes)


class PointerCast(Node):

    """
    A node encapsulating a cast of a raw C pointer to a
    struct or object.
    """

    def __init__(self, object):
        self.object = object

    @property
    def functions(self):
        """
        Return all :class:`Function` objects used by this :class:`PointerCast`
        """
        return ()

    @property
    def defines(self):
        """
        Return the base symbol an :class:`PointerCast` defines.
        """
        return ()

    @property
    def free_symbols(self):
        """
        Return the symbols required to perform an :class:`PointerCast`.

        This includes the :class:`AbstractFunction` object that
        defines the data, as well as the dimension sizes.
        """
        return (self.object, )


class LocalExpression(Expression):

    """
    A node encapsulating a SymPy equation which also defines its LHS.
    """

    @property
    def defines(self):
        """
        Return any symbols an :class:`LocalExpression` may define.
        """
        return (self.write, )


class UnboundedIndex(object):

    """
    A generic loop iteration index that can be used in a :class:`Iteration` to
    add a non-linear traversal of the iteration space.
    """

    def __init__(self, index, start=0, step=None, dim=None, expr=None):
        self.name = index
        self.index = index
        self.dim = dim
        self.expr = expr

        try:
            self.start = as_symbol(start)
        except TypeError:
            self.start = start

        try:
            if step is None:
                self.step = index + 1
            else:
                self.step = as_symbol(step)
        except TypeError:
            self.step = step

    @property
    def free_symbols(self):
        """
        Return the symbols used by this :class:`UnboundedIndex`.
        """
        free = self.index.free_symbols
        free.update(self.start.free_symbols)
        free.update(self.step.free_symbols)
        return tuple(free)


MetaCall = namedtuple('MetaCall', 'root local')
"""
Metadata for :class:`Callable`s. ``root`` is a pointer to the callable
Iteration/Expression tree. ``local`` is a boolean indicating whether the
definition of the callable is known or not.
"""

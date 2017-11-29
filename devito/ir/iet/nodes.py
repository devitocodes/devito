"""The Iteration/Expression Tree (IET) hierarchy."""

from __future__ import absolute_import

import inspect
from collections import Iterable, OrderedDict

import cgen as c
from sympy import Eq, Indexed, Symbol

from devito.cgen_utils import ccode
from devito.ir.iet import (IterationProperty, SEQUENTIAL, PARALLEL,
                           VECTOR, ELEMENTAL, REMAINDER, WRAPPABLE,
                           tagger, ntags)
from devito.ir.support import Stencil
from devito.symbolics import as_symbol, retrieve_terminals
from devito.tools import as_tuple, filter_ordered, filter_sorted, flatten
from devito.arguments import ArgumentProvider, Argument
import devito.types as types

__all__ = ['Node', 'Block', 'Denormals', 'Expression', 'Element', 'Callable',
           'Call', 'Iteration', 'List', 'LocalExpression', 'TimedList',
           'UnboundedIndex']


class Node(object):

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
        return ()

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

    @property
    def children(self):
        return (self.body,)


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


class Expression(Node):

    """A node encapsulating a single SymPy equation."""

    is_Expression = True

    def __init__(self, expr, dtype=None):
        assert isinstance(expr, Eq)
        assert isinstance(expr.lhs, (Symbol, Indexed))
        self.expr = expr
        self.dtype = dtype

        # Traverse /expression/ to determine meta information
        # Note: at this point, expressions have already been indexified
        self.reads = [i for i in retrieve_terminals(self.expr.rhs)
                      if isinstance(i, (types.Indexed, types.Symbol))]
        self.reads = filter_ordered(self.reads)
        self.functions = [self.write] + [i.base.function for i in self.reads]
        self.functions = filter_ordered(self.functions)
        # Filter collected dimensions and functions
        self.dimensions = flatten(i.indices for i in self.functions)
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
    def output(self):
        """
        Return the symbol written by this Expression.
        """
        return self.expr.lhs

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
    def shape(self):
        """
        Return the shape of the written LHS.
        """
        return () if self.is_scalar else self.expr.lhs.shape

    @property
    def stencil(self):
        """Compute the stencil of the expression."""
        return Stencil(self.expr)


class Iteration(Node):
    """Implement a for-loop over nodes.

    :param nodes: Single or list of :class:`Node` objects defining the loop body.
    :param dimension: :class:`Dimension` object over which to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    :param index: Symbol to be used as iteration variable.
    :param offsets: Optional map list of offsets to honour in the loop.
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
                 properties=None, pragmas=None, uindices=None):
        # Ensure we deal with a list of Expression objects internally
        nodes = as_tuple(nodes)
        self.nodes = as_tuple([n if isinstance(n, Node) else Expression(n)
                               for n in nodes])
        assert all(isinstance(i, Node) for i in self.nodes)

        self.dim = dimension
        self.index = index or self.dim.name
        # Store direction, as it might change on the dimension
        # before we use it during code generation.
        self.reverse = self.dim.reverse

        # Generate loop limits
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = list(limits)
        else:
            self.limits = list((0, limits, 1))

        # Record offsets to later adjust loop limits accordingly
        self.offsets = [0, 0]
        for off in (offsets or {}):
            self.offsets[0] = min(self.offsets[0], int(off))
            self.offsets[1] = max(self.offsets[1], int(off))

        # Track this Iteration's properties, pragmas and unbounded indices
        properties = as_tuple(properties)
        assert (i in IterationProperty._KNOWN for i in properties)
        self.properties = as_tuple(filter_sorted(properties, key=lambda i: i.name))
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
    def is_Open(self):
        return self.dim.size is None

    @property
    def is_Closed(self):
        return not self.is_Open

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
        return (start - as_symbol(self.offsets[0]), end - as_symbol(self.offsets[1]))

    @property
    def extent_symbolic(self):
        """
        Return the symbolic extent of the Iteration.
        """
        return self.bounds_symbolic[1] - self.bounds_symbolic[0]

    @property
    def start_symbolic(self):
        """
        Return the symbolic extent of the Iteration.
        """
        return self.bounds_symbolic[0]

    @property
    def end_symbolic(self):
        """
        Return the symbolic extent of the Iteration.
        """
        return self.bounds_symbolic[1]

    def bounds(self, start=None, finish=None):
        """Return the start and end points of the Iteration if the limits are
        available (either statically known or provided through ``start``/
        ``finish``). ``None`` is used as a placeholder in the returned 2-tuple
        if a limit is unknown."""
        lower = start if start is not None else self.limits[0]
        upper = finish if finish is not None else self.limits[1]
        if lower and self.offsets[0]:
            lower = lower - self.offsets[0]

        if upper and self.offsets[1]:
            upper = upper - self.offsets[1]

        return (lower, upper)

    def extent(self, start=None, finish=None):
        """Return the number of iterations executed if the limits are known,
        ``None`` otherwise."""
        start, finish = self.bounds(start, finish)
        try:
            return finish - start
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
    def children(self):
        """Return the traversable children."""
        return (self.nodes,)


class Callable(Node):

    """A node representing a function.

    :param name: The name of the function.
    :param body: A :class:`Node` or an iterable of :class:`Node` objects representing
                 the body of the function.
    :param retval: The type of the value returned by the function.
    :param parameters: An iterable of :class:`SymbolicData` objects in input to the
                       function, or ``None`` if the function takes no parameter.
    :param prefix: An iterable of qualifiers to prepend to the function declaration.
                   The default value is ('static', 'inline').
    """

    is_Callable = True

    _traversable = ['body']

    def __init__(self, name, body, retval, parameters=None, prefix=('static', 'inline')):
        self.name = name
        self.body = as_tuple(body)
        self.retval = retval
        self.prefix = prefix

        if all(isinstance(i, ArgumentProvider) for i in parameters):
            args = flatten([i.rtargs for i in parameters])
        else:
            assert(all(isinstance(i, Argument) for i in parameters))
            args = parameters
        self.parameters = as_tuple(args)

    def __repr__(self):
        parameters = ",".join(['void*' if i.is_PtrArgument else c.dtype_to_ctype(i.dtype)
                               for i in self.parameters])
        body = "\n\t".join([str(s) for s in self.body])
        return "Function[%s]<%s; %s>::\n\t%s" % (self.name, self.retval, parameters, body)

    @property
    def children(self):
        return (self.body,)


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


class LocalExpression(Expression):

    """
    A node encapsulating a single SymPy equation with known data type,
    represented as a NumPy data type.
    """

    def __init__(self, expr, dtype):
        super(LocalExpression, self).__init__(expr)
        self.dtype = dtype


class UnboundedIndex(object):

    """
    A generic loop iteration index that can be used in a :class:`Iteration` to
    add a non-linear traversal of the iteration space.
    """

    def __init__(self, index, start=0, step=None):
        self.index = index
        self.start = start
        self.step = index + 1 if step is None else step

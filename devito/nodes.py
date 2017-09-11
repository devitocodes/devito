"""The Iteration/Expression hierarchy."""

from __future__ import absolute_import

import inspect
from collections import Iterable, OrderedDict

import cgen as c
from sympy import Eq

from devito.cgen_utils import ccode
from devito.dse import as_symbol, retrieve_terminals
from devito.interfaces import Indexed, Symbol
from devito.stencil import Stencil
from devito.tools import as_tuple, filter_ordered, flatten
from devito.arguments import ArgumentProvider, Argument

__all__ = ['Node', 'Block', 'Denormals', 'Expression', 'Function', 'FunCall',
           'Iteration', 'List', 'LocalExpression', 'TimedList']


class Node(object):

    is_Node = True
    is_Block = False
    is_Iteration = False
    is_IterationFold = False
    is_Expression = False
    is_Function = False
    is_FunCall = False
    is_List = False
    is_Element = False

    """
    :attr:`_traversable`. A list of traversable objects (ie, traversed by
    :class:`Visitor` objects). A traversable object is intended as an argument
    of a Node constructor and is represented as a string.
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

            .. code-block::
              from devito.visitors import CGen
              CGen().visit(self)
        """
        from devito.visitors import CGen
        return CGen().visit(self)

    @property
    def view(self):
        """
        Generate a representation of the Iteration/Expression tree rooted in ``self``.
        """
        from devito.visitors import printAST
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


class Block(Node):

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

    """Class representing a sequence of one or more statements."""

    is_List = True


class Element(Node):

    """A generic node that is worth identifying in an Iteration/Expression tree.

    It corresponds to a single :class:`cgen.Statement`.
    """

    is_Element = True

    def __init__(self, element):
        assert isinstance(element, (c.Comment, c.Statement, c.Value,
                                    c.Pragma, c.Line, c.Assign, c.POD))
        self.element = element

    def __repr__(self):
        return "Element::\n\t%s" % (self.element)


class FunCall(Node):

    """A node representing a function call."""

    is_FunCall = True

    def __init__(self, name, params=None):
        self.name = name
        self.params = as_tuple(params)

    def __repr__(self):
        return "FunCall::\n\t%s(...)" % self.name


class Expression(Node):

    """Class encpasulating a single SymPy equation."""

    is_Expression = True

    def __init__(self, expr, dtype=None):
        assert isinstance(expr, Eq)
        self.expr = expr
        self.dtype = dtype

        # Traverse /expression/ to determine meta information
        # Note: at this point, expressions have already been indexified
        self.functions = [i.base.function for i in retrieve_terminals(self.expr)
                          if isinstance(i, (Indexed, Symbol))]
        self.dimensions = flatten(i.indices for i in self.functions)
        # Filter collected dimensions and functions
        self.functions = filter_ordered(self.functions)
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
        return as_symbol(self.expr.lhs)

    @property
    def output_function(self):
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
    """Iteration object that encapsualtes a single loop over nodes.

    :param nodes: Single or list of :class:`Node` objects defining the loop body.
    :param dimension: :class:`Dimension` object over which to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    :param index: Symbol to be used as iteration variable.
    :param offsets: Optional map list of offsets to honour in the loop.
    :param properties: A bag of strings indicating properties of this Iteration.
                       For example, the string 'parallel' may be used to identify
                       a parallelizable Iteration.
    :param pragmas: A bag of pragmas attached to this Iteration.
    :param uindices: a bag of UnboundedIndex objects, representing free iteration
                     variables (i.e., the Iteration end point is independent of
                     any of these UnboundedIndex).
    """

    is_Iteration = True

    _traversable = ['nodes']

    """
    List of known properties, usable to decorate an Iteration: ::

        * sequential: An inherently sequential iteration space.
        * parallel: An iteration space whose iterations can safely be
                    executed in parallel.
        * vector-dim: A (SIMD) vectorizable iteration space.
        * elemental: Hoistable to an elemental function.
        * remainder: A remainder iteration (e.g., by-product of some transformations)
    """

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

        # Replace open limits with variables names
        if self.limits[1] is None:
            # FIXME: Add dimension size as variable bound.
            # Needs further generalisation to support loop blocking.
            dim = self.dim.parent if self.dim.is_Buffered else self.dim
            self.limits[1] = dim.size or dim.symbolic_size

        # Record offsets to later adjust loop limits accordingly
        self.offsets = [0, 0]
        for off in (offsets or {}):
            self.offsets[0] = min(self.offsets[0], int(off))
            self.offsets[1] = max(self.offsets[1], int(off))

        # Track this Iteration's properties, pragmas and unbounded indices
        self.properties = as_tuple(properties)
        assert (i in known_properties for i in self.properties)
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
        try:
            lower = int(self.limits[0]) - self.offsets[0]
        except (TypeError, ValueError):
            if isinstance(start, int):
                lower = start - self.offsets[0]
        try:
            upper = int(self.limits[1]) - self.offsets[1]
        except (TypeError, ValueError):
            if isinstance(finish, int):
                upper = finish - self.offsets[1]
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


class Function(Node):

    """Represent a C function.

    :param name: The name of the function.
    :param body: A :class:`Node` or an iterable of :class:`Node` objects representing
                 the body of the function.
    :param retval: The type of the value returned by the function.
    :param parameters: An iterable of :class:`SymbolicData` objects in input to the
                       function, or ``None`` if the function takes no parameter.
    :param prefix: An iterable of qualifiers to prepend to the function declaration.
                   The default value is ('static', 'inline').
    """

    is_Function = True

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
        parameters = ",".join([c.dtype_to_ctype(i.dtype) for i in self.parameters])
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
    Class encpasulating a single expression with known data type
    (represented as a NumPy data type).
    """

    def __init__(self, expr, dtype):
        super(LocalExpression, self).__init__(expr)
        self.dtype = dtype


# Iteration utilities

class IterationProperty(object):

    """
    An IterationProperty is an object that can be used to decorate an Iteration.
    """

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def __eq__(self, other):
        if not isinstance(other, IterationProperty):
            return False
        return self.name == other.name and self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.name if self.val is None else '%s%s' % (self.name, str(self.val))

    def __repr__(self):
        if self.val is None:
            return "Property: %s" % self.name
        else:
            return "Property: %s[%s]" % (self.name, str(self.val))


SEQUENTIAL = IterationProperty('sequential')
PARALLEL = IterationProperty('parallel')
VECTOR = IterationProperty('vector-dim')
ELEMENTAL = IterationProperty('elemental')
REMAINDER = IterationProperty('remainder')

known_properties = [SEQUENTIAL, PARALLEL, VECTOR, ELEMENTAL, REMAINDER]


def tagger(i):
    handle = IterationProperty('tag', i)
    if handle not in known_properties:
        known_properties.append(handle)
    return handle


def ntags():
    return len(known_properties) - ntags.n_original_properties
ntags.n_original_properties = len(known_properties)  # noqa


class UnboundedIndex(object):

    """
    A generic loop iteration index that can be used in a :class:`Iteration` to
    add a non-linear traversal of the iteration space.
    """

    def __init__(self, index, start=0, step=None):
        self.index = index
        self.start = start
        self.step = index + 1 if step is None else step

"""The Iteration/Expression hierarchy."""

from __future__ import absolute_import

import inspect
from collections import Iterable, OrderedDict

import cgen as c
from sympy import Eq, IndexedBase, preorder_traversal

from devito.codeprinter import ccode
from devito.dimension import Dimension
from devito.dse.inspection import retrieve_indexed
from devito.interfaces import SymbolicData, TensorData
from devito.tools import DefaultOrderedDict, as_tuple, filter_ordered

__all__ = ['Node', 'Block', 'Expression', 'Function', 'Iteration', 'List',
           'TimedList']


class Node(object):

    is_Node = True
    is_Block = False
    is_Iteration = False
    is_Expression = False
    is_Function = False
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
        """Generate C code."""
        raise NotImplementedError()

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

    _wrapper = c.Block

    _traversable = ['body']

    def __init__(self, header=None, body=None, footer=None):
        self.header = as_tuple(header)
        self.body = as_tuple(body)
        self.footer = as_tuple(footer)

    def __repr__(self):
        return "<%s (%d, %d, %d)>" % (self.__class__.__name__, len(self.header),
                                      len(self.body), len(self.footer))

    @property
    def ccode(self):
        body = tuple(s.ccode for s in self.body)
        return c.Module(self.header + (self._wrapper(body),) + self.footer)

    @property
    def children(self):
        return (self.body,)


class List(Block):

    """Class representing a sequence of one or more statements."""

    is_List = True

    _wrapper = c.Collection


class Element(Node):

    """A generic node that is worth identifying in an Iteration/Expression tree.

    It corresponds to a single :class:`cgen.Statement`.
    """

    is_Element = True

    def __init__(self, element):
        assert isinstance(element, (c.Comment, c.Statement, c.Value))
        self.element = element

    def __repr__(self):
        return "Element::\n\t%s" % (self.element)

    @property
    def ccode(self):
        return self.element


class Expression(Node):

    """Class encpasulating a single stencil expression"""

    is_Expression = True

    def __init__(self, stencil):
        assert isinstance(stencil, Eq)
        self.stencil = stencil

        self.dimensions = []
        self.functions = []
        # Traverse stencil to determine meta information
        for e in preorder_traversal(self.stencil):
            if isinstance(e, SymbolicData):
                self.dimensions += list(e.indices)
                self.functions += [e]
            if isinstance(e, IndexedBase):
                self.dimensions += list(e.function.indices)
                self.functions += [e.function]
        # Filter collected dimensions and functions
        self.dimensions = filter_ordered(self.dimensions)
        self.functions = filter_ordered(self.functions)

    def __repr__(self):
        return "<Expression::%s>" % filter_ordered([f.func for f in self.functions])

    def substitute(self, substitutions):
        """Apply substitutions to the expression stencil

        :param substitutions: Dict containing the substitutions to apply to
                              the stored loop stencils.
        """
        self.stencil = self.stencil.xreplace(substitutions)

    @property
    def ccode(self):
        return c.Assign(ccode(self.stencil.lhs), ccode(self.stencil.rhs))

    @property
    def output(self):
        """
        Return the symbol written by this Expression.
        """
        lhs = self.stencil.lhs
        if lhs.is_Symbol:
            return lhs
        else:
            # An Indexed
            return lhs.base.label

    @property
    def index_offsets(self):
        """Mapping of :class:`Dimension` symbols to each integer
        stencil offset used with it in this expression.

        This also include zero offsets, so that the keys of this map
        may be used as the set of :class:`Dimension` symbols used in
        this expression.

        Note: This assumes we have indexified the stencil expression.
        """
        offsets = DefaultOrderedDict(set)

        # Enforce left-first traversal order
        indexed = list(retrieve_indexed(self.stencil.lhs))
        indexed += list(retrieve_indexed(self.stencil.rhs))
        for e in indexed:
            for a in e.indices:
                if isinstance(a, Dimension):
                    offsets[a].update([0])
                d = None
                off = []
                for idx in a.args:
                    if isinstance(idx, Dimension):
                        d = idx
                    elif idx.is_integer:
                        off += [idx]
                if d is not None:
                    offsets[d].update(off)
        return offsets


class TypedExpression(Expression):

    """Class encpasulating a single stencil expression with known data type,
    represented as a NumPy data type."""

    def __init__(self, stencil, dtype):
        super(TypedExpression, self).__init__(stencil)
        self.dtype = dtype

    @property
    def ccode(self):
        ctype = c.dtype_to_ctype(self.dtype)
        return c.Initializer(c.Value(ctype, ccode(self.stencil.lhs)),
                             ccode(self.stencil.rhs))


class Iteration(Node):
    """Iteration object that encapsualtes a single loop over nodes, possibly
    just SymPy expressions.

    :param nodes: Single or list of :class:`Node` objects that
                        define the loop body.
    :param dimension: :class:`Dimension` object over which to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    :param index: Symbol to be used as iteration variable.
    :param offsets: Optional map list of offsets to honour in the loop.
    :param properties: A bag of strings indicating properties of this Iteration.
                       For example, the string 'parallel' may be used to identify
                       a parallelizable Iteration.
    """

    is_Iteration = True

    _traversable = ['nodes']

    def __init__(self, nodes, dimension, limits, index=None, offsets=None,
                 properties=None):
        # Ensure we deal with a list of Expression objects internally
        nodes = as_tuple(nodes)
        self.nodes = as_tuple([n if isinstance(n, Node) else Expression(n)
                               for n in nodes])
        assert all(isinstance(i, Node) for i in self.nodes)

        self.dim = dimension
        self.index = index or self.dim.name

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
            self.limits[1] = IterationBound("%s_size" % self.dim.name, self.dim)

        # Record offsets to later adjust loop limits accordingly
        self.offsets = [0, 0]
        for off in (offsets or {}):
            self.offsets[0] = min(self.offsets[0], int(off))
            self.offsets[1] = max(self.offsets[1], int(off))

        # Track this Iteration's properties
        self.properties = as_tuple(properties)

    def __repr__(self):
        properties = ""
        if self.properties:
            properties = "WithProperties[%s]::" % ",".join(self.properties)
        return "<%sIteration %s; %s>" % (properties, self.index, self.limits)

    @property
    def ccode(self):
        """Generate C code for the represented stencil loop

        :returns: :class:`cgen.For` object representing the loop
        """
        loop_body = [s.ccode for s in self.nodes]

        # Start
        if self.offsets[0] != 0:
            val = "%s + %s" % (self.limits[0], -self.offsets[0])
            try:
                val = eval(val)
            except (NameError, TypeError):
                pass
        else:
            val = self.limits[0]
        loop_init = c.InlineInitializer(c.Value("int", self.index), ccode(val))

        # Bound
        if self.offsets[1] != 0:
            val = "%s - %s" % (self.limits[1], self.offsets[1])
            try:
                val = eval(val)
            except (NameError, TypeError):
                pass
        else:
            val = self.limits[1]
        loop_cond = '%s < %s' % (self.index, ccode(val))

        # Increment
        loop_inc = '%s += %s' % (self.index, self.limits[2])

        return c.For(loop_init, loop_cond, loop_inc, c.Block(loop_body))

    @property
    def is_Open(self):
        return self.dim.size is not None

    @property
    def is_Closed(self):
        return not self.is_Open

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
        self.parameters = as_tuple(parameters)
        self.prefix = prefix

    def __repr__(self):
        parameters = ",".join([c.dtype_to_ctype(i.dtype) for i in self.parameters])
        body = "\n\t".join([str(s) for s in self.body])
        return "Function[%s]<%s; %s>::\n\t%s" % (self.name, self.retval, parameters, body)

    @property
    def _cparameters(self):
        """Generate arguments signature."""
        cparameters = []
        for v in self.parameters:
            if isinstance(v, Dimension):
                cparameters.append(v.decl)
            elif v.is_ScalarData:
                cparameters.append(c.Value('const int', v.name))
            else:
                cparameters.append(c.Pointer(c.POD(v.dtype, '%s_vec' % v.name)))
        return cparameters

    @property
    def _ccasts(self):
        """Generate data casts."""
        handle = [f for f in self.parameters if isinstance(f, TensorData)]
        shapes = [(f, ''.join(["[%s]" % i.ccode for i in f.indices[1:]])) for f in handle]
        casts = [c.Initializer(c.POD(v.dtype, '(*%s)%s' % (v.name, shape)),
                               '(%s (*)%s) %s' % (c.dtype_to_ctype(v.dtype),
                                                  shape, '%s_vec' % v.name))
                 for v, shape in shapes]
        return casts

    @property
    def _ctop(self):
        """Generate the function declaration."""
        return c.FunctionDeclaration(c.Value(self.retval, self.name), self._cparameters)

    @property
    def ccode(self):
        """Generate C code for the represented C routine.

        :returns: :class:`cgen.FunctionDeclaration` object representing the function.
        """
        body = [e.ccode for e in self.body]
        return c.FunctionBody(self._ctop, c.Block(self._ccasts + body))

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


class Denormals(Block):

    """Macros to make sure denormal numbers are flushed in hardware."""

    def __init__(self, header=None, body=None, footer=None):
        b = [Element(c.Comment('Flush denormal numbers to zero in hardware')),
             Element(c.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)')),
             Element(c.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'))]
        super(Denormals, self).__init__(header, b, footer)

    def __repr__(self):
        return "<DenormalsMacro>"


class IterationBound(object):
    """Utility class to encapsulate variable loop bounds and link them
    back to the respective :class:`Dimension` object.

    :param name: Variable name for the open loop bound variable.
    :param dim: The corresponding :class:`Dimension` object.
    :param expr: An expression to calculate the loop limit, in case this does
                 not coincide with the open loop bound variable itself.
    """

    def __init__(self, name, dim, expr=None):
        self.name = name
        self.dim = dim
        self.expr = expr

    def __repr__(self):
        return repr(self.expr).replace(' ', '') if self.expr else self.name

    def __eq__(self, bound):
        return self.name == bound.name and self.dim == bound.dim

    @property
    def ccode(self):
        """C code for the variable declaration within a kernel signature"""
        return c.Value('const int', self.name)

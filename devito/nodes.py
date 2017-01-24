"""The Iteration/Expression hierarchy."""

from __future__ import absolute_import

import inspect
from collections import Iterable, OrderedDict, defaultdict
from itertools import chain
from operator import attrgetter

import cgen as c
from sympy import Eq, IndexedBase, Symbol, preorder_traversal

from devito.codeprinter import ccode
from devito.dimension import Dimension
from devito.dse.inspection import terminals
from devito.interfaces import SymbolicData
from devito.logger import warning
from devito.tools import as_tuple, filter_ordered, flatten

__all__ = ['Node', 'Block', 'Expression', 'Iteration', 'Timer']


class Node(object):

    is_Node = True
    is_Block = False
    is_Iteration = False
    is_Expression = False

    """
    :attr:`_traversable`. A list of traversable objects (ie, traversed by
    :class:`Visitor` objects). A traversable object is intended as an argument
    of a Node constructor and is represented as a string.
    """
    _traversable = []

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames = inspect.getargspec(cls.__init__).args
        obj._args = OrderedDict(list(zip(argnames[1:], args)) + list(kwargs.items()))
        return obj

    def _children(self):
        """Return the traversable children."""
        return []

    def _rebuild(self, *args, **kwargs):
        """Reconstruct self. None of the embedded Sympy expressions are rebuilt."""
        handle = [i for i in self._traversable if i not in kwargs]
        handle = OrderedDict([(k, v) for k, v in zip(handle, args)])
        return type(self)(**dict(handle.items() + kwargs.items()))

    def indexify(self):
        """Convert all enclosed nodes to "indexed" format"""
        for e in self._children():
            e.indexify()

    def substitute(self, substitutions):
        """Apply substitutions to children nodes

        :param substitutions: Dict containing the substitutions to apply.
        """
        candidates = [n for n in self._children() if isinstance(n, Node)]
        for n in candidates:
            n.substitute(substitutions)

    @property
    def index_offsets(self):
        """Collect all non-zero offsets used with each index in a map

        Note: This assumes we have indexified the stencil expression."""
        return defaultdict(list)

    @property
    def ccode(self):
        """Generate C code."""
        raise NotImplementedError()

    @property
    def signature(self):
        """List of data objects used by the Node."""
        return list(flatten([e.signature for e in self._children()]))

    @property
    def args(self):
        """Arguments used to construct the Node."""
        return self._args

    @property
    def args_frozen(self):
        """Arguments used to construct the Node that cannot be traversed."""
        return OrderedDict([(k, v) for k, v in self.args.items()
                            if k not in self._traversable])


class Block(Node):

    is_Block = True

    _traversable = ['body']

    def __init__(self, header=None, body=None, footer=None):
        self.header = as_tuple(header)
        self.body = as_tuple(body)
        self.footer = as_tuple(footer)

    def __repr__(self):
        header = "".join([str(s) for s in self.header])
        body = "Block::\n\t%s" % "\n\t".join([str(s) for s in self.body])
        footer = "".join([str(s) for s in self.footer])
        return header + body + footer

    @property
    def ccode(self):
        body = tuple(s.ccode for s in self.body)
        return c.Block(self.header + body + self.footer)

    def _children(self):
        return self.body


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
        return "Expression<%s = %s>" % (self.stencil.lhs, self.stencil.rhs)

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
    def signature(self):
        """List of data objects used by the expression

        :returns: List of unique data objects required by the expression
        """
        return filter_ordered([f for f in self.functions],
                              key=attrgetter('name'))

    def indexify(self):
        """Convert stencil expression to "indexed" format"""
        self.stencil = indexify(self.stencil)

    @property
    def index_offsets(self):
        """Collect all non-zero offsets used with each index in a map

        Note: This assumes we have indexified the stencil expression."""
        offsets = defaultdict(list)
        for e in terminals(self.stencil):
            for a in e.indices:
                d = None
                off = []
                for idx in a.args:
                    if isinstance(idx, Dimension):
                        d = idx
                    else:
                        off += [idx]
                if d is not None:
                    offsets[d] += off
        return offsets


class Iteration(Node):
    """Iteration object that encapsualtes a single loop over nodes, possibly
    just SymPy expressions.

    :param nodes: Single or list of :class:`Node` objects that
                        define the loop body.
    :param dimension: :class:`Dimension` object over which to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    :param offsets: Optional map list of offsets to honour in the loop
    """

    is_Iteration = True

    _traversable = ['nodes']

    def __init__(self, nodes, dimension, limits, offsets=None):
        # Ensure we deal with a list of Expression objects internally
        self.nodes = as_tuple(nodes)
        self.nodes = [n if isinstance(n, Node) else Expression(n)
                      for n in self.nodes]
        assert all(isinstance(i, Node) for i in self.nodes)

        # Generate index variable name and variable substitutions
        self.dim = dimension
        if isinstance(self.dim, Dimension):
            if self.dim.buffered is None:
                # Generate index variable from dimension
                self.index = str(self.dim.get_varname())
                self.subs = {self.dim: Symbol(self.index)}
            else:
                # Generate numbered indices for each buffer
                self.index = self.dim.name
                self.subs = {self.dim: Symbol(self.dim.get_varname())}
                for offset in self.index_offsets[self.dim]:
                    self.subs[self.dim + offset] = Symbol(self.dim.get_varname())
        else:
            warning("Generating Iteration without Dimension object")
            self.index = str(dimension)

        # Propagate variable names to the lower expressions
        self.substitute(self.subs)

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

    def __repr__(self):
        str_expr = "\n\t".join([str(s) for s in self.nodes])
        return "Iteration<%s; %s>::\n\t%s" % (self.index, self.limits,
                                              str_expr)

    @property
    def ccode(self):
        """Generate C code for the represented stencil loop

        :returns: :class:`cgen.For` object representing the loop
        """
        loop_body = [s.ccode for s in self.nodes]
        if self.dim.buffered is not None:
            modulo = self.dim.buffered
            v_subs = [c.Initializer(c.Value('int', v), "(%s) %% %d" % (s, modulo))
                      for s, v in self.subs.items()]
            loop_body = v_subs + loop_body
        loop_init = c.InlineInitializer(
            c.Value("int", self.index), "%d + %d" % (self.limits[0], -self.offsets[0]))
        loop_cond = '%s %s %s' % (self.index, '<' if self.limits[2] >= 0 else '>',
                                  "%s - %d" % (self.limits[1], self.offsets[1]))
        loop_inc = '%s += %s' % (self.index, self.limits[2])
        return c.For(loop_init, loop_cond, loop_inc, c.Block(loop_body))

    @property
    def signature(self):
        """List of data objects used by the loop and it's body

        :returns: List of unique data objects required by the loop
        """
        signature = [e.signature for e in self.nodes]
        signature = filter_ordered(chain(*signature))
        if isinstance(self.limits[1], IterationBound):
            signature += [self.dim]
        return signature

    @property
    def index_offsets(self):
        """Collect all non-zero offsets used with each index in a map

        Note: This assumes we have indexified the stencil expression."""
        offsets = defaultdict(list)
        for n in self.nodes:
            offsets.update(n.index_offsets)
        return offsets

    def _children(self):
        """Return the traversable children."""
        return self.nodes


# Utilities

class Timer(Block):

    """Wrap a Node with C-level timers."""

    def __init__(self, lname, gname, body):
        """
        Initialize a Timer object.

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
        super(Timer, self).__init__(header, body, footer)

    @property
    def name(self):
        return self._name


class IterationBound(object):
    """Utility class to encapsulate variable loop bounds and link them
    back to the respective Dimension object.

    :param name: Variable name for the open loop bound variable
    """

    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def __repr__(self):
        return self.name

    @property
    def ccode(self):
        """C code for the variable declaration within a kernel signature"""
        return c.Value('const int', self.name)

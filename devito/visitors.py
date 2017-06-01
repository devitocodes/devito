"""
Visitor hierarchy to inspect and/or create Expression/Iteration trees.

The main Visitor class is extracted from https://github.com/coneoproject/COFFEE.
"""

from __future__ import absolute_import

import inspect
from collections import Iterable, OrderedDict, defaultdict
from operator import attrgetter

import cgen as c
from sympy import Symbol

from devito.dimension import LoweredDimension
from devito.exceptions import VisitorException
from devito.nodes import Iteration, List, Node
from devito.tools import as_tuple, filter_ordered, filter_sorted, flatten


__all__ = ['FindNodes', 'FindSections', 'FindSymbols', 'FindScopes',
           'IsPerfectIteration', 'SubstituteExpression', 'printAST',
           'ResolveIterationVariable', 'Transformer', 'NestedTransformer',
           'FindAdjacentIterations']


class Visitor(object):

    """
    A generic visitor for an Expression/Iteration tree.

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.
    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.

    The method signature is:

        .. code-block::
           def visit_Foo(self, o, [*args, **kwargs]):
               pass

    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:

        .. code-block::
           def visit_Foo(self, o, parent=None, *args, **kwargs):
               pass
    """

    def __init__(self):
        handlers = {}
        # visit methods are spelt visit_Foo.
        prefix = "visit_"
        # Inspect the methods on this instance to find out which
        # handlers are defined.
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            # Check the argument specification
            # Valid options are:
            #    visit_Foo(self, o, [*args, **kwargs])
            argspec = inspect.getargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    """
    :attr:`default_args`. A dict of default keyword arguments for the visitor.
    These are not used by default in :meth:`visit`, however, a caller may pass
    them explicitly to :meth:`visit` by accessing :attr:`default_args`.
    For example::

        .. code-block::
           v = FooVisitor()
           v.visit(node, **v.default_args)
    """
    default_args = {}

    @classmethod
    def default_retval(cls):
        """A method that returns an object to use to populate return values.

        If your visitor combines values in a tree-walk, it may be useful to
        provide a object to combine the results into. :meth:`default_retval`
        may be defined by the visitor to be called to provide an empty object
        of appropriate type.
        """
        return None

    def lookup_method(self, instance):
        """Look up a handler method for a visitee.

        :param instance: The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            # Do we have a method handler defined for this type name
            return self._handlers[cls.__name__]
        except KeyError:
            # No, walk the MRO.
            for klass in cls.mro()[1:]:
                entry = self._handlers.get(klass.__name__)
                if entry:
                    # Save it on this type name for faster lookup next time
                    self._handlers[cls.__name__] = entry
                    return entry
        raise RuntimeError("No handler found for class %s", cls.__name__)

    def visit(self, o, *args, **kwargs):
        """Apply this :class:`Visitor` to an AST.

            :param o: The :class:`Node` to visit.
            :param args: Optional arguments to pass to the visit methods.
            :param kwargs: Optional keyword arguments to pass to the visit methods.
        """
        meth = self.lookup_method(o)
        return meth(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):
        return self.default_retval()

    def visit_Node(self, o, **kwargs):
        return self.visit(o.children, **kwargs)

    def reuse(self, o, *args, **kwargs):
        """A visit method to reuse a node, ignoring children."""
        return o

    def maybe_rebuild(self, o, *args, **kwargs):
        """A visit method that rebuilds nodes if their children have changed."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        if all(a is b for a, b in zip(ops, new_ops)):
            return o
        return o._rebuild(*new_ops, **okwargs)

    def always_rebuild(self, o, *args, **kwargs):
        """A visit method that always rebuilds nodes."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        return o._rebuild(*new_ops, **okwargs)


class PrintAST(Visitor):

    _depth = 0

    def __init__(self, verbose=True):
        super(PrintAST, self).__init__()
        self.verbose = verbose

    @classmethod
    def default_retval(cls):
        return "<>"

    @property
    def indent(self):
        return '  ' * self._depth

    def visit_Node(self, o):
        return self.indent + '<%s>' % o.__class__.__name__

    def visit_Generable(self, o):
        body = ' %s' % str(o) if self.verbose else ''
        return self.indent + '<C.%s%s>' % (o.__class__.__name__, body)

    def visit_Element(self, o):
        body = ' %s' % str(o.element) if self.verbose else ''
        return self.indent + '<Element%s>' % body

    def visit_Function(self, o):
        self._depth += 1
        body = self.visit(o.children)
        self._depth -= 1
        return self.indent + '<Function %s>\n%s' % (o.name, body)

    def visit_list(self, o):
        return ('\n').join([self.visit(i) for i in o])

    def visit_tuple(self, o):
        return '\n'.join([self.visit(i) for i in o])

    def visit_Block(self, o):
        self._depth += 1
        if self.verbose:
            body = [self.visit(o.header), self.visit(o.body), self.visit(o.footer)]
        else:
            body = [self.visit(o.body)]
        self._depth -= 1
        return self.indent + "<%s>\n%s" % (o.__class__.__name__, '\n'.join(body))

    def visit_Iteration(self, o):
        self._depth += 1
        body = self.visit(o.children)
        self._depth -= 1
        if self.verbose:
            detail = '::%s::%s::%s' % (o.index, o.limits, o.offsets)
            props = '[%s] ' % ','.join(o.properties) if o.properties else ''
        else:
            detail, props = '', ''
        return self.indent + "<%sIteration %s%s>\n%s" % (props, o.dim.name, detail, body)

    def visit_Expression(self, o):
        if self.verbose:
            body = "%s = %s" % (o.expr.lhs, o.expr.rhs)
            return self.indent + "<Expression %s>" % body
        else:
            return self.indent + str(o)


class FindSections(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict()

    """Find all sections in an Iteration/Expression tree. A section is a map
    from an iteration space (ie, a sequence of :class:`Iteration` obects) to
    a set of expressions (ie, the :class:`Expression` objects enclosed by the
    iteration space).
    """

    def visit_tuple(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        for i in o:
            ret = self.visit(i, ret=ret, queue=queue)
        return ret

    def visit_Node(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        for i in o.children:
            ret = self.visit(i, ret=ret, queue=queue)
        return ret

    def visit_Iteration(self, o, ret=None, queue=None):
        if queue is None:
            queue = [o]
        else:
            queue.append(o)
        for i in o.children:
            ret = self.visit(i, ret=ret, queue=queue)
        queue.remove(o)
        return ret

    def visit_Expression(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        key = tuple(queue) if queue is not None else ()
        ret.setdefault(key, []).append(o)
        return ret

    visit_Element = visit_Expression
    visit_FunCall = visit_Expression


class FindScopes(FindSections):

    @classmethod
    def default_retval(cls):
        return OrderedDict()

    """
    Map each written variable or :class:`FunCall` object in the Iteration/Expression
    tree to its section.
    """

    def visit_FunCall(self, o, ret=None, queue=None, in_omp_region=False):
        if ret is None:
            ret = self.default_retval()
        ret[o] = as_tuple(queue)
        return ret

    visit_Expression = visit_FunCall
    visit_Element = FindSections.visit_Node


class FindSymbols(Visitor):

    @classmethod
    def default_retval(cls):
        return []

    """Find symbols in an Iteration/Expression tree.

    :param mode: Drive the search. Accepted values are: ::

        * 'kernel-data' (default): Collect :class:`SymbolicData` objects.
        * 'symbolics': Collect :class:`AbstractSymbol` objects.
        * 'symbolics-writes': Collect written :class:`AbstractSymbol` objects.
        * 'free-symbols': Collect all free symbols.
        * 'dimensions': Collect :class:`Dimension` objects only.
    """

    rules = {
        'kernel-data': lambda e: [i for i in e.functions if i.is_SymbolicData],
        'symbolics': lambda e: e.functions,
        'symbolics-writes': lambda e: as_tuple(e.output_function),
        'free-symbols': lambda e: e.expr.free_symbols,
        'dimensions': lambda e: e.dimensions,
    }

    def __init__(self, mode='kernel-data'):
        super(FindSymbols, self).__init__()
        self.rule = self.rules[mode]

    def visit_tuple(self, o):
        symbols = flatten([self.visit(i) for i in o])
        return filter_sorted(symbols, key=attrgetter('name'))

    def visit_Iteration(self, o):
        symbols = flatten([self.visit(i) for i in o.children])
        return filter_sorted(symbols, key=attrgetter('name'))

    def visit_Expression(self, o):
        return filter_sorted([f for f in self.rule(o)], key=attrgetter('name'))


class FindNodes(Visitor):

    @classmethod
    def default_retval(cls):
        return []

    """
    Find :class:`Node` instances.

    :param match: Pattern to look for.
    :param mode: Drive the search. Accepted values are: ::

        * 'type' (default): Collect all instances of type ``match``.
        * 'scope': Return the scope in which the object ``match`` appears.
    """

    rules = {
        'type': lambda match, o: isinstance(o, match),
        'scope': lambda match, o: match in flatten(o.children)
    }

    def __init__(self, match, mode='type'):
        super(FindNodes, self).__init__()
        self.match = match
        self.rule = self.rules[mode]

    def visit_object(self, o, ret=None):
        return ret

    def visit_tuple(self, o, ret=None):
        for i in o:
            ret = self.visit(i, ret=ret)
        return ret

    def visit_Node(self, o, ret=None):
        if ret is None:
            ret = self.default_retval()
        if self.rule(self.match, o):
            ret.append(o)
        for i in o.children:
            ret = self.visit(i, ret=ret)
        return ret


class FindAdjacentIterations(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict([('seen_iteration', False)])

    """
    Return a mapper from nodes N in an Expression/Iteration tree to sequences of
    :class:`Iteration` objects I = [I_0, I_1, ...], where N is the direct ancestor of
    the items in I and all items in I are adjacent nodes in the tree.
    """

    def handler(self, o, parent=None, ret=None):
        if ret is None:
            ret = self.default_retval()
        if parent is None:
            return ret
        group = []
        for i in o:
            ret = self.visit(i, parent=parent, ret=ret)
            if ret['seen_iteration'] is True:
                group.append(i)
            else:
                if len(group) > 1:
                    ret.setdefault(parent, []).append(tuple(group))
                # Reset the group, Iterations no longer adjacent
                group = []
        # Potential leftover
        if len(group) > 1:
            ret.setdefault(parent, []).append(tuple(group))
        return ret

    def visit_object(self, o, parent=None, ret=None):
        return ret

    def visit_tuple(self, o, parent=None, ret=None):
        return self.handler(o, parent=parent, ret=ret)

    def visit_Node(self, o, parent=None, ret=None):
        ret = self.handler(o.children, parent=o, ret=ret)
        ret['seen_iteration'] = False
        return ret

    def visit_Iteration(self, o, parent=None, ret=None):
        ret = self.handler(o.children, parent=o, ret=ret)
        ret['seen_iteration'] = True
        return ret


class IsPerfectIteration(Visitor):

    """
    Return True if an :class:`Iteration` defines a perfect loop nest, False otherwise.
    """

    def visit_object(self, o, **kwargs):
        return False

    def visit_tuple(self, o, **kwargs):
        return all(self.visit(i, **kwargs) for i in o)

    def visit_Node(self, o, found=False, **kwargs):
        # Assume all nodes are in a perfect loop if they're in a loop.
        return found

    def visit_Iteration(self, o, found=False, multi=False):
        if found and multi:
            return False
        multi = len(o.nodes) > 1
        return all(self.visit(i, found=True, multi=multi) for i in o.children)


class Transformer(Visitor):

    """
    Given an Iteration/Expression tree T and a mapper from nodes in T to
    a set of new nodes L, M : N --> L, build a new Iteration/Expression tree T'
    where a node ``n`` in N is replaced with ``M[n]``.

    In the special case in which ``M[n]`` is None, ``n`` is dropped from T'.

    In the special case in which ``M[n]`` is an iterable of nodes, ``n`` is
    "extended" by pre-pending to its body the nodes in ``M[n]``.
    """

    def __init__(self, mapper={}):
        super(Transformer, self).__init__()
        self.mapper = mapper.copy()
        self.rebuilt = {}

    def visit_object(self, o, **kwargs):
        return o

    def visit_tuple(self, o, **kwargs):
        visited = tuple(self.visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            handle = self.mapper[o]
            if handle is None:
                # None -> drop /o/
                return None
            elif isinstance(handle, Iterable):
                if not o.children:
                    raise VisitorException
                extended = (tuple(handle) + o.children[0],) + o.children[1:]
                return o._rebuild(*extended, **o.args_frozen)
            else:
                return handle._rebuild(**handle.args)
        else:
            rebuilt = [self.visit(i, **kwargs) for i in o.children]
            return o._rebuild(*rebuilt, **o.args_frozen)

    def visit(self, o, *args, **kwargs):
        obj = super(Transformer, self).visit(o, *args, **kwargs)
        if isinstance(o, Node) and obj is not o:
            self.rebuilt[o] = obj
        return obj


class NestedTransformer(Transformer):

    """
    Unlike a :class:`Transformer`, a :class:`NestedTransforer` applies
    replacements in a depth-first fashion.
    """

    def visit_Node(self, o, **kwargs):
        rebuilt = [self.visit(i, **kwargs) for i in o.children]
        handle = self.mapper.get(o, o)
        if handle is None:
            # None -> drop /o/
            return None
        elif isinstance(handle, Iterable):
            if not o.children:
                raise VisitorException
            extended = [tuple(handle) + rebuilt[0]] + rebuilt[1:]
            return o._rebuild(*extended, **o.args_frozen)
        else:
            return handle._rebuild(*rebuilt, **handle.args_frozen)


class SubstituteExpression(Transformer):
    """
    :class:`Transformer` that performs symbol substitution on
    :class:`Expression` objects in a given tree.

    :param subs: Dict defining the symbol substitution
    """

    def __init__(self, subs={}):
        super(SubstituteExpression, self).__init__()
        self.subs = subs

    def visit_Expression(self, o):
        o.substitute(self.subs)
        return o._rebuild(expr=o.expr)


class ResolveIterationVariable(Transformer):
    """
    :class:`Transformer` class that creates a substitution dictionary
    for replacing :class:`Dimension` instances with explicit loop
    variables in :class:`Iteration` nodes. For buffered dimensions it
    also inserts the relevant definitions for buffer index variables,
    for exaple.:

        .. code-block:: c

           for (int t = 0; t < t_size; t += 1)
           {
               int t0 = (t) % 2;
               int t1 = (t + 1) % 2;
    """

    def visit_Iteration(self, o, subs={}, offsets=defaultdict(set)):
        nodes = self.visit(o.children, subs=subs, offsets=offsets)
        if o.dim.is_Buffered:
            # For buffered dimensions insert the explicit
            # definition of buffered variables, eg. t+1 => t1
            init = []
            for i, off in enumerate(filter_ordered(offsets[o.dim])):
                vname = "%s%d" % (o.dim.name, i)
                value = o.dim.parent + off
                modulo = o.dim.modulo
                init += [c.Initializer(c.Value('int', vname),
                                       "(%s) %% %d" % (value, modulo))]
                subs[o.dim + off] = LoweredDimension(vname, o.dim, off)
            # Always lower to symbol
            subs[o.dim.parent] = Symbol(o.dim.parent.name)
            # Insert block with modulo initialisations
            newnodes = (List(header=init, body=nodes[0]), )
            return o._rebuild(newnodes, index=o.dim.parent.name)
        else:
            return o._rebuild(*nodes)

    def visit_Expression(self, o, subs={}, offsets=defaultdict(set)):
        """Collect all offsets used with a dimension"""
        for dim, offs in o.stencil.entries:
            offsets[dim].update(offs)
        return o


class MergeOuterIterations(Transformer):
    """
    :class:`Transformer` that merges subsequent :class:`Iteration`
    objects iff their dimenions agree.
    """

    def is_mergable(self, iter1, iter2):
        """Defines if two :class:`Iteration` objects are mergeable.

        Note: This currently does not(!) consider data dependencies
        between the loops. A deeper analysis is required for this that
        will be added soon.
        """
        if iter1.dim.is_Buffered:
            # Aliasing only works one-way because we left-merge
            if iter1.dim.parent == iter2.dim:
                return True
            if iter2.dim.is_Buffered and iter1.dim.parent == iter2.dim.parent:
                return True
        return iter1.dim == iter2.dim and iter1.bounds_symbolic == iter2.bounds_symbolic

    def merge(self, iter1, iter2):
        """Creates a new merged :class:`Iteration` object from two
        loops along the same dimension.
        """
        newexpr = iter1.nodes + iter2.nodes
        return Iteration(newexpr, dimension=iter1.dim,
                         limits=iter1.limits,
                         offsets=iter1.offsets)

    def visit_Iteration(self, o):
        rebuilt = self.visit(o.children)
        ret = o._rebuild(*rebuilt, **o.args_frozen)
        return ret

    def visit_list(self, o):
        head = self.visit(o[0])
        if len(o) < 2:
            return tuple([head])
        body = self.visit(o[1:])
        if head.is_Iteration and body[0].is_Iteration:
            if self.is_mergable(head, body[0]):
                newit = self.merge(head, body[0])
                ret = self.visit([newit] + list(body[1:]))
                return as_tuple(ret)
        return tuple([head] + list(body))

    visit_tuple = visit_list


def printAST(node, verbose=True):
    return PrintAST(verbose=verbose).visit(node)

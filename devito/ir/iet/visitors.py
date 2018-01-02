"""
Visitor hierarchy to inspect and/or create IETs.

The main Visitor class is adapted from https://github.com/coneoproject/COFFEE.
"""

from __future__ import absolute_import

from collections import Iterable, OrderedDict, defaultdict
from operator import attrgetter

import cgen as c
import numpy as np

from devito.cgen_utils import blankline, ccode
from devito.dimension import LoweredDimension
from devito.exceptions import VisitorException
from devito.ir.iet.nodes import Node, UnboundedIndex
from devito.types import Scalar
from devito.tools import (as_tuple, filter_ordered, filter_sorted, flatten, ctypes_to_C,
                          GenericVisitor)
from devito.arguments import runtime_arguments


__all__ = ['FindNodes', 'FindSections', 'FindSymbols', 'MapExpressions',
           'IsPerfectIteration', 'SubstituteExpression', 'printAST', 'CGen',
           'ResolveTimeStepping', 'Transformer', 'NestedTransformer',
           'FindAdjacentIterations', 'MapIteration']


class Visitor(GenericVisitor):

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

    """
    Return a representation of the Iteration/Expression tree as a string,
    highlighting tree structure and node properties while dropping non-essential
    information.
    """

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

    def visit_Callable(self, o):
        self._depth += 1
        body = self.visit(o.children)
        self._depth -= 1
        return self.indent + '<Callable %s>\n%s' % (o.name, body)

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
            props = [str(i) for i in o.properties]
            props = '[%s] ' % ','.join(props) if props else ''
        else:
            detail, props = '', ''
        return self.indent + "<%sIteration %s%s>\n%s" % (props, o.dim.name, detail, body)

    def visit_Expression(self, o):
        if self.verbose:
            body = "%s = %s" % (o.expr.lhs, o.expr.rhs)
            return self.indent + "<Expression %s>" % body
        else:
            return self.indent + str(o)


class CGen(Visitor):

    """
    Return a representation of the Iteration/Expression tree as a :module:`cgen` tree.
    """

    def _args_decl(self, args):
        """Convert an iterable of :class:`Argument` into cgen format."""
        ret = []
        for i in args:
            if i.is_ScalarArgument:
                ret.append(c.Value('const %s' % c.dtype_to_ctype(i.dtype), i.name))
            elif i.is_TensorArgument:
                ret.append(c.Value(c.dtype_to_ctype(i.dtype),
                                   '*restrict %s_vec' % i.name))
            else:
                ret.append(c.Value('void', '*_%s' % i.name))
        return ret

    def _args_cast(self, args):
        """Build cgen type casts for an iterable of :class:`Argument`."""
        ret = []
        for i in args:
            if i.is_TensorArgument:
                align = "__attribute__((aligned(64)))"
                shape = ''.join(["[%s]" % ccode(j)
                                 for j in i.provider.symbolic_shape[1:]])
                lvalue = c.POD(i.dtype, '(*restrict %s)%s %s' % (i.name, shape, align))
                rvalue = '(%s (*)%s) %s' % (c.dtype_to_ctype(i.dtype), shape,
                                            '%s_vec' % i.name)
                ret.append(c.Initializer(lvalue, rvalue))
            elif i.is_PtrArgument:
                ctype = ctypes_to_C(i.dtype)
                lvalue = c.Pointer(c.Value(ctype, i.name))
                rvalue = '(%s*) %s' % (ctype, '_%s' % i.name)
                ret.append(c.Initializer(lvalue, rvalue))
        return ret

    def visit_tuple(self, o):
        return tuple(self.visit(i) for i in o)

    def visit_Block(self, o):
        body = flatten(self.visit(i) for i in o.children)
        return c.Module(o.header + (c.Block(body),) + o.footer)

    def visit_List(self, o):
        body = flatten(self.visit(i) for i in o.children)
        return c.Module(o.header + (c.Collection(body),) + o.footer)

    def visit_Element(self, o):
        return o.element

    def visit_Expression(self, o):
        return c.Assign(ccode(o.expr.lhs), ccode(o.expr.rhs))

    def visit_LocalExpression(self, o):
        return c.Initializer(c.Value(c.dtype_to_ctype(o.dtype),
                             ccode(o.expr.lhs)), ccode(o.expr.rhs))

    def visit_Call(self, o):
        return c.Statement('%s(%s)' % (o.name, ','.join(o.params)))

    def visit_Iteration(self, o):
        body = flatten(self.visit(i) for i in o.children)

        # Start
        if o.offsets[0] != 0:
            start = str(o.limits[0] + o.offsets[0])
            try:
                start = eval(start)
            except (NameError, TypeError):
                pass
        else:
            start = o.limits[0]

        # Bound
        if o.offsets[1] != 0:
            end = str(o.limits[1] + o.offsets[1])
            try:
                end = eval(end)
            except (NameError, TypeError):
                pass
        else:
            end = o.limits[1]

        # For reverse dimensions flip loop bounds
        if o.reverse:
            loop_init = 'int %s = %s' % (o.index, ccode('%s - 1' % end))
            loop_cond = '%s >= %s' % (o.index, ccode(start))
            loop_inc = '%s -= %s' % (o.index, o.limits[2])
        else:
            loop_init = 'int %s = %s' % (o.index, ccode(start))
            loop_cond = '%s < %s' % (o.index, ccode(end))
            loop_inc = '%s += %s' % (o.index, o.limits[2])

        # Append unbounded indices, if any
        if o.uindices:
            uinit = ['%s = %s' % (i.index, ccode(i.start)) for i in o.uindices]
            loop_init = c.Line(', '.join([loop_init] + uinit))
            ustep = ['%s = %s' % (i.index, ccode(i.step)) for i in o.uindices]
            loop_inc = c.Line(', '.join([loop_inc] + ustep))

        # Create For header+body
        handle = c.For(loop_init, loop_cond, loop_inc, c.Block(body))

        # Attach pragmas, if any
        if o.pragmas:
            handle = c.Module(o.pragmas + (handle,))

        return handle

    def visit_Callable(self, o):
        body = flatten(self.visit(i) for i in o.children)
        params = runtime_arguments(o.parameters)
        decls = self._args_decl(params)
        casts = self._args_cast(params)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        return c.FunctionBody(signature, c.Block(casts + body))

    def visit_Operator(self, o):
        # Kernel signature and body
        body = flatten(self.visit(i) for i in o.children)
        params = runtime_arguments(o.parameters)
        decls = self._args_decl(params)
        casts = self._args_cast(params)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        retval = [c.Statement("return 0")]
        kernel = c.FunctionBody(signature, c.Block(casts + body + retval))

        # Elemental functions
        efuncs = [i.root.ccode for i in o.func_table.values() if i.local] + [blankline]

        # Header files, extra definitions, ...
        header = [c.Line(i) for i in o._headers]
        includes = [c.Include(i, system=False) for i in o._includes]
        includes += [blankline]
        cglobals = list(o._globals)
        if o._compiler.src_ext == 'cpp':
            cglobals += [c.Extern('C', signature)]
        cglobals = [i for j in cglobals for i in (j, blankline)]

        return c.Module(header + includes + cglobals + efuncs + [kernel])


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
        if queue is not None:
            ret.setdefault(tuple(queue), []).append(o)
        return ret

    visit_Element = visit_Expression
    visit_Call = visit_Expression


class MapExpressions(FindSections):

    """
    Map :class:`Expression` and :class:`Call` objects in the Iteration/Expression
    tree to their respective section.
    """

    def visit_Call(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        ret[o] = as_tuple(queue)
        return ret

    visit_Expression = visit_Call
    visit_Element = FindSections.visit_Node


class MapIteration(FindSections):

    """
    Map each :class:`Iteration` object in the Iteration/Expression tree to the
    enclosed :class:`Expression` and :class:`Call` objects.
    """

    def visit_Call(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        for i in as_tuple(queue):
            ret.setdefault(i, []).append(o)
        return ret

    visit_Expression = visit_Call
    visit_Element = FindSections.visit_Node


class FindSymbols(Visitor):

    @classmethod
    def default_retval(cls):
        return []

    """Find symbols in an Iteration/Expression tree.

    :param mode: Drive the search. Accepted values are: ::

        * 'kernel-data' (default): Collect :class:`SymbolicFunction` objects.
        * 'symbolics': Collect :class:`AbstractSymbol` objects.
        * 'symbolics-writes': Collect written :class:`AbstractSymbol` objects.
        * 'free-symbols': Collect all free symbols.
        * 'dimensions': Collect :class:`Dimension` objects only.
    """

    rules = {
        'kernel-data': lambda e: [i for i in e.functions if i.is_SymbolicFunction],
        'symbolics': lambda e: e.functions,
        'symbolics-writes': lambda e: as_tuple(e.write),
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


class ResolveTimeStepping(Transformer):
    """
    :class:`Transformer` class that creates a substitution dictionary
    for replacing :class:`Dimension` instances with explicit loop
    variables in :class:`Iteration` nodes. For stepping dimensions it
    also inserts the relevant definitions for buffer index variables,
    for exaple.:

        .. code-block:: c

           for (int t = 0; t < t_size; t += 1)
           {
               int t0 = (t) % 2;
               int t1 = (t + 1) % 2;
    """

    def visit_object(self, o, subs, **kwargs):
        return o, subs

    def visit_tuple(self, o, subs, **kwargs):
        visited = []
        for i in o:
            handle, subs = self.visit(i, subs, **kwargs)
            visited.append(handle)
        return tuple(visited), subs

    visit_list = visit_object

    def visit_Node(self, o, subs, **kwargs):
        rebuilt, _ = zip(*[self.visit(i, subs, **kwargs) for i in o.children])
        return o._rebuild(*rebuilt, **o.args_frozen), subs

    def visit_Iteration(self, o, subs, offsets=defaultdict(set)):
        nodes, subs = self.visit(o.children, subs, offsets=offsets)
        if o.dim.is_Stepping:
            # For SteppingDimension insert the explicit
            # definition of buffered variables, eg. t+1 => t1
            init = []
            for i, off in enumerate(filter_ordered(offsets[o.dim])):
                vname = Scalar(name="%s%d" % (o.dim.name, i), dtype=np.int32)
                value = (o.dim.parent + off) % o.dim.modulo
                init.append(UnboundedIndex(vname, value, value))
                subs[o.dim + off] = LoweredDimension(name=vname.name, origin=o.dim + off)
            # Always lower to symbol
            subs[o.dim.parent] = Scalar(name=o.dim.parent.name, dtype=np.int32)
            return o._rebuild(index=o.dim.parent.name, uindices=init,
                              limits=o.dim.parent.limits), subs
        else:
            return o._rebuild(*nodes), subs

    def visit_Expression(self, o, subs, offsets=defaultdict(set)):
        """Collect all offsets used with a dimension"""
        for dim, offs in o.stencil.entries:
            offsets[dim].update(offs)
        return o, subs

    def visit(self, o, subs=None, **kwargs):
        if subs is None:
            subs = {}
        obj, subs = super(ResolveTimeStepping, self).visit(o, subs, **kwargs)
        return obj, subs


def printAST(node, verbose=True):
    return PrintAST(verbose=verbose).visit(node)

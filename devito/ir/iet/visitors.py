"""
Visitor hierarchy to inspect and/or create IETs.

The main Visitor class is adapted from https://github.com/coneoproject/COFFEE.
"""

from __future__ import absolute_import

from collections import Iterable, OrderedDict
from operator import attrgetter

import cgen as c

from devito.cgen_utils import blankline, ccode
from devito.exceptions import VisitorException
from devito.ir.iet.nodes import Node
from devito.ir.support.space import Backward
from devito.tools import as_tuple, filter_sorted, flatten, ctypes_to_C, GenericVisitor


__all__ = ['FindNodes', 'FindSections', 'FindSymbols', 'MapExpressions',
           'IsPerfectIteration', 'SubstituteExpression', 'printAST', 'CGen',
           'Transformer', 'NestedTransformer', 'FindAdjacentIterations',
           'MapIteration']


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

    def visit_Conditional(self, o):
        self._depth += 1
        then_body = self.visit(o.then_body)
        self._depth -= 1
        if o.else_body:
            else_body = self.visit(o.else_body)
            return self.indent + "<If %s>\n%s\n<Else>\n%s" % (o.condition,
                                                              then_body, else_body)
        else:
            return self.indent + "<If %s>\n%s" % (o.condition, then_body)


class CGen(Visitor):

    """
    Return a representation of the Iteration/Expression tree as a :module:`cgen` tree.
    """

    def _args_decl(self, args):
        """Generate cgen declarations from an iterable of symbols and expressions."""
        ret = []
        for i in args:
            if i.is_Object:
                ret.append(c.Value('void', '*_%s' % i.name))
            elif i.is_Scalar:
                ret.append(c.Value('const %s' % c.dtype_to_ctype(i.dtype), i.name))
            elif i.is_Tensor:
                ret.append(c.Value(c.dtype_to_ctype(i.dtype),
                                   '*restrict %s_vec' % i.name))
            elif i.is_Lowered:
                ret.append(c.Value('const %s' % c.dtype_to_ctype(i.dtype), i.name))
            else:
                ret.append(c.Value('void', '*_%s' % i.name))
        return ret

    def _args_call(self, args):
        """Generate cgen function call arguments from an iterable of symbols and
        expressions."""
        ret = []
        for i in args:
            try:
                if i.is_Object:
                    ret.append('*_%s' % i.name)
                elif i.is_Array:
                    ret.append("(%s*)%s" % (c.dtype_to_ctype(i.dtype), i.name))
                elif i.is_Scalar:
                    ret.append(i.name)
                elif i.is_TensorFunction:
                    ret.append('%s_vec' % i.name)
            except AttributeError:
                ret.append(ccode(i))
        return ret

    def visit_ArrayCast(self, o):
        """
        Build cgen type casts for an :class:`AbstractFunction`.
        """
        f = o.function
        align = "__attribute__((aligned(64)))"
        shape = ''.join(["[%s]" % ccode(j) for j in f.symbolic_shape[1:]])
        lvalue = c.POD(f.dtype, '(*restrict %s)%s %s' % (f.name, shape, align))
        rvalue = '(%s (*)%s) %s' % (c.dtype_to_ctype(f.dtype), shape, '%s_vec' % f.name)
        return c.Initializer(lvalue, rvalue)

    def visit_PointerCast(self, o):
        """
        Build cgen pointer casts for an :class:`Object`.
        """
        ctype = ctypes_to_C(o.object.dtype)
        lvalue = c.Pointer(c.Value(ctype, o.object.name))
        rvalue = '(%s*) %s' % (ctype, '_%s' % o.object.name)
        return c.Initializer(lvalue, rvalue)

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
        arguments = self._args_call(o.params)
        return c.Statement('%s(%s)' % (o.name, ','.join(arguments)))

    def visit_Conditional(self, o):
        then_body = c.Block(self.visit(o.then_body))
        if o.else_body:
            else_body = c.Block(self.visit(o.else_body))
            return c.If(ccode(o.condition), then_body, else_body)
        else:
            return c.If(ccode(o.condition), then_body)

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

        # For backward direction flip loop bounds
        if o.direction == Backward:
            loop_init = 'int %s = %s' % (o.index, ccode(end))
            loop_cond = '%s >= %s' % (o.index, ccode(start))
            loop_inc = '%s -= %s' % (o.index, o.limits[2])
        else:
            loop_init = 'int %s = %s' % (o.index, ccode(start))
            loop_cond = '%s <= %s' % (o.index, ccode(end))
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
        params = o.parameters
        decls = self._args_decl(params)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        return c.FunctionBody(signature, c.Block(body))

    def visit_Operator(self, o):
        # Kernel signature and body
        body = flatten(self.visit(i) for i in o.children)
        params = o.parameters
        decls = self._args_decl(params)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        retval = [c.Statement("return 0")]
        kernel = c.FunctionBody(signature, c.Block(body + retval))

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

        * 'symbolics': Collect :class:`AbstractSymbol` objects.
        * 'symbolics-writes': Collect written :class:`AbstractSymbol` objects.
        * 'free-symbols': Collect all free symbols.
    """

    rules = {
        'symbolics': lambda e: e.functions,
        'symbolics-writes': lambda e: as_tuple(e.write),
        'free-symbols': lambda e: e.free_symbols,
        'defines': lambda e: as_tuple(e.defines),
    }

    def __init__(self, mode='symbolics'):
        super(FindSymbols, self).__init__()
        self.rule = self.rules[mode]

    def visit_tuple(self, o):
        symbols = flatten([self.visit(i) for i in o])
        return filter_sorted(symbols, key=attrgetter('name'))

    def visit_Iteration(self, o):
        symbols = flatten([self.visit(i) for i in o.children])
        symbols += self.rule(o)
        return filter_sorted(symbols, key=attrgetter('name'))

    def visit_Expression(self, o):
        return filter_sorted([f for f in self.rule(o)], key=attrgetter('name'))

    visit_ArrayCast = visit_Expression
    visit_PointerCast = visit_Expression
    visit_Call = visit_Expression


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
            if i and ret['seen_iteration'] is True:
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
        if not found:
            return False
        return all(self.visit(i, found=found, **kwargs) for i in o.children)

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


def printAST(node, verbose=True):
    return PrintAST(verbose=verbose).visit(node)

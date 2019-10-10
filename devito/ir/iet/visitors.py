"""
Visitor hierarchy to inspect and/or create IETs.

The main Visitor class is adapted from https://github.com/coneoproject/COFFEE.
"""

from collections import OrderedDict
from collections.abc import Iterable
from operator import attrgetter

import cgen as c

from devito.exceptions import VisitorException
from devito.ir.iet.nodes import Node, Iteration, Expression, Call
from devito.ir.support.space import Backward
from devito.symbolics import ccode
from devito.tools import GenericVisitor, as_tuple, filter_sorted, flatten


__all__ = ['FindNodes', 'FindSections', 'FindSymbols', 'MapSections', 'MapNodes',
           'IsPerfectIteration', 'XSubs', 'printAST', 'CGen', 'Transformer',
           'FindAdjacent']


class Visitor(GenericVisitor):

    def visit_Node(self, o, **kwargs):
        return self._visit(o.children, **kwargs)

    def reuse(self, o, *args, **kwargs):
        """A visit method to reuse a node, ignoring children."""
        return o

    def maybe_rebuild(self, o, *args, **kwargs):
        """A visit method that rebuilds nodes if their children have changed."""
        ops, okwargs = o.operands()
        new_ops = [self._visit(op, *args, **kwargs) for op in ops]
        if all(a is b for a, b in zip(ops, new_ops)):
            return o
        return o._rebuild(*new_ops, **okwargs)

    def always_rebuild(self, o, *args, **kwargs):
        """A visit method that always rebuilds nodes."""
        ops, okwargs = o.operands()
        new_ops = [self._visit(op, *args, **kwargs) for op in ops]
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
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + '<Callable %s>\n%s' % (o.name, body)

    def visit_list(self, o):
        return ('\n').join([self._visit(i) for i in o])

    def visit_tuple(self, o):
        return '\n'.join([self._visit(i) for i in o])

    def visit_Block(self, o):
        self._depth += 1
        if self.verbose:
            body = [self._visit(o.header), self._visit(o.body), self._visit(o.footer)]
        else:
            body = [self._visit(o.body)]
        self._depth -= 1
        return self.indent + "%s\n%s" % (o.__repr__(), '\n'.join(body))

    def visit_Iteration(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        if self.verbose:
            detail = '::%s::%s::%s' % (o.index, o.limits, o.offsets)
            props = [str(i) for i in o.properties]
            props = '[%s] ' % ','.join(props) if props else ''
        else:
            detail, props = '', ''
        return self.indent + "<%sIteration %s%s>\n%s" % (props, o.dim.name, detail, body)

    def visit_While(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + "<While %s>\n%s" % (o.condition, body)

    def visit_Expression(self, o):
        if self.verbose:
            body = "%s = %s" % (o.expr.lhs, o.expr.rhs)
            return self.indent + "<Expression %s>" % body
        else:
            return self.indent + str(o)

    def visit_AugmentedExpression(self, o):
        if self.verbose:
            body = "%s %s= %s" % (o.expr.lhs, o.op, o.expr.rhs)
            return self.indent + "<%s %s>" % (o.__class__.__name__, body)
        else:
            return self.indent + str(o)

    def visit_ForeignExpression(self, o):
        if self.verbose:
            return self.indent + "<Expression %s>" % o.expr
        else:
            return self.indent + str(o)

    def visit_HaloSpot(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + "%s\n%s" % (o.__repr__(), body)

    def visit_Conditional(self, o):
        self._depth += 1
        then_body = self._visit(o.then_body)
        self._depth -= 1
        if o.else_body:
            else_body = self._visit(o.else_body)
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
            if i.is_Tensor:
                ret.append(c.Value('%srestrict' % i._C_typename, i._C_name))
            elif i.is_AbstractObject or i.is_Symbol:
                ret.append(c.Value(i._C_typename, i._C_name))
            else:
                ret.append(c.Value('void', '*_%s' % i._C_name))
        return ret

    def _args_call(self, args):
        """Generate cgen function call arguments from an iterable of symbols and
        expressions."""
        ret = []
        for i in args:
            try:
                if isinstance(i, Call):
                    ret.append(self.visit(i).text)
                elif i.is_LocalObject:
                    ret.append('&%s' % i._C_name)
                elif i.is_Array:
                    ret.append("(%s)%s" % (i._C_typename, i.name))
                else:
                    ret.append(i._C_name)
            except AttributeError:
                ret.append(ccode(i))
        return ret

    def visit_ArrayCast(self, o):
        f = o.function
        # rvalue
        shape = ''.join("[%s]" % ccode(i) for i in o.castshape)
        if f.is_DiscreteFunction:
            rvalue = '(%s (*)%s) %s->%s' % (f._C_typedata, shape, f._C_name,
                                            f._C_field_data)
        else:
            rvalue = '(%s (*)%s) %s' % (f._C_typedata, shape, f._C_name)
        # lvalue
        lvalue = c.AlignedAttribute(f._data_alignment,
                                    c.Value(f._C_typedata,
                                            '(*restrict %s)%s' % (f.name, shape)))
        return c.Initializer(lvalue, rvalue)

    def visit_tuple(self, o):
        return tuple(self._visit(i) for i in o)

    def visit_Block(self, o):
        body = flatten(self._visit(i) for i in o.children)
        return c.Module(o.header + (c.Block(body),) + o.footer)

    def visit_List(self, o):
        body = flatten(self._visit(i) for i in o.children)
        return c.Module(o.header + (c.Collection(body),) + o.footer)

    def visit_Section(self, o):
        header = c.Comment("Begin %s" % o.name)
        body = flatten(self._visit(i) for i in o.children)
        footer = c.Comment("End %s" % o.name)
        return c.Module([header] + body + [footer])

    def visit_Element(self, o):
        return o.element

    def visit_Expression(self, o):
        return c.Assign(ccode(o.expr.lhs, dtype=o.dtype),
                        ccode(o.expr.rhs, dtype=o.dtype))

    def visit_AugmentedExpression(self, o):
        return c.Statement("%s %s= %s" % (ccode(o.expr.lhs, dtype=o.dtype), o.op,
                                          ccode(o.expr.rhs, dtype=o.dtype)))

    def visit_LocalExpression(self, o):
        if o.write.is_Array:
            lhs = '%s%s' % (
                o.expr.lhs.name,
                ''.join(['[%s]' % d.symbolic_size for d in o.expr.lhs.dimensions])
            )
        else:
            lhs = ccode(o.expr.lhs, dtype=o.dtype)

        return c.Initializer(c.Value(o.expr.lhs._C_typedata, lhs),
                             ccode(o.expr.rhs, dtype=o.dtype))

    def visit_ForeignExpression(self, o):
        return c.Statement(ccode(o.expr))

    def visit_Call(self, o):
        arguments = self._args_call(o.arguments)
        code = '%s(%s)' % (o.name, ','.join(arguments))
        return c.Statement(code)

    def visit_Conditional(self, o):
        then_body = c.Block(self._visit(o.then_body))
        if o.else_body:
            else_body = c.Block(self._visit(o.else_body))
            return c.If(ccode(o.condition), then_body, else_body)
        else:
            return c.If(ccode(o.condition), then_body)

    def visit_Iteration(self, o):
        body = flatten(self._visit(i) for i in o.children)

        # Start
        if o.offsets[0] != 0:
            _min = str(o.limits[0] + o.offsets[0])
            try:
                _min = eval(_min)
            except (NameError, TypeError):
                pass
        else:
            _min = o.limits[0]

        # Bound
        if o.offsets[1] != 0:
            _max = str(o.limits[1] + o.offsets[1])
            try:
                _max = eval(_max)
            except (NameError, TypeError):
                pass
        else:
            _max = o.limits[1]

        # For backward direction flip loop bounds
        if o.direction == Backward:
            loop_init = 'int %s = %s' % (o.index, ccode(_max))
            loop_cond = '%s >= %s' % (o.index, ccode(_min))
            loop_inc = '%s -= %s' % (o.index, o.limits[2])
        else:
            loop_init = 'int %s = %s' % (o.index, ccode(_min))
            loop_cond = '%s <= %s' % (o.index, ccode(_max))
            loop_inc = '%s += %s' % (o.index, o.limits[2])

        # Append unbounded indices, if any
        if o.uindices:
            uinit = ['%s = %s' % (i.name, ccode(i.symbolic_min)) for i in o.uindices]
            loop_init = c.Line(', '.join([loop_init] + uinit))
            ustep = ['%s = %s' % (i.name, ccode(i.symbolic_incr)) for i in o.uindices]
            loop_inc = c.Line(', '.join([loop_inc] + ustep))

        # Create For header+body
        handle = c.For(loop_init, loop_cond, loop_inc, c.Block(body))

        # Attach pragmas, if any
        if o.pragmas:
            handle = c.Module(o.pragmas + (handle,))

        return handle

    def visit_While(self, o):
        condition = ccode(o.condition)
        if o.body:
            body = flatten(self._visit(i) for i in o.children)
            return c.While(condition, body)
        else:
            # Hack: cgen doesn't support body-less while-loops, i.e. `while(...);`
            return c.Statement('while(%s)' % condition)

    def visit_Callable(self, o):
        body = flatten(self._visit(i) for i in o.children)
        decls = self._args_decl(o.parameters)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        return c.FunctionBody(signature, c.Block(body))

    def visit_HaloSpot(self, o):
        body = flatten(self._visit(i) for i in o.children)
        return c.Collection(body)

    def visit_Operator(self, o):
        blankline = c.Line("")

        # Kernel signature and body
        body = flatten(self._visit(i) for i in o.children)
        decls = self._args_decl(o.parameters)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        retval = [c.Statement("return 0")]
        kernel = c.FunctionBody(signature, c.Block(body + retval))

        # Elemental functions
        esigns = []
        efuncs = [blankline]
        for i in o._func_table.values():
            if i.local:
                esigns.append(c.FunctionDeclaration(c.Value(i.root.retval, i.root.name),
                                                    self._args_decl(i.root.parameters)))
                efuncs.extend([i.root.ccode, blankline])

        # Header files, extra definitions, ...
        header = [c.Line(i) for i in o._headers]
        includes = [c.Include(i, system=False) for i in o._includes]
        includes += [blankline]
        cdefs = [i._C_typedecl for i in o.parameters if i._C_typedecl is not None]
        cdefs = filter_sorted(cdefs, key=lambda i: i.tpname)
        if o._compiler.src_ext == 'cpp':
            cdefs += [c.Extern('C', signature)]
        cdefs = [i for j in cdefs for i in (j, blankline)]

        return c.Module(header + includes + cdefs +
                        esigns + [blankline, kernel] + efuncs)


class FindSections(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict()

    """
    Find all sections in an Iteration/Expression tree. A section is a map
    from an Iteration nest to the enclosed statements (e.g., Expressions,
    Conditionals, Calls, ...).
    """

    def visit_tuple(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        for i in o:
            ret = self._visit(i, ret=ret, queue=queue)
        return ret

    visit_list = visit_tuple

    def visit_Node(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        for i in o.children:
            ret = self._visit(i, ret=ret, queue=queue)
        return ret

    def visit_Iteration(self, o, ret=None, queue=None):
        if queue is None:
            queue = [o]
        else:
            queue.append(o)
        for i in o.children:
            ret = self._visit(i, ret=ret, queue=queue)
        queue.remove(o)
        return ret

    def visit_Simple(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        if queue is not None:
            ret.setdefault(tuple(queue), []).append(o)
        return ret

    def visit_Conditional(self, o, ret=None, queue=None):
        # Essentially like visit_Simple, but also go down through the children
        if ret is None:
            ret = self.default_retval()
        if queue is not None:
            ret.setdefault(tuple(queue), []).append(o)
        for i in o.children:
            ret = self._visit(i, ret=ret, queue=queue)
        return ret


class MapSections(FindSections):

    """
    Construct a mapper from Simple Nodes (i.e., Nodes that do *not* contain
    other Nodes, such as Expressions and Calls) to the enclosing Iteration nest.
    """

    def visit_Simple(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        ret[o] = as_tuple(queue)
        return ret

    visit_Conditional = FindSections.visit_Node


class MapNodes(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict()

    """
    Given an Iteration/Expression tree, build a mapper between parent and
    children nodes of given type.

    Parameters
    ----------
    parent_type : Node or str, optional
        By default, parents are of type Iteration. One can alternatively supply
        a different type. Optionally, the keyword 'any' can be supplied, in which
        case the parent can be a generic Node.
    child_types : Node or list of Node, optional
        By default, children of type Call and Expression are retrieved.
        One can alternatively supply one or more different types.
    mode : str, optional
        By default, all ancestors matching the ``parent_type`` are mapped to
        the nodes of type ``child_types`` retrieved by the search. This behaviour
        can be changed through this parameter. Accepted values are:
        - 'immediate': only the closest matching ancestor is mapped.
        - 'groupby': the matching ancestors are grouped together as a single key.
    """

    def __init__(self, parent_type=None, child_types=None, mode=None):
        super(MapNodes, self).__init__()
        if parent_type is None:
            self.parent_type = Iteration
        elif parent_type == 'any':
            self.parent_type = Node
        else:
            assert issubclass(parent_type, Node)
            self.parent_type = parent_type
        self.child_types = as_tuple(child_types) or (Call, Expression)
        assert mode in (None, 'immediate', 'groupby')
        self.mode = mode

    def visit_object(self, o, ret=None, **kwargs):
        return ret

    def visit_tuple(self, o, ret=None, parents=None, in_parent=False):
        for i in o:
            ret = self._visit(i, ret=ret, parents=parents, in_parent=in_parent)
        return ret

    def visit_Node(self, o, ret=None, parents=None, in_parent=False):
        if ret is None:
            ret = self.default_retval()
        if parents is None:
            parents = []
        if isinstance(o, self.child_types):
            if self.mode == 'groupby':
                ret.setdefault(as_tuple(parents), []).append(o)
            elif self.mode == 'immediate':
                if in_parent:
                    ret.setdefault(parents[-1], []).append(o)
                else:
                    ret.setdefault(None, []).append(o)
            else:
                for i in parents:
                    ret.setdefault(i, []).append(o)
        if isinstance(o, self.parent_type):
            parents.append(o)
            for i in o.children:
                ret = self._visit(i, ret=ret, parents=parents, in_parent=True)
            parents.remove(o)
        else:
            for i in o.children:
                ret = self._visit(i, ret=ret, parents=parents, in_parent=in_parent)

        return ret


class FindSymbols(Visitor):

    @classmethod
    def default_retval(cls):
        return []

    """
    Find symbols in an Iteration/Expression tree.

    Parameters
    ----------
    mode : str, optional
        Drive the search. Accepted:
        - ``symbolics``: Collect AbstractSymbol objects, default.
        - ``free-symbols``: Collect all free symbols.
        - ``defines``: Collect all defined (bound) objects.
    """

    rules = {
        'symbolics': lambda e: e.functions,
        'free-symbols': lambda e: e.free_symbols,
        'defines': lambda e: as_tuple(e.defines),
    }

    def __init__(self, mode='symbolics'):
        super(FindSymbols, self).__init__()
        self.rule = self.rules[mode]

    def visit_tuple(self, o):
        symbols = flatten([self._visit(i) for i in o])
        return filter_sorted(symbols, key=attrgetter('name'))

    visit_list = visit_tuple

    def visit_Iteration(self, o):
        symbols = flatten([self._visit(i) for i in o.children])
        symbols += self.rule(o)
        return filter_sorted(symbols, key=attrgetter('name'))

    visit_Block = visit_Iteration
    visit_Conditional = visit_Iteration

    def visit_Expression(self, o):
        return filter_sorted([f for f in self.rule(o)], key=attrgetter('name'))

    def visit_Call(self, o):
        symbols = self._visit(o.children)
        symbols.extend([f for f in self.rule(o)])
        return filter_sorted(symbols, key=attrgetter('name'))

    visit_ArrayCast = visit_Expression


class FindNodes(Visitor):

    @classmethod
    def default_retval(cls):
        return []

    """
    Find all instances of given type.

    Parameters
    ----------
    match : type
        Searched type.
    mode : str, optional
        Drive the search. Accepted:
        - ``type``: Collect all instances of type ``match``, default.
        - ``scope``: Collect the scope in which the object of type ``match``
                     appears.
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
            ret = self._visit(i, ret=ret)
        return ret

    visit_list = visit_tuple

    def visit_Node(self, o, ret=None):
        if ret is None:
            ret = self.default_retval()
        if self.rule(self.match, o):
            ret.append(o)
        for i in o.children:
            ret = self._visit(i, ret=ret)
        return ret


class FindAdjacent(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict([('seen_type', False)])

    """
    Return a mapper from nodes N in an Expression/Iteration tree to sequences of
    objects I = [I_0, I_1, ...] of type T, where N is the direct ancestor of
    the items in I and all items in I are adjacent nodes in the tree.
    """

    def __init__(self, match):
        super(FindAdjacent, self).__init__()
        self.match = as_tuple(match)

    def handler(self, o, parent=None, ret=None):
        if ret is None:
            ret = self.default_retval()
        if parent is None:
            return ret
        group = []
        for i in o:
            ret = self._visit(i, parent=parent, ret=ret)
            if i and ret['seen_type'] is True:
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

    def _post_visit(self, ret):
        ret.pop('seen_type', None)
        return ret

    def visit_object(self, o, parent=None, ret=None):
        return ret

    def visit_tuple(self, o, parent=None, ret=None):
        return self.handler(o, parent=parent, ret=ret)

    def visit_Node(self, o, parent=None, ret=None):
        ret = self.handler(o.children, parent=o, ret=ret)
        ret['seen_type'] = type(o) in self.match
        return ret


class IsPerfectIteration(Visitor):

    """
    Return True if an Iteration defines a perfect loop nest, False otherwise.
    """

    def visit_object(self, o, **kwargs):
        return False

    def visit_tuple(self, o, **kwargs):
        return all(self._visit(i, **kwargs) for i in o)

    def visit_Node(self, o, found=False, **kwargs):
        if not found:
            return False
        return all(self._visit(i, found=found, **kwargs) for i in o.children)

    def visit_Conditional(self, o, found=False, **kwargs):
        if not found:
            return False
        return all(self._visit(i, found=found, nomore=True) for i in o.children)

    def visit_Iteration(self, o, found=False, nomore=False):
        if found and nomore:
            return False
        nomore = len(o.nodes) > 1
        return all(self._visit(i, found=True, nomore=nomore) for i in o.children)


class Transformer(Visitor):

    """
    Given an Iteration/Expression tree T and a mapper from nodes in T to
    a set of new nodes L, M : N --> L, build a new Iteration/Expression tree T'
    where a node ``n`` in N is replaced with ``M[n]``.

    In the special case in which ``M[n]`` is None, ``n`` is dropped from T'.

    In the special case in which ``M[n]`` is an iterable of nodes, ``n`` is
    "extended" by pre-pending to its body the nodes in ``M[n]``.
    """

    def __init__(self, mapper={}, nested=False):
        super(Transformer, self).__init__()
        self.mapper = mapper.copy()
        self.nested = nested

    def visit_object(self, o, **kwargs):
        return o

    def visit_tuple(self, o, **kwargs):
        visited = tuple(self._visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            handle = self.mapper[o]
            if handle is None:
                # None -> drop `o`
                return None
            elif isinstance(handle, Iterable):
                # Iterable -> inject `handle` into `o`'s children
                if not o.children:
                    raise VisitorException
                if self.nested:
                    children = [self._visit(i, **kwargs) for i in o.children]
                else:
                    children = o.children
                children = (tuple(handle) + children[0],) + tuple(children[1:])
                return o._rebuild(*children, **o.args_frozen)
            else:
                # Replace `o` with `handle`
                if self.nested:
                    children = [self._visit(i, **kwargs) for i in handle.children]
                    return handle._rebuild(*children, **handle.args_frozen)
                else:
                    return handle._rebuild(**handle.args)
        else:
            children = [self._visit(i, **kwargs) for i in o.children]
            return o._rebuild(*children, **o.args_frozen)

    def visit_Operator(self, o, **kwargs):
        raise ValueError("Cannot apply a Transformer visitor to an Operator directly")


class XSubs(Transformer):
    """
    Transformer that performs substitutions on Expressions
    in a given tree, akin to SymPy's ``subs``.

    Parameters
    ----------
    mapper : dict, optional
        The substitution rules.
    replacer : callable, optional
        An ad-hoc function to perform the substitution. Defaults to SymPy's ``subs``.
    """

    def __init__(self, mapper=None, replacer=None):
        super(XSubs, self).__init__()
        self.replacer = replacer or (lambda i: i.subs(mapper))

    def visit_Expression(self, o):
        return o._rebuild(expr=self.replacer(o.expr))


def printAST(node, verbose=True):
    return PrintAST(verbose=verbose)._visit(node)

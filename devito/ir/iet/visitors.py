"""
Visitor hierarchy to inspect and/or create IETs.

The main Visitor class is adapted from https://github.com/coneoproject/COFFEE.
"""

from collections import OrderedDict
from collections.abc import Iterable
from itertools import chain, groupby
import ctypes

import cgen as c
from sympy import IndexedBase

from devito.exceptions import VisitorException
from devito.ir.iet.nodes import (Node, Iteration, Expression, ExpressionBundle,
                                 Call, Lambda, BlankLine, Section)
from devito.ir.support.space import Backward
from devito.symbolics import ListInitializer, ccode, uxreplace
from devito.tools import (GenericVisitor, as_tuple, ctypes_to_cstr, filter_ordered,
                          filter_sorted, flatten, is_external_ctype, c_restrict_void_p)
from devito.types.basic import AbstractFunction, Basic
from devito.types import (ArrayObject, CompositeObject, Dimension, Pointer,
                          IndexedData, DeviceMap)


__all__ = ['FindNodes', 'FindSections', 'FindSymbols', 'MapExprStmts', 'MapNodes',
           'IsPerfectIteration', 'printAST', 'CGen', 'CInterface', 'Transformer',
           'Uxreplace']


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

    def visit_Callable(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + '<Callable %s>\n%s' % (o.name, body)

    def visit_CallableBody(self, o):
        self._depth += 1
        body = [self._visit(o.init), self._visit(o.unpacks), self._visit(o.body)]
        self._depth -= 1
        return self.indent + "%s\n%s" % (o.__repr__(), '\n'.join([i for i in body if i]))

    def visit_list(self, o):
        return ('\n').join([self._visit(i) for i in o])

    def visit_tuple(self, o):
        return '\n'.join([self._visit(i) for i in o])

    def visit_List(self, o):
        self._depth += 1
        if self.verbose:
            body = [self._visit(o.header), self._visit(o.body), self._visit(o.footer)]
        else:
            body = [self._visit(o.body)]
        self._depth -= 1
        return self.indent + "%s\n%s" % (o.__repr__(), '\n'.join(body))

    def visit_TimedList(self, o):
        self._depth += 1
        body = [self._visit(o.body)]
        self._depth -= 1
        return self.indent + "%s\n%s" % (o.__repr__(), '\n'.join(body))

    def visit_Iteration(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        if self.verbose:
            detail = '::%s::%s' % (o.index, o.limits)
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

    def __init__(self, *args, compiler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiler = compiler

    # The following mappers may be customized by subclasses (that is,
    # backend-specific CGen-erators)
    _qualifiers_mapper = {
        'is_const': 'const',
        'is_volatile': 'volatile',
        '_mem_constant': 'static',
        '_mem_shared': '',
    }
    _restrict_keyword = 'restrict'

    def _gen_struct_decl(self, obj, masked=()):
        """
        Convert ctypes.Struct -> cgen.Structure.
        """
        ctype = obj._C_ctype
        while issubclass(ctype, ctypes._Pointer):
            ctype = ctype._type_

        if not issubclass(ctype, ctypes.Structure):
            return None

        try:
            return obj._C_typedecl
        except AttributeError:
            pass

        # Most of the times we end up here -- a generic procedure to
        # automatically derive a cgen.Structure from the object _C_ctype

        try:
            fields = obj.fields
        except AttributeError:
            fields = (None,)*len(ctype._fields_)

        entries = []
        for i, (n, ct) in zip(fields, ctype._fields_):
            try:
                entries.append(self._gen_value(i, 0, masked=('const',)))
            except AttributeError:
                cstr = ctypes_to_cstr(ct)
                if ct is c_restrict_void_p:
                    cstr = '%srestrict' % cstr
                entries.append(c.Value(cstr, n))

        return c.Struct(ctype.__name__, entries)

    def _gen_value(self, obj, level=2, masked=()):
        qualifiers = [v for k, v in self._qualifiers_mapper.items()
                      if getattr(obj, k, False) and v not in masked]

        if (obj._mem_stack or obj._mem_constant) and level == 2:
            strtype = obj._C_typedata
            strshape = ''.join('[%s]' % ccode(i) for i in obj.symbolic_shape)
        else:
            strtype = ctypes_to_cstr(obj._C_ctype)
            strshape = ''
            if isinstance(obj, (AbstractFunction, IndexedData)) and level >= 1:
                strtype = '%s%s' % (strtype, self._restrict_keyword)
        strtype = ' '.join(qualifiers + [strtype])

        strname = obj._C_name
        strobj = '%s%s' % (strname, strshape)

        try:
            if obj.cargs:
                strobj = MultilineCall(strobj, obj.cargs, True)
        except AttributeError:
            pass

        value = c.Value(strtype, strobj)

        try:
            if obj.is_AbstractFunction and obj._data_alignment and level == 2:
                value = c.AlignedAttribute(obj._data_alignment, value)
        except AttributeError:
            pass

        try:
            if obj.initvalue is not None and level == 2:
                init = ListInitializer(obj.initvalue)
                if not obj._mem_constant or init.is_numeric:
                    value = c.Initializer(value, ccode(init))
        except AttributeError:
            pass

        return value

    def _gen_rettype(self, obj):
        try:
            return self._gen_value(obj, 0).typename
        except AttributeError:
            assert isinstance(obj, str)
            return obj

    def _args_decl(self, args):
        """Generate cgen declarations from an iterable of symbols and expressions."""
        return [self._gen_value(i, 1) for i in args]

    def _args_call(self, args):
        """
        Generate cgen function call arguments from an iterable of symbols and expressions.
        """
        ret = []
        for i in args:
            try:
                if isinstance(i, Call):
                    ret.append(self._visit(i, nested_call=True))
                elif isinstance(i, Lambda):
                    ret.append(self._visit(i))
                else:
                    ret.append(i._C_name)
            except AttributeError:
                ret.append(ccode(i))
        return ret

    def _blankline_logic(self, children):
        """
        Generate cgen blank lines in between logical units.
        """
        candidates = (Expression, ExpressionBundle, Iteration, Section)

        processed = []
        for child in children:
            prev = None
            rebuilt = []
            for k, group in groupby(child, key=type):
                g = list(group)

                if k in (ExpressionBundle, Section) and len(g) >= 2:
                    # Separate consecutive Sections/ExpressionBundles with
                    # BlankLine
                    for i in g[:-1]:
                        rebuilt.append(i)
                        rebuilt.append(BlankLine)
                    rebuilt.append(g[-1])
                elif (k is Iteration and
                      prev is ExpressionBundle and
                      all(i.dim.is_Stencil for i in g)):
                    rebuilt.extend(g)
                elif prev in candidates and k in candidates:
                    rebuilt.append(BlankLine)
                    rebuilt.extend(g)
                else:
                    rebuilt.extend(g)

                prev = k

            processed.append(tuple(rebuilt))

        return tuple(processed)

    def visit_object(self, o):
        return o

    visit_Generable = visit_object
    visit_Collection = visit_object

    def visit_tuple(self, o):
        return tuple(self._visit(i) for i in o)

    def visit_PointerCast(self, o):
        f = o.function
        i = f.indexed

        if f.is_PointerArray:
            # lvalue
            lvalue = c.Value(i._C_typedata, '**%s' % f.name)

            # rvalue
            if isinstance(o.obj, ArrayObject):
                v = '%s->%s' % (o.obj.name, f._C_name)
            elif isinstance(o.obj, IndexedData):
                v = f._C_name
            else:
                assert False
            rvalue = '(%s**) %s' % (i._C_typedata, v)

        else:
            # lvalue
            if f.is_DiscreteFunction or (f.is_Array and f._mem_mapped):
                v = o.obj.name
            else:
                v = f.name
            if o.flat is None:
                shape = ''.join("[%s]" % ccode(i) for i in o.castshape)
                rshape = '(*)%s' % shape
                lvalue = c.Value(i._C_typedata, '(*restrict %s)%s' % (v, shape))
            else:
                rshape = '*'
                lvalue = c.Value(i._C_typedata, '*%s' % v)
            if o.alignment:
                lvalue = c.AlignedAttribute(f._data_alignment, lvalue)

            # rvalue
            if f.is_DiscreteFunction or (f.is_Array and f._mem_mapped):
                if isinstance(o.obj, IndexedData):
                    v = f._C_field_data
                elif isinstance(o.obj, DeviceMap):
                    v = f._C_field_dmap
                else:
                    assert False

                rvalue = '(%s %s) %s->%s' % (i._C_typedata, rshape, f._C_name, v)
            else:
                if isinstance(o.obj, Pointer):
                    v = o.obj.name
                else:
                    v = f._C_name

                rvalue = '(%s %s) %s' % (i._C_typedata, rshape, v)

        return c.Initializer(lvalue, rvalue)

    def visit_Dereference(self, o):
        a0, a1 = o.functions
        if a1.is_PointerArray or a1.is_TempFunction:
            i = a1.indexed
            if o.flat is None:
                shape = ''.join("[%s]" % ccode(i) for i in a0.symbolic_shape[1:])
                rvalue = '(%s (*)%s) %s[%s]' % (i._C_typedata, shape, a1.name,
                                                a1.dim.name)
                lvalue = c.AlignedAttribute(
                    a0._data_alignment,
                    c.Value(i._C_typedata, '(*restrict %s)%s' % (a0.name, shape))
                )
            else:
                rvalue = '(%s *) %s[%s]' % (i._C_typedata, a1.name, a1.dim.name)
                lvalue = c.AlignedAttribute(
                    a0._data_alignment, c.Value(i._C_typedata, '*restrict %s' % a0.name)
                )
        else:
            rvalue = '%s->%s' % (a1.name, a0._C_name)
            lvalue = self._gen_value(a0, 0)
        return c.Initializer(lvalue, rvalue)

    def visit_Block(self, o):
        body = flatten(self._visit(i) for i in self._blankline_logic(o.children))
        return c.Module(o.header + (c.Block(body),) + o.footer)

    def visit_List(self, o):
        body = flatten(self._visit(i) for i in self._blankline_logic(o.children))
        return c.Module(o.header + (c.Collection(body),) + o.footer)

    def visit_Section(self, o):
        body = flatten(self._visit(i) for i in o.children)
        if o.is_subsection:
            header = []
            footer = []
        else:
            header = [c.Comment("Begin %s" % o.name)]
            footer = [c.Comment("End %s" % o.name)]
        return c.Module(header + body + footer)

    def visit_Return(self, o):
        v = 'return'
        if o.value is not None:
            v += ' %s' % o.value
        return c.Statement(v)

    def visit_Definition(self, o):
        return self._gen_value(o.function)

    def visit_Expression(self, o):
        lhs = ccode(o.expr.lhs, dtype=o.dtype, compiler=self._compiler)
        rhs = ccode(o.expr.rhs, dtype=o.dtype, compiler=self._compiler)

        if o.init:
            code = c.Initializer(self._gen_value(o.expr.lhs, 0), rhs)
        else:
            code = c.Assign(lhs, rhs)

        if o.pragmas:
            code = c.Module(list(o.pragmas) + [code])

        return code

    def visit_AugmentedExpression(self, o):
        c_lhs = ccode(o.expr.lhs, dtype=o.dtype, compiler=self._compiler)
        c_rhs = ccode(o.expr.rhs, dtype=o.dtype, compiler=self._compiler)
        code = c.Statement("%s %s= %s" % (c_lhs, o.op, c_rhs))
        if o.pragmas:
            code = c.Module(list(o.pragmas) + [code])
        return code

    def visit_Call(self, o, nested_call=False):
        retobj = o.retobj
        cast = o.cast and self._gen_rettype(retobj)
        arguments = self._args_call(o.arguments)
        if retobj is None:
            return MultilineCall(o.name, arguments, nested_call, o.is_indirect, cast)
        else:
            call = MultilineCall(o.name, arguments, True, o.is_indirect, cast)
            if retobj.is_Indexed:
                return c.Assign(ccode(retobj), call)
            else:
                rettype = self._gen_rettype(retobj)
                return c.Initializer(c.Value(rettype, retobj._C_name), call)

    def visit_Conditional(self, o):
        try:
            then_body, else_body = self._blankline_logic(o.children)
        except ValueError:
            # Some special subclasses of Conditional such as ThreadedProdder
            # have zero children actually
            then_body, else_body = o.then_body, o.else_body
        then_body = c.Block(self._visit(then_body))
        if else_body:
            else_body = c.Block(self._visit(else_body))
            return c.If(ccode(o.condition), then_body, else_body)
        else:
            return c.If(ccode(o.condition), then_body)

    def visit_Iteration(self, o):
        body = flatten(self._visit(i) for i in self._blankline_logic(o.children))

        _min = o.limits[0]
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

            ustep = []
            for i in o.uindices:
                op = '=' if i.is_Modulo else '+='
                ustep.append('%s %s %s' % (i.name, op, ccode(i.symbolic_incr)))
            loop_inc = c.Line(', '.join([loop_inc] + ustep))

        # Create For header+body
        handle = c.For(loop_init, loop_cond, loop_inc, c.Block(body))

        # Attach pragmas, if any
        if o.pragmas:
            handle = c.Module(o.pragmas + (handle,))

        return handle

    def visit_Pragma(self, o):
        if len(o.pragmas) == 1:
            return o.pragmas[0]
        else:
            return c.Collection(o.pragmas)

    def visit_While(self, o):
        condition = ccode(o.condition)
        if o.body:
            body = flatten(self._visit(i) for i in o.children)
            return c.While(condition, c.Block(body))
        else:
            # Hack: cgen doesn't support body-less while-loops, i.e. `while(...);`
            return c.Statement('while(%s)' % condition)

    def visit_Callable(self, o):
        body = flatten(self._visit(i) for i in o.children)
        decls = self._args_decl(o.parameters)
        prefix = ' '.join(o.prefix + (self._gen_rettype(o.retval),))
        signature = c.FunctionDeclaration(c.Value(prefix, o.name), decls)
        return c.FunctionBody(signature, c.Block(body))

    def visit_CallableBody(self, o):
        body = []
        prev = None
        for i in o.children:
            v = self._visit(i)
            if v:
                if prev:
                    body.append(c.Line())
                prev = v
                body.extend(as_tuple(v))
        return c.Collection(body)

    def visit_Lambda(self, o):
        body = flatten(self._visit(i) for i in o.children)
        captures = [str(i) for i in o.captures]
        decls = [i.inline() for i in self._args_decl(o.parameters)]
        top = c.Line('[%s](%s)' % (', '.join(captures), ', '.join(decls)))
        return LambdaCollection([top, c.Block(body)])

    def visit_HaloSpot(self, o):
        body = flatten(self._visit(i) for i in o.children)
        return c.Collection(body)

    # Operator-handle machinery

    def _operator_includes(self, o):
        return [c.Include(i, system=(False if i.endswith('.h') else True))
                for i in o._includes]

    def _operator_typedecls(self, o, mode='all'):
        xfilter0 = lambda i: self._gen_struct_decl(i) is not None

        if mode == 'all':
            xfilter1 = xfilter0
        else:
            public_types = (AbstractFunction, CompositeObject)
            if mode == 'public':
                xfilter1 = lambda i: xfilter0(i) and isinstance(i, public_types)
            else:
                xfilter1 = lambda i: xfilter0(i) and not isinstance(i, public_types)

        # This is essentially to rule out vector types which are declared already
        # in some external headers
        xfilter = lambda i: xfilter1(i) and not is_external_ctype(i._C_ctype, o._includes)

        candidates = o.parameters + tuple(o._dspace.parts)
        typedecls = [self._gen_struct_decl(i) for i in candidates if xfilter(i)]
        for i in o._func_table.values():
            if not i.local:
                continue
            typedecls.extend([self._gen_struct_decl(j) for j in i.root.parameters
                              if xfilter(j)])
        typedecls = filter_sorted(typedecls, key=lambda i: i.tpname)

        return typedecls

    def _operator_globals(self, o, mode='all'):
        # Sorting for deterministic code generation
        v = sorted(o._globals, key=lambda i: i.name)

        return [self._gen_value(i) for i in v]

    def visit_Operator(self, o, mode='all'):
        # Kernel signature and body
        body = flatten(self._visit(i) for i in o.children)
        decls = self._args_decl(o.parameters)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        retval = [c.Line(), c.Statement("return 0")]
        kernel = c.FunctionBody(signature, c.Block(body + retval))

        # Elemental functions
        esigns = []
        efuncs = [blankline]
        for i in o._func_table.values():
            if i.local:
                rettype = self._gen_rettype(i.root.retval)
                prefix = ' '.join(i.root.prefix + (rettype,))
                esigns.append(c.FunctionDeclaration(c.Value(prefix, i.root.name),
                                                    self._args_decl(i.root.parameters)))
                efuncs.extend([self._visit(i.root), blankline])

        # Definitions
        headers = [c.Define(*i) for i in o._headers] + [blankline]

        # Header files
        includes = self._operator_includes(o) + [blankline]

        # Type declarations
        typedecls = self._operator_typedecls(o, mode)
        if mode in ('all', 'public') and o._compiler.src_ext in ('cpp', 'cu'):
            typedecls.append(c.Extern('C', signature))
        typedecls = [i for j in typedecls for i in (j, blankline)]

        # Global variables
        globs = self._operator_globals(o, mode)
        if globs:
            globs.append(blankline)

        return c.Module(headers + includes + typedecls + globs +
                        esigns + [blankline, kernel] + efuncs)


class CInterface(CGen):

    def _operator_includes(self, o):
        includes = super()._operator_includes(o)
        includes.append(c.Include("%s.h" % o.name, system=False))

        return includes

    def visit_Operator(self, o):
        # Generate the code for the cfile
        ccode = super().visit_Operator(o, mode='private')

        # Generate the code for the hfile
        typedecls = self._operator_typedecls(o, mode='public')
        guarded_typedecls = []
        for i in typedecls:
            guard = "DEVITO_%s" % i.tpname.upper()
            iflines = [c.Define(guard, ""), blankline, i, blankline]
            guarded_typedecl = c.IfNDef(guard, iflines, [])
            guarded_typedecls.extend([guarded_typedecl, blankline])

        decls = self._args_decl(o.parameters)
        signature = c.FunctionDeclaration(c.Value(o.retval, o.name), decls)
        hcode = c.Module(guarded_typedecls + [blankline, signature, blankline])

        return ccode, hcode


class FindSections(Visitor):

    @classmethod
    def default_retval(cls):
        return OrderedDict()

    """
    Find all sections in an Iteration/Expression tree. A section is a map
    from an Iteration nest to the enclosed statements (e.g., Expressions,
    Conditionals, Calls, ...).
    """

    def visit_object(self, o, ret=None, queue=None):
        return ret

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

    def visit_ExprStmt(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        if queue is not None:
            ret.setdefault(tuple(queue), []).append(o)
        return ret

    def visit_Conditional(self, o, ret=None, queue=None):
        # Essentially like visit_ExprStmt, but also go down through the children
        if ret is None:
            ret = self.default_retval()
        if queue is not None:
            ret.setdefault(tuple(queue), []).append(o)
        for i in o.children:
            ret = self._visit(i, ret=ret, queue=queue)
        return ret


class MapExprStmts(FindSections):

    """
    Construct a mapper from ExprStmts, i.e. expression statements such as Calls
    and Expressions, to their enclosing block (e.g., Iteration, Block).
    """

    def visit_ExprStmt(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        ret[o] = as_tuple(queue)
        return ret

    visit_Conditional = FindSections.visit_Iteration
    visit_Block = FindSections.visit_Iteration


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

    class Retval(list):
        def __init__(self, *retvals):
            elements = filter_ordered(flatten(retvals), key=id)
            super().__init__(elements)

    @classmethod
    def default_retval(cls):
        return cls.Retval()

    """
    Find symbols in an Iteration/Expression tree.

    Parameters
    ----------
    mode : str, optional
        Drive the search. Accepted:
        - `symbolics`: Collect all AbstractFunction objects, default
        - `basics`: Collect all Basic objects
        - `dimensions`: Collect all Dimensions
        - `indexeds`: Collect all Indexed objects
        - `indexedbases`: Collect all IndexedBase objects
        - `defines`: Collect all defined objects
        - `defines-aliases`: Collect all defined objects and their aliases
    """

    def _defines_aliases(n):
        retval = []
        for i in n.defines:
            f = i.function
            if f.is_ArrayBasic:
                retval.extend([f, f.indexed])
            else:
                retval.append(i)
        return tuple(retval)

    rules = {
        'symbolics': lambda n: n.functions,
        'basics': lambda n: [i for i in n.expr_symbols if isinstance(i, Basic)],
        'dimensions': lambda n: [i for i in n.expr_symbols if isinstance(i, Dimension)],
        'indexeds': lambda n: [i for i in n.expr_symbols if i.is_Indexed],
        'indexedbases': lambda n: [i for i in n.expr_symbols
                                   if isinstance(i, IndexedBase)],
        'writes': lambda n: as_tuple(n.writes),
        'defines': lambda n: as_tuple(n.defines),
        'globals': lambda n: [f.indexed for f in n.functions if f._mem_constant],
        'defines-aliases': _defines_aliases
    }

    def __init__(self, mode='symbolics'):
        super().__init__()

        modes = mode.split('|')
        if len(modes) == 1:
            self.rule = self.rules[mode]
        else:
            self.rule = lambda n: chain(*[self.rules[mode](n) for mode in modes])

    def _post_visit(self, ret):
        return sorted(ret, key=lambda i: str(i))

    def visit_tuple(self, o):
        return self.Retval(*[self._visit(i) for i in o])

    visit_list = visit_tuple

    def visit_Node(self, o):
        return self.Retval(self._visit(o.children), self.rule(o))

    def visit_Operator(self, o):
        ret = self._visit(o.body)
        ret.extend(flatten(self._visit(v) for v in o._func_table.values()))
        return self.Retval(ret, self.rule(o))


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


class IsPerfectIteration(Visitor):

    """
    Return True if an Iteration defines a perfect loop nest, False otherwise.
    """

    def __init__(self, depth=None):
        super(IsPerfectIteration, self).__init__()

        assert depth is None or isinstance(depth, Iteration)
        self.depth = depth

    def visit_object(self, o, **kwargs):
        return False

    def visit_tuple(self, o, found=False, nomore=False):
        nomore = nomore or (found and len(o) > 1)
        return all(self._visit(i, found=found, nomore=nomore) for i in o)

    def visit_Node(self, o, found=False, nomore=False):
        if not found:
            return False
        nomore = nomore or len(o.children) > 1
        return all(self._visit(i, found=found, nomore=nomore) for i in o.children)

    def visit_While(self, o, **kwargs):
        return False

    def visit_HaloSpot(self, o, found=False, **kwargs):
        if not found:
            return False
        return all(self._visit(i, found=found, nomore=True) for i in o.children)

    def visit_List(self, o, found=False, nomore=False):
        nomore = nomore or (found and (len(o.children) > 1) or o.header or o.footer)
        return all(self._visit(i, found=found, nomore=nomore) for i in o.children)

    def visit_Iteration(self, o, found=False, nomore=False):
        if found and nomore:
            return False
        if self.depth is o:
            return True
        nomore = len(o.nodes) > 1
        return all(self._visit(i, found=True, nomore=nomore) for i in o.children)


class Transformer(Visitor):

    """
    Given an Iteration/Expression tree T and a mapper M from nodes in T to
    a set of new nodes L, M : N --> L, build a new Iteration/Expression tree T'
    where a node ``n`` in N is replaced with ``M[n]``.

    In the special case in which ``M[n]`` is None, ``n`` is dropped from T'.

    In the special case in which ``M[n]`` is an iterable of nodes, ``n`` is
    "extended" by pre-pending to its body the nodes in ``M[n]``.
    """

    def __init__(self, mapper, nested=False):
        super(Transformer, self).__init__()
        self.mapper = mapper
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
                    return handle
        else:
            children = [self._visit(i, **kwargs) for i in o.children]
            return o._rebuild(*children, **o.args_frozen)

    def visit_Operator(self, o, **kwargs):
        raise ValueError("Cannot apply a Transformer visitor to an Operator directly")


class Uxreplace(Transformer):
    """
    Apply substitutions to SymPy objects wrapped in IET nodes.
    This is the IET-equivalent of `uxreplace` in the expressions layer.

    Parameters
    ----------
    mapper : dict
        The substitution rules.
    """

    def visit_Expression(self, o):
        return o._rebuild(expr=uxreplace(o.expr, self.mapper))

    def visit_Iteration(self, o):
        nodes = self._visit(o.nodes)
        limits = [uxreplace(i, self.mapper) for i in o.limits]
        return o._rebuild(nodes=nodes, limits=limits)

    def visit_Definition(self, o):
        try:
            return o._rebuild(function=self.mapper[o.function])
        except KeyError:
            return o

    def visit_Return(self, o):
        try:
            return o._rebuild(value=self.mapper[o.value])
        except KeyError:
            return o

    def visit_Callable(self, o):
        body = self._visit(o.body)
        parameters = [self.mapper.get(i, i) for i in o.parameters]
        return o._rebuild(body=body, parameters=parameters)

    def visit_Call(self, o):
        arguments = [uxreplace(i, self.mapper) for i in o.arguments]
        if o.retobj is not None:
            retobj = uxreplace(o.retobj, self.mapper)
            return o._rebuild(arguments=arguments, retobj=retobj)
        else:
            return o._rebuild(arguments=arguments)

    def visit_Conditional(self, o):
        condition = uxreplace(o.condition, self.mapper)
        then_body = self._visit(o.then_body)
        else_body = self._visit(o.else_body)
        return o._rebuild(condition=condition, then_body=then_body, else_body=else_body)

    def visit_PointerCast(self, o):
        function = self.mapper.get(o.function, o.function)
        obj = self.mapper.get(o.obj, o.obj)
        return o._rebuild(function=function, obj=obj)

    def visit_Pragma(self, o):
        arguments = [uxreplace(i, self.mapper) for i in o.arguments]
        return o._rebuild(arguments=arguments)

    def visit_PragmaTransfer(self, o):
        function = uxreplace(o.function, self.mapper)
        arguments = [uxreplace(i, self.mapper) for i in o.arguments]
        return o._rebuild(function=function, arguments=arguments)

    def visit_HaloSpot(self, o):
        hs = o.halo_scheme
        fmapper = {self.mapper.get(k, k): v for k, v in hs.fmapper.items()}
        halo_scheme = hs.build(fmapper, hs.honored)
        body = self._visit(o.body)
        return o._rebuild(halo_scheme=halo_scheme, body=body)

    visit_ThreadedProdder = visit_Call


# Utils

blankline = c.Line("")


def printAST(node, verbose=True):
    return PrintAST(verbose=verbose)._visit(node)


class LambdaCollection(c.Collection):
    pass


class MultilineCall(c.Generable):

    def __init__(self, name, arguments, is_expr=False, is_indirect=False, cast=None):
        self.name = name
        self.arguments = as_tuple(arguments)
        self.is_expr = is_expr
        self.is_indirect = is_indirect
        self.cast = cast

    def generate(self):
        if not self.is_indirect:
            tip = "%s(" % self.name
        else:
            tip = "%s%s" % (self.name, ',' if self.arguments else '')
        processed = []
        for i in self.arguments:
            if isinstance(i, (MultilineCall, LambdaCollection)):
                lines = list(i.generate())
                if len(lines) > 1:
                    yield tip + ",".join(processed + [lines[0]])
                    for line in lines[1:-1]:
                        yield line
                    tip = ""
                    processed = [lines[-1]]
                else:
                    assert len(lines) == 1
                    processed.append(lines[0])
            else:
                processed.append(str(i))
        tip = tip + ",".join(processed)
        if not self.is_indirect:
            tip += ")"
        if not self.is_expr:
            tip += ";"
        if self.cast:
            tip = '(%s)%s' % (self.cast, tip)
        yield tip

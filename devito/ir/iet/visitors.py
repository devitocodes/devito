"""
Visitor hierarchy to inspect and/or create IETs.

The main Visitor class is adapted from https://github.com/coneoproject/COFFEE.
"""

from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from itertools import chain, groupby
from typing import Any, Generic, TypeVar
import ctypes

import cgen as c
from sympy import IndexedBase
from sympy.core.function import Application

from devito.exceptions import CompilationError
from devito.ir.iet.nodes import (Node, Iteration, Expression, ExpressionBundle,
                                 Call, Lambda, BlankLine, Section, ListMajor)
from devito.ir.support.space import Backward
from devito.symbolics import (FieldFromComposite, FieldFromPointer,
                              ListInitializer, uxreplace)
from devito.symbolics.extended_dtypes import NoDeclStruct
from devito.tools import (GenericVisitor, as_tuple, filter_ordered,
                          filter_sorted, flatten, is_external_ctype,
                          c_restrict_void_p, sorted_priority)
from devito.types.basic import AbstractFunction, AbstractSymbol, Basic
from devito.types import (ArrayObject, CompositeObject, Dimension, Pointer,
                          IndexedData, DeviceMap)


__all__ = ['FindApplications', 'FindNodes', 'FindWithin', 'FindSections',
           'FindSymbols', 'MapExprStmts', 'MapHaloSpots', 'MapNodes',
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


# Type variables for LazyVisitor
YieldType = TypeVar('YieldType', covariant=True)
FlagType = TypeVar('FlagType', covariant=True)
ResultType = TypeVar('ResultType', covariant=True)

# Describes the return type of a LazyVisitor visit method which yields objects of
# type YieldType and returns a FlagType (or NoneType)
LazyVisit = Generator[YieldType, None, FlagType]


class LazyVisitor(GenericVisitor, Generic[YieldType, ResultType, FlagType]):

    """
    A generic visitor that lazily yields results instead of flattening results
    from children at every step. Intermediate visit methods may return a flag
    of type FlagType in addition to yielding results; by default, the last flag
    returned by a child is the one propagated.

    Subclass-defined visit methods should be generators.
    """

    def lookup_method(self, instance) \
            -> Callable[..., LazyVisit[YieldType, FlagType]]:
        return super().lookup_method(instance)

    def _visit(self, o, *args, **kwargs) -> LazyVisit[YieldType, FlagType]:
        meth = self.lookup_method(o)
        flag = yield from meth(o, *args, **kwargs)
        return flag

    def _post_visit(self, ret: LazyVisit[YieldType, FlagType]) -> ResultType:
        return list(ret)

    def visit_object(self, o: object, **kwargs) -> LazyVisit[YieldType, FlagType]:
        yield from ()

    def visit_Node(self, o: Node, **kwargs) -> LazyVisit[YieldType, FlagType]:
        flag = yield from self._visit(o.children, **kwargs)
        return flag

    def visit_tuple(self, o: Sequence[Any], **kwargs) -> LazyVisit[YieldType, FlagType]:
        flag: FlagType = None
        for i in o:
            flag = yield from self._visit(i, **kwargs)
        return flag

    visit_list = visit_tuple


class PrintAST(Visitor):

    _depth = 0

    """
    Return a representation of the Iteration/Expression tree as a string,
    highlighting tree structure and node properties while dropping non-essential
    information.
    """

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    @classmethod
    def default_retval(cls):
        return "<>"

    @property
    def indent(self):
        return '  ' * self._depth

    def visit_Node(self, o):
        return self.indent + f'<{o.__class__.__name__}>'

    def visit_Generable(self, o):
        body = f" {str(o) if self.verbose else ''}"
        return self.indent + f'<C.{o.__class__.__name__}{body}>'

    def visit_Callable(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + f'<Callable {o.name}>\n{body}'

    def visit_CallableBody(self, o):
        self._depth += 1
        body = [self._visit(o.init), self._visit(o.unpacks), self._visit(o.body)]
        self._depth -= 1
        cbody = '\n'.join([i for i in body if i])
        return self.indent + f"{o.__repr__()}\n{cbody}"

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
        cbody = '\n'.join(body)
        return self.indent + f"{o.__repr__()}\n{cbody}"

    def visit_TimedList(self, o):
        self._depth += 1
        body = [self._visit(o.body)]
        self._depth -= 1
        cbody = '\n'.join(body)
        return self.indent + f"{o.__repr__()}\n{cbody}"

    def visit_Iteration(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        if self.verbose:
            detail = f'::{o.index}::{o.limits}'
            props = [str(i) for i in o.properties]
            if props:
                cprops = ','.join(props)
                props = f'[{cprops}] '
            else:
                props = ''
        else:
            detail, props = '', ''
        return self.indent + f"<{props}Iteration {o.dim.name}{detail}>\n{body}"

    def visit_While(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + f"<While {o.condition}>\n{body}"

    def visit_Expression(self, o):
        if self.verbose:
            body = f"{o.expr.lhs} = {o.expr.rhs}"
            return self.indent + f"<Expression {body}>"
        else:
            return self.indent + str(o)

    def visit_AugmentedExpression(self, o):
        if self.verbose:
            body = f"{o.expr.lhs} {o.op}= {o.expr.rhs}"
            return self.indent + f"<{o.__class__.__name__} {body}>"
        else:
            return self.indent + str(o)

    def visit_HaloSpot(self, o):
        self._depth += 1
        body = self._visit(o.children)
        self._depth -= 1
        return self.indent + f"{o.__repr__()}\n{body}"

    def visit_Conditional(self, o):
        self._depth += 1
        then_body = self._visit(o.then_body)
        self._depth -= 1
        if o.else_body:
            else_body = self._visit(o.else_body)
            return self.indent + f"<If {o.condition}>\n{then_body}\n<Else>\n{else_body}"
        else:
            return self.indent + f"<If {o.condition}>\n{then_body}"


class CGen(Visitor):

    """
    Return a representation of the Iteration/Expression tree as a :module:`cgen` tree.
    """

    def __init__(self, *args, printer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if printer is None:
            from devito.passes.iet.languages.C import CPrinter
            printer = CPrinter
        self.printer = printer

    def ccode(self, expr, **kwargs):
        return self.printer(settings=kwargs).doprint(expr, None)

    @property
    def _qualifiers_mapper(self):
        return self.printer._qualifiers_mapper

    @property
    def _restrict_keyword(self):
        return self.printer._restrict_keyword

    def _gen_struct_decl(self, obj, masked=()):
        """
        Convert ctypes.Struct -> cgen.Structure.
        """
        ctype = obj._C_ctype
        try:
            while issubclass(ctype, ctypes._Pointer):
                ctype = ctype._type_

            if not issubclass(ctype, ctypes.Structure) or \
               issubclass(ctype, NoDeclStruct):
                return None
        except TypeError:
            # E.g., `ctype` is of type `dtypes_lowering.CustomDtype`
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
                cstr = self.ccode(ct)
                if ct is c_restrict_void_p:
                    cstr = f'{cstr}{self._restrict_keyword}'
                entries.append(c.Value(cstr, n))

        return c.Struct(ctype.__name__, entries)

    def _gen_value(self, obj, mode=1, masked=()):
        """
        Convert a devito.types.Basic object into a cgen declaration/definition.

        A Basic object may need to be declared and optionally defined in three
        different ways, which correspond to the three possible values of `mode`:

            * 0: Simple. E.g., `int a = 1`;
            * 1: Comprehensive. E.g., `const int *restrict a`, `int a[10]`;
            * 2: Declaration suitable for a function parameter list.
        """
        qualifiers = [v for k, v in self._qualifiers_mapper.items()
                      if getattr(obj.function, k, False) and v not in masked]

        if (obj._mem_stack or obj._mem_constant) and mode == 1:
            strtype = self.ccode(obj._C_typedata)
            strshape = ''.join(f'[{self.ccode(i)}]' for i in obj.symbolic_shape)
        else:
            strtype = self.ccode(obj._C_ctype)
            strshape = ''
            if isinstance(obj, (AbstractFunction, IndexedData)) and mode >= 1:
                if not obj._mem_stack:
                    strtype = f'{strtype}{self._restrict_keyword}'
        strtype = ' '.join(qualifiers + [strtype])

        if obj.is_LocalObject and obj._C_modifier is not None and mode == 2:
            strtype += obj._C_modifier

        strname = obj._C_name
        strobj = f'{strname}{strshape}'

        if obj.is_LocalObject and obj.cargs and mode == 1:
            arguments = [self.ccode(i) for i in obj.cargs]
            strobj = MultilineCall(strobj, arguments, True)

        value = c.Value(strtype, strobj)

        try:
            if obj.is_AbstractFunction and obj._data_alignment and mode == 1:
                value = c.AlignedAttribute(obj._data_alignment, value)
        except AttributeError:
            pass

        if obj.is_Array and obj.initvalue is not None and mode == 1:
            init = ListInitializer(obj.initvalue)
            if not obj._mem_constant or init.is_numeric:
                value = c.Initializer(value, self.ccode(init))
        elif obj.is_LocalObject and obj.initvalue is not None and mode == 1:
            value = c.Initializer(value, self.ccode(obj.initvalue))

        return value

    def _gen_rettype(self, obj):
        try:
            return self._gen_value(obj, 0).typename
        except AttributeError:
            pass
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, (FieldFromComposite, FieldFromPointer)):
            return self._gen_value(obj.function.base, 0).typename
        else:
            return None

    def _args_decl(self, args):
        """Generate cgen declarations from an iterable of symbols and expressions."""
        return [self._gen_value(i, 2) for i in args]

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
                ret.append(self.ccode(i))
        return ret

    def _gen_signature(self, o, is_declaration=False):
        decls = self._args_decl(o.parameters)

        prefix = ' '.join(o.prefix + (self._gen_rettype(o.retval),))

        if o.attributes:
            # NOTE: ugly, but I can't bother extending `c.FunctionDeclaration`
            # for such a tiny thing
            v = f"{' '.join(o.attributes)} {o.name}"
        else:
            v = o.name

        signature = c.FunctionDeclaration(c.Value(prefix, v), decls)

        if o.templates:
            tparams = ', '.join([i.inline() for i in self._args_decl(o.templates)])
            if is_declaration:
                signature = TemplateDecl(tparams, signature)
            else:
                signature = c.Template(tparams, signature)

        return signature

    def _blankline_logic(self, children):
        """
        Generate cgen blank lines in between logical units.
        """
        candidates = (Expression, ExpressionBundle, Iteration, Section,
                      ListMajor)

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
                elif (prev in candidates and k in candidates) or \
                     (prev is not None and k in (ListMajor, Section)) or \
                     (prev in (ListMajor, Section)):
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
        cstr = self.ccode(i._C_typedata)

        if f.is_PointerArray:
            # lvalue
            lvalue = c.Value(cstr, f'**{f.name}')

            # rvalue
            if isinstance(o.obj, ArrayObject):
                v = f'{o.obj.name}->{f._C_name}'
            elif isinstance(o.obj, IndexedData):
                v = f._C_name
            else:
                assert False
            rvalue = f'({cstr}**) {v}'

        else:
            # lvalue
            if f.is_DiscreteFunction or (f.is_Array and f._mem_mapped):
                v = o.obj.name
            else:
                v = f.name
            if o.flat is None:
                shape = ''.join(f"[{self.ccode(i)}]" for i in o.castshape)
                rshape = f'(*){shape}'
                if shape:
                    lvalue = c.Value(cstr, f'(*{self._restrict_keyword} {v}){shape}')
                else:
                    lvalue = c.Value(cstr, f'*{self._restrict_keyword} {v}')
            else:
                rshape = '*'
                lvalue = c.Value(cstr, f'*{v}')
            if o.alignment and f._data_alignment:
                lvalue = c.AlignedAttribute(f._data_alignment, lvalue)

            # rvalue
            if f.is_DiscreteFunction or (f.is_Array and f._mem_mapped):
                if isinstance(o.obj, IndexedData):
                    v = f._C_field_data
                elif isinstance(o.obj, DeviceMap):
                    v = f._C_field_dmap
                else:
                    assert False

                rvalue = f'({cstr} {rshape}) {f._C_name}->{v}'
            else:
                if isinstance(o.obj, Pointer):
                    v = o.obj.name
                else:
                    v = f._C_name

                rvalue = f'({cstr} {rshape}) {v}'

        return c.Initializer(lvalue, rvalue)

    def visit_Dereference(self, o):
        a0, a1 = o.functions
        if a1.is_PointerArray or a1.is_TempFunction:
            i = a1.indexed
            cstr = self.ccode(i._C_typedata)
            if o.flat is None:
                shape = ''.join(f"[{self.ccode(i)}]" for i in a0.symbolic_shape[1:])
                rvalue = f'({cstr} (*){shape}) {a1.name}[{a1.dim.name}]'
                lvalue = c.Value(cstr, f'(*{self._restrict_keyword} {a0.name}){shape}')
            else:
                rvalue = f'({cstr} *) {a1.name}[{a1.dim.name}]'
                lvalue = c.Value(cstr, f'*{self._restrict_keyword} {a0.name}')
            if a0._data_alignment:
                lvalue = c.AlignedAttribute(a0._data_alignment, lvalue)
        else:
            if a1.is_Symbol:
                rvalue = f'*{a1.name}'
            else:
                rvalue = f'{a1.name}->{a0._C_name}'
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
        return c.Module(body)

    def visit_Break(self, o):
        return c.Statement('break')

    def visit_Return(self, o):
        v = 'return'
        if o.value is not None:
            v += f' {self.ccode(o.value)}'
        return c.Statement(v)

    def visit_Definition(self, o):
        return self._gen_value(o.function)

    def visit_Expression(self, o):
        lhs = self.ccode(o.expr.lhs, dtype=o.dtype)
        rhs = self.ccode(o.expr.rhs, dtype=o.dtype)

        if o.init:
            code = c.Initializer(self._gen_value(o.expr.lhs, 0), rhs)
        else:
            code = c.Assign(lhs, rhs)

        if o.pragmas:
            code = c.Module(self._visit(o.pragmas) + (code,))

        return code

    def visit_AugmentedExpression(self, o):
        c_lhs = self.ccode(o.expr.lhs, dtype=o.dtype)
        c_rhs = self.ccode(o.expr.rhs, dtype=o.dtype)
        code = c.Statement(f"{c_lhs} {o.op}= {c_rhs}")
        if o.pragmas:
            code = c.Module(self._visit(o.pragmas) + (code,))
        return code

    def visit_Call(self, o, nested_call=False):
        retobj = o.retobj
        rettype = self._gen_rettype(retobj)
        cast = o.cast and rettype
        arguments = self._args_call(o.arguments)
        if retobj is None:
            return MultilineCall(o.name, arguments, nested_call, o.is_indirect,
                                 cast, o.templates)
        else:
            call = MultilineCall(o.name, arguments, True, o.is_indirect, cast,
                                 o.templates)
            if retobj.is_Indexed or \
               isinstance(retobj, (FieldFromComposite, FieldFromPointer)):
                return c.Assign(self.ccode(retobj), call)
            else:
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
            return c.If(self.ccode(o.condition), then_body, else_body)
        else:
            return c.If(self.ccode(o.condition), then_body)

    def visit_Iteration(self, o):
        body = flatten(self._visit(i) for i in self._blankline_logic(o.children))

        _min = o.limits[0]
        _max = o.limits[1]

        # For backward direction flip loop bounds
        if o.direction == Backward:
            loop_init = f'int {o.index} = {self.ccode(_max)}'
            loop_cond = f'{o.index} >= {self.ccode(_min)}'
            loop_inc = f'{o.index} -= {o.limits[2]}'
        else:
            loop_init = f'int {o.index} = {self.ccode(_min)}'
            loop_cond = f'{o.index} <= {self.ccode(_max)}'
            loop_inc = f'{o.index} += {o.limits[2]}'

        # Append unbounded indices, if any
        if o.uindices:
            uinit = [f'{i.name} = {self.ccode(i.symbolic_min)}' for i in o.uindices]
            loop_init = c.Line(', '.join([loop_init] + uinit))

            ustep = []
            for i in o.uindices:
                op = '=' if i.is_Modulo else '+='
                ustep.append(f'{i.name} {op} {self.ccode(i.symbolic_incr)}')
            loop_inc = c.Line(', '.join([loop_inc] + ustep))

        # Create For header+body
        handle = c.For(loop_init, loop_cond, loop_inc, c.Block(body))

        # Attach pragmas, if any
        if o.pragmas:
            pragmas = tuple(self._visit(i) for i in o.pragmas)
            handle = c.Module(pragmas + (handle,))

        return handle

    def visit_Pragma(self, o):
        return c.Pragma(o._generate)

    def visit_While(self, o):
        condition = self.ccode(o.condition)
        if o.body:
            body = flatten(self._visit(i) for i in o.children)
            return c.While(condition, c.Block(body))
        else:
            # Hack: cgen doesn't support body-less while-loops, i.e. `while(...);`
            return c.Statement(f'while({condition})')

    def visit_Callable(self, o):
        body = flatten(self._visit(i) for i in o.children)
        signature = self._gen_signature(o)
        return c.FunctionBody(signature, c.Block(body))

    def visit_MultiTraversable(self, o):
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

    def visit_Using(self, o):
        return c.Statement(f'using {str(o.name)}')

    def visit_UsingNamespace(self, o):
        return c.Statement(f'using namespace {str(o.namespace)}')

    def visit_Lambda(self, o):
        body = []
        for i in o.children:
            v = self._visit(i)
            if v:
                if body:
                    body.append(c.Line())
                body.extend(as_tuple(v))

        captures = [str(i) for i in o.captures]
        decls = [i.inline() for i in self._args_decl(o.parameters)]

        extra = []
        if o.special:
            extra.append(' ')
            extra.append(' '.join(str(i) for i in o.special))
        if o.attributes:
            extra.append(' ')
            extra.append(' '.join(f'[[{i}]]' for i in o.attributes))

        top = c.Line(f"[{', '.join(captures)}]({', '.join(decls)}){''.join(extra)}")
        return LambdaCollection([top, c.Block(body)])

    def visit_HaloSpot(self, o):
        body = flatten(self._visit(i) for i in o.children)
        return c.Collection(body)

    def visit_KernelLaunch(self, o):
        if o.templates:
            templates = f"<{','.join([str(i) for i in o.templates])}>"
        else:
            templates = ''

        launch_args = [o.grid, o.block]
        if o.shm is not None:
            launch_args.append(o.shm)
        if o.stream is not None:
            launch_args.append(o.stream)
        launch_config = ','.join(str(i) for i in launch_args)

        arguments = self._args_call(o.arguments)
        arguments = ','.join(arguments)

        return c.Statement(f'{o.name}{templates}<<<{launch_config}>>>({arguments})')

    # Operator-handle machinery
    def _operator_description(self, o):
        """
        Generate cgen description from an iterable of symbols and expressions.
        """
        if o.description:
            if isinstance(o.description, str):
                return [c.Comment(o.description), blankline]
            elif isinstance(o.description, Iterable):
                return [c.MultilineComment(o.description), blankline]
        else:
            return [c.Comment("Devito generated operator"), blankline]

    def _operator_includes(self, o):
        """
        Generate cgen includes from an iterable of symbols and expressions.
        """
        return [c.Include(i, system=(False if i.endswith('.h') else True))
                for i in o.includes] + [blankline]

    def _operator_namespaces(self, o):
        """
        Generate cgen namespaces from an iterable of symbols and expressions.
        """
        namespaces = [self._visit(i) for i in o.namespaces]
        if namespaces:
            namespaces.append(blankline)
        return namespaces

    def _operator_headers(self, o):
        """
        Generate cgen headers from an iterable of symbols and expressions.
        """
        headers = [c.Define(*as_tuple(i)) for i in o.headers]
        if headers:
            headers.append(blankline)
        return headers

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
        xfilter = lambda i: (xfilter1(i) and
                             not is_external_ctype(i._C_ctype, o._includes))

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
        signature = self._gen_signature(o)

        # Honor the `retstmt` flag if set
        if o.body.retstmt:
            retval = []
        else:
            retval = [c.Line(), c.Statement("return 0")]

        kernel = c.FunctionBody(signature, c.Block(body + retval))

        # Elemental functions
        esigns = []
        efuncs = [blankline]
        items = [i.root for i in o._func_table.values() if i.local]
        for i in sorted_efuncs(items):
            esigns.append(self._gen_signature(i, is_declaration=True))
            efuncs.extend([self._visit(i), blankline])

        # Top description
        description = self._operator_description(o)

        # Definitions
        headers = self._operator_headers(o)

        # Header files
        includes = self._operator_includes(o)

        # Namespaces
        namespaces = self._operator_namespaces(o)

        # Type declarations
        typedecls = self._operator_typedecls(o, mode)
        if mode in ('all', 'public') and o._compiler.src_ext in ('cpp', 'cu'):
            typedecls.append(c.Extern('C', signature))
        typedecls = [i for j in typedecls for i in (j, blankline)]

        # Global variables
        globs = self._operator_globals(o, mode)
        if globs:
            globs.append(blankline)

        return c.Module(description + headers + includes + namespaces + typedecls
                        + globs + esigns + [blankline, kernel] + efuncs)


class CInterface(CGen):

    def _operator_includes(self, o):
        includes = super()._operator_includes(o)
        includes.append(c.Include(f"{o.name}.h", system=False))

        return includes

    def visit_Operator(self, o):
        # Generate the code for the cfile
        ccode = super().visit_Operator(o, mode='private')

        # Generate the code for the hfile
        typedecls = self._operator_typedecls(o, mode='public')
        guarded_typedecls = []
        for i in typedecls:
            guard = f"DEVITO_{i.tpname.upper()}"
            iflines = [c.Define(guard, ""), blankline, i, blankline]
            guarded_typedecl = c.IfNDef(guard, iflines, [])
            guarded_typedecls.extend([guarded_typedecl, blankline])

        signature = self._gen_signature(o)
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

    visit_Call = visit_Conditional


class MapKind(FindSections):

    """
    Base class to construct mappers from Nodes of given type to their enclosing
    scope of Nodes.
    """

    # NOTE: Ideally, we would use a metaclass that dynamically constructs mappers
    # for the kind supplied by the caller, but it'd be overkill at the moment

    def visit_dummy(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        ret[o] = as_tuple(queue)
        return ret

    visit_Conditional = FindSections.visit_Iteration
    visit_Block = FindSections.visit_Iteration
    visit_Lambda = FindSections.visit_Iteration


class MapExprStmts(MapKind):

    visit_ExprStmt = MapKind.visit_dummy

    def visit_Call(self, o, ret=None, queue=None):
        if ret is None:
            ret = self.default_retval()
        ret[o] = as_tuple(queue)
        for i in o.children:
            ret = self._visit(i, ret=ret, queue=queue)
        return ret


class MapHaloSpots(MapKind):
    visit_HaloSpot = MapKind.visit_dummy


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
        super().__init__()
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


class FindSymbols(LazyVisitor[Any, list[Any], None]):

    """
    Find symbols in an Iteration/Expression tree.

    Parameters
    ----------
    mode : str, optional
        Drive the search. Accepted:
        - `symbolics`: Collect all AbstractFunction objects, default
        - `basics`: Collect all Basic objects
        - `abstractsymbols`: Collect all AbstractSymbol objects
        - `dimensions`: Collect all Dimensions
        - `indexeds`: Collect all Indexed objects
        - `indexedbases`: Collect all IndexedBase objects
        - `defines`: Collect all defined objects
        - `defines-aliases`: Collect all defined objects and their aliases
    """

    @staticmethod
    def _defines_aliases(n):
        for i in n.defines:
            f = i.function
            if f.is_ArrayBasic:
                yield from (f, f.indexed)
            else:
                yield i

    RulesDict = dict[str, Callable[[Node], Iterator[Any]]]
    rules: RulesDict = {
        'symbolics': lambda n: n.functions,
        'basics': lambda n: (i for i in n.expr_symbols if isinstance(i, Basic)),
        'symbols': lambda n: (i for i in n.expr_symbols
                              if isinstance(i, AbstractSymbol)),
        'dimensions': lambda n: (i for i in n.expr_symbols if isinstance(i, Dimension)),
        'indexeds': lambda n: (i for i in n.expr_symbols if i.is_Indexed),
        'indexedbases': lambda n: (i for i in n.expr_symbols
                                   if isinstance(i, IndexedBase)),
        'writes': lambda n: as_tuple(n.writes),
        'defines': lambda n: as_tuple(n.defines),
        'globals': lambda n: (f.base for f in n.functions if f._mem_global),
        'defines-aliases': _defines_aliases
    }

    def __init__(self, mode: str = 'symbolics') -> None:
        super().__init__()

        modes = mode.split('|')
        if len(modes) == 1:
            self.rule = self.rules[mode]
        else:
            self.rule = lambda n: chain(*[self.rules[mode](n) for mode in modes])

    def _post_visit(self, ret):
        return sorted(filter_ordered(ret, key=id), key=str)

    def visit_Node(self, o: Node) -> Iterator[Any]:
        yield from self._visit(o.children)
        yield from self.rule(o)

    def visit_ThreadedProdder(self, o) -> Iterator[Any]:
        # TODO: this handle required because ThreadedProdder suffers from the
        # long-standing issue affecting all Node subclasses which rely on
        # multiple inheritance
        yield from self._visit(o.then_body)
        yield from self.rule(o)

    def visit_Operator(self, o) -> Iterator[Any]:
        yield from self._visit(o.body)
        yield from self.rule(o)
        for i in o._func_table.values():
            yield from self._visit(i)


class FindNodes(LazyVisitor[Node, list[Node], None]):

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

    RulesDict = dict[str, Callable[[type, Node], bool]]
    rules: RulesDict = {
        'type': lambda match, o: isinstance(o, match),
        'scope': lambda match, o: match in flatten(o.children)
    }

    def __init__(self, match: type, mode: str = 'type') -> None:
        super().__init__()
        self.match = match
        self.rule = self.rules[mode]

    def visit_Node(self, o: Node, **kwargs) -> Iterator[Node]:
        if self.rule(self.match, o):
            yield o
        for i in o.children:
            yield from self._visit(i, **kwargs)


class FindWithin(FindNodes, LazyVisitor[Node, list[Node], bool]):

    """
    Like FindNodes, but given an additional parameter `within=(start, stop)`,
    it starts collecting matching nodes only after `start` is found, and stops
    collecting matching nodes after `stop` is found.
    """

    def __init__(self, match: type, start: Node, stop: Node | None = None) -> None:
        super().__init__(match)
        self.start = start
        self.stop = stop

    def visit_object(self, o: object, flag: bool = False) -> LazyVisit[Node, bool]:
        yield from ()
        return flag

    def visit_tuple(self, o: Sequence[Any], flag: bool = False) -> LazyVisit[Node, bool]:
        for el in o:
            # Yield results from visiting this element, and update the flag
            flag = yield from self._visit(el, flag=flag)

        return flag

    visit_list = visit_tuple

    def visit_Node(self, o: Node, flag: bool = False) -> LazyVisit[Node, bool]:
        flag = flag or (o is self.start)

        if flag and self.rule(self.match, o):
            yield o

        for child in o.children:
            # Yield results from this child and retrieve its flag
            nflag = yield from self._visit(child, flag=flag)

            # If we started collecting outside of here and the child found a stop,
            # don't visit the rest of the children
            if flag and not nflag:
                return False
            flag = nflag

        # Update the flag if we found a stop
        flag &= (o is not self.stop)

        return flag


ApplicationType = TypeVar('ApplicationType')


class FindApplications(LazyVisitor[ApplicationType, set[ApplicationType], None]):

    """
    Find all SymPy applied functions (aka, `Application`s). The user may refine
    the search by supplying a different target class.
    """

    def __init__(self, cls: type[ApplicationType] = Application):
        super().__init__()
        self.match = lambda i: isinstance(i, cls) and not isinstance(i, Basic)

    def _post_visit(self, ret):
        return set(ret)

    def visit_Expression(self, o: Expression, **kwargs) -> Iterator[ApplicationType]:
        yield from o.expr.find(self.match)

    def visit_Iteration(self, o: Iteration, **kwargs) -> Iterator[ApplicationType]:
        yield from self._visit(o.children)
        yield from o.symbolic_min.find(self.match)
        yield from o.symbolic_max.find(self.match)

    def visit_Call(self, o: Call, **kwargs) -> Iterator[ApplicationType]:
        for i in o.arguments:
            try:
                yield from i.find(self.match)
            except (AttributeError, TypeError):
                yield from self._visit(i)


class IsPerfectIteration(Visitor):

    """
    Return True if an Iteration defines a perfect loop nest, False otherwise.
    """

    def __init__(self, depth=None):
        super().__init__()

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
        super().__init__()
        self.mapper = mapper
        self.nested = nested

    def transform(self, o, handle, **kwargs):
        if handle is None:
            # None -> drop `o`
            return None
        elif isinstance(handle, Iterable):
            # Iterable -> inject `handle` into `o`'s children
            if not o.children:
                raise CompilationError("Cannot inject nodes in a leaf node")
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

    def visit_object(self, o, **kwargs):
        return o

    def visit_tuple(self, o, **kwargs):
        visited = tuple(self._visit(i, **kwargs) for i in o)
        return tuple(i for i in visited if i is not None)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        if o in self.mapper:
            handle = self.mapper[o]
            return self.transform(o, handle, **kwargs)
        children = [self._visit(i, **kwargs) for i in o.children]
        if o._traversable and not any(children) and any(o.children):
            return None
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
        dimension = uxreplace(o.dim, self.mapper)
        limits = [uxreplace(i, self.mapper) for i in o.limits]
        pragmas = self._visit(o.pragmas)

        uindices = [uxreplace(i, self.mapper) for i in o.uindices]
        uindices = filter_ordered(i for i in uindices if isinstance(i, Dimension))

        return o._rebuild(nodes=nodes, dimension=dimension, limits=limits,
                          pragmas=pragmas, uindices=uindices)

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
        arguments = []
        for i in o.arguments:
            if i in o.children:
                arguments.append(self._visit(i))
            else:
                arguments.append(uxreplace(i, self.mapper))
        if o.retobj is not None:
            retobj = uxreplace(o.retobj, self.mapper)
            return o._rebuild(arguments=arguments, retobj=retobj)
        else:
            return o._rebuild(arguments=arguments)

    def visit_Lambda(self, o):
        body = self._visit(o.body)
        parameters = [self.mapper.get(i, i) for i in o.parameters]
        return o._rebuild(body=body, parameters=parameters)

    def visit_Conditional(self, o):
        condition = uxreplace(o.condition, self.mapper)
        then_body = self._visit(o.then_body)
        else_body = self._visit(o.else_body)
        return o._rebuild(condition=condition, then_body=then_body,
                          else_body=else_body)

    def visit_PointerCast(self, o):
        function = self.mapper.get(o.function, o.function)
        obj = self.mapper.get(o.obj, o.obj)
        return o._rebuild(function=function, obj=obj)

    def visit_Dereference(self, o):
        pointee = self.mapper.get(o.pointee, o.pointee)
        pointer = self.mapper.get(o.pointer, o.pointer)
        return o._rebuild(pointee=pointee, pointer=pointer)

    def visit_Pragma(self, o):
        arguments = [uxreplace(i, self.mapper) for i in o.arguments]
        return o._rebuild(arguments=arguments)

    def visit_PragmaTransfer(self, o):
        function = uxreplace(o.function, self.mapper)
        arguments = [uxreplace(i, self.mapper) for i in o.arguments]
        if o.imask is None:
            return o._rebuild(function=function, arguments=arguments)

        # An `imask` may be None, a list of symbols/numbers, or a list of
        # 2-tuples representing ranges
        imask = []
        for v in o.imask:
            try:
                i, j = v
                imask.append((uxreplace(i, self.mapper),
                              uxreplace(j, self.mapper)))
            except TypeError:
                imask.append(uxreplace(v, self.mapper))
        return o._rebuild(function=function, imask=imask, arguments=arguments)

    def visit_ParallelTree(self, o):
        prefix = self._visit(o.prefix)
        body = self._visit(o.body)
        nthreads = self.mapper.get(o.nthreads, o.nthreads)
        return o._rebuild(prefix=prefix, body=body, nthreads=nthreads)

    def visit_HaloSpot(self, o):
        hs = o.halo_scheme
        fmapper = {self.mapper.get(k, k): v for k, v in hs.fmapper.items()}
        halo_scheme = hs.build(fmapper, hs.honored)
        body = self._visit(o.body)
        return o._rebuild(halo_scheme=halo_scheme, body=body)

    def visit_While(self, o, **kwargs):
        condition = uxreplace(o.condition, self.mapper)
        body = self._visit(o.body)
        return o._rebuild(condition=condition, body=body)

    visit_ThreadedProdder = visit_Call

    def visit_KernelLaunch(self, o):
        arguments = [uxreplace(i, self.mapper) for i in o.arguments]
        grid = self.mapper.get(o.grid, o.grid)
        block = self.mapper.get(o.block, o.block)
        stream = self.mapper.get(o.stream, o.stream)
        return o._rebuild(grid=grid, block=block, stream=stream,
                          arguments=arguments)


# Utils

blankline = c.Line("")


def printAST(node, verbose=True):
    return PrintAST(verbose=verbose)._visit(node)


class LambdaCollection(c.Collection):
    pass


class MultilineCall(c.Generable):

    def __init__(self, name, arguments, is_expr=False, is_indirect=False,
                 cast=None, templates=None):
        self.name = name
        self.arguments = as_tuple(arguments)
        self.is_expr = is_expr
        self.is_indirect = is_indirect
        self.cast = cast
        self.templates = templates

    def generate(self):
        if self.templates:
            tip = f"{self.name}<{', '.join(str(i) for i in self.templates)}>"
        else:
            tip = self.name
        if not self.is_indirect:
            tip = f"{tip}("
        else:
            cargs = ',' if self.arguments else ''
            tip = f"{tip}{cargs}"
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
            tip = f'({self.cast}){tip}'
        yield tip


class TemplateDecl(c.Template):

    # Workaround to generate ';' at the end

    def generate(self):
        return super().generate(with_semicolon=True)


def sorted_efuncs(efuncs):
    from devito.ir.iet.efunc import (CommCallable, DeviceFunction,
                                     ThreadCallable, ElementalFunction)

    priority = {
        DeviceFunction: 3,
        ThreadCallable: 2,
        ElementalFunction: 1,
        CommCallable: 1
    }
    return sorted_priority(efuncs, priority)

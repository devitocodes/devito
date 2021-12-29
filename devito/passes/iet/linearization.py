from collections import defaultdict
from functools import singledispatch

import numpy as np

from devito.data import FULL
from devito.ir import (BlankLine, DummyExpr, Dereference, Expression, List,
                       PointerCast, PragmaTransfer, FindNodes, FindSymbols,
                       Transformer)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.parpragma import PragmaLangBB
from devito.symbolics import (DefFunction, MacroArgument, ccode, retrieve_indexed,
                              uxreplace)
from devito.tools import Bunch, DefaultOrderedDict, filter_ordered, flatten, prod
from devito.types import Array, Symbol, FIndexed, Indexed, Wildcard
from devito.types.basic import IndexedData
from devito.types.dense import DiscreteFunction


__all__ = ['linearize']


def linearize(graph, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j]` -> `a[i*n + j]`. The row-major format
    of the underlying Function objects is honored.
    """
    # Simple data structure to avoid generation of duplicated code
    cache = defaultdict(lambda: Bunch(stmts0=[], stmts1=[], cbk=None))

    linearization(graph, cache=cache, **kwargs)


@iet_pass
def linearization(iet, **kwargs):
    """
    Carry out the actual work of `linearize`.
    """
    mode = kwargs['mode']
    sregistry = kwargs['sregistry']
    cache = kwargs['cache']

    # Pre-process the `mode` opt option
    # `mode` may be a callback describing what Function types, and under what
    # conditions, should linearization be applied
    if not mode:
        return iet, {}
    elif callable(mode):
        key = mode
    else:
        # Default
        key = lambda f: f.is_DiscreteFunction or f.is_Array

    iet, headers = linearize_accesses(iet, key, cache, sregistry)
    iet = linearize_pointers(iet)
    iet = linearize_transfers(iet, sregistry)

    return iet, {'headers': headers}


def linearize_accesses(iet, key, cache, sregistry):
    """
    Turn Indexeds into FIndexeds and create the necessary access Macros.
    """
    # Find all objects amenable to linearization
    symbol_names = {i.name for i in FindSymbols('indexeds').visit(iet)}
    functions = [f for f in FindSymbols().visit(iet)
                 if key(f) and f.ndim > 1 and f.name in symbol_names]
    functions = sorted(functions, key=lambda f: len(f.dimensions), reverse=True)

    # Find unique sizes (unique -> minimize necessary registers)
    mapper = DefaultOrderedDict(list)
    for f in functions:
        if f not in cache:
            # NOTE: the outermost dimension is unnecessary
            for d in f.dimensions[1:]:
                # TODO: same grid + same halo => same padding, however this is
                # never asserted throughout the compiler yet... maybe should do
                # it when in debug mode at `prepare_arguments` time, ie right
                # before jumping to C?
                mapper[(d, f._size_halo[d], getattr(f, 'grid', None))].append(f)

    # Build all exprs such as `x_fsz0 = u_vec->size[1]`
    imapper = DefaultOrderedDict(list)
    for (d, halo, _), v in mapper.items():
        expr = _generate_fsz(v[0], d, sregistry)
        if expr:
            for f in v:
                imapper[f].append((d, expr.write))
                cache[f].stmts0.append(expr)

    # Build all exprs such as `y_slc0 = y_fsz0*z_fsz0`
    built = {}
    mapper = DefaultOrderedDict(list)
    for f, v in imapper.items():
        for n, (d, _) in enumerate(v):
            expr = prod(list(zip(*v[n:]))[1])
            try:
                stmt = built[expr]
            except KeyError:
                name = sregistry.make_name(prefix='%s_slc' % d.name)
                s = Symbol(name=name, dtype=np.uint32, is_const=True)
                stmt = built[expr] = DummyExpr(s, expr, init=True)
            mapper[f].append(stmt.write)
            cache[f].stmts1.append(stmt)
    mapper.update([(f, []) for f in functions if f not in mapper])

    # Build defines. For example:
    # `define uL(t, x, y, z) u[(t)*t_slc0 + (x)*x_slc0 + (y)*y_slc0 + (z)]`
    headers = []
    findexeds = {}
    for f, szs in mapper.items():
        if cache[f].cbk is not None:
            # Perhaps we've already built an access macro for `f` through another efunc
            findexeds[f] = cache[f].cbk
        else:
            header, cbk = _generate_macro(f, szs, sregistry)
            headers.append(header)
            cache[f].cbk = findexeds[f] = cbk

    # Build "functional" Indexeds. For example:
    # `u[t2, x+8, y+9, z+7] => uL(t2, x+8, y+9, z+7)`
    mapper = {}
    for n in FindNodes(Expression).visit(iet):
        subs = {}
        for i in retrieve_indexed(n.expr):
            try:
                subs[i] = findexeds[i.function](i)
            except KeyError:
                pass
        mapper[n] = n._rebuild(expr=uxreplace(n.expr, subs))

    # Put together all of the necessary exprs for `y_fsz0`, ..., `y_slc0`, ...
    stmts0 = filter_ordered(flatten(cache[f].stmts0 for f in functions))
    if stmts0:
        stmts0.append(BlankLine)
    stmts1 = filter_ordered(flatten(cache[f].stmts1 for f in functions))
    if stmts1:
        stmts1.append(BlankLine)

    iet = Transformer(mapper).visit(iet)
    body = iet.body._rebuild(body=tuple(stmts0) + tuple(stmts1) + iet.body.body)
    iet = iet._rebuild(body=body)

    return iet, headers


@singledispatch
def _generate_fsz(f, d, sregistry):
    return


@_generate_fsz.register(DiscreteFunction)
def _(f, d, sregistry):
    name = sregistry.make_name(prefix='%s_fsz' % d.name)
    s = Symbol(name=name, dtype=np.uint32, is_const=True)
    return DummyExpr(s, f._C_get_field(FULL, d).size, init=True)


@_generate_fsz.register(Array)
def _(f, d, sregistry):
    name = sregistry.make_name(prefix='%s_fsz' % d.name)
    s = Symbol(name=name, dtype=np.uint32, is_const=True)
    return DummyExpr(s, f.symbolic_shape[d], init=True)


@singledispatch
def _generate_macro(f, szs, sregistry):
    return


@_generate_macro.register(DiscreteFunction)
@_generate_macro.register(Array)
def _(f, szs, sregistry):
    assert len(szs) == len(f.dimensions) - 1

    pname = sregistry.make_name(prefix='%sL' % f.name)
    cbk = lambda i, pname=pname: FIndexed(i, pname)

    expr = sum([MacroArgument(d.name)*s for d, s in zip(f.dimensions, szs)])
    expr += MacroArgument(f.dimensions[-1].name)
    expr = Indexed(IndexedData(f.name, None, f), expr)
    define = DefFunction(pname, f.dimensions)
    header = (ccode(define), ccode(expr))

    return header, cbk


def linearize_pointers(iet):
    """
    Flatten n-dimensional PointerCasts/Dereferences.
    """
    indexeds = [i for i in FindSymbols('indexeds').visit(iet)]
    candidates = {i.function for i in indexeds if isinstance(i, FIndexed)}

    mapper = {}

    # Linearize casts, e.g. `float *u = (float*) u_vec->data`
    mapper.update({n: n._rebuild(flat=n.function.name)
                   for n in FindNodes(PointerCast).visit(iet)
                   if n.function in candidates})

    # Linearize array dereferences, e.g. `float *r1 = (float*) pr1[tid]`
    mapper.update({n: n._rebuild(flat=n.pointee.name)
                   for n in FindNodes(Dereference).visit(iet)
                   if n.pointer.is_PointerArray and n.pointee in candidates})

    iet = Transformer(mapper).visit(iet)

    return iet


def linearize_transfers(iet, sregistry):
    casts = FindNodes(PointerCast).visit(iet)
    candidates = {i.function for i in casts if i.flat is not None}

    mapper = {}
    for n in FindNodes(PragmaTransfer).visit(iet):
        if n.function not in candidates:
            continue

        try:
            imask0 = n.kwargs['imask']
        except KeyError:
            imask0 = []

        try:
            index = imask0.index(FULL)
        except ValueError:
            index = len(imask0)

        # Drop entries being flatten
        imask = imask0[:index]

        # The NVC 21.2 compiler (as well as all previous and potentially some
        # future versions as well) suffers from a bug in the parsing of pragmas
        # using subarrays in data clauses. For example, the following pragma
        # excerpt `... copyin(a[0]:b[0])` leads to a compiler error, despite
        # being perfectly legal OpenACC code. The workaround consists of
        # generating `const int ofs = a[0]; ... copyin(n:b[0])`
        exprs = []
        if len(imask) < len(imask0) and len(imask) > 0:
            assert len(imask) == 1
            try:
                start, size = imask[0]
            except TypeError:
                start, size = imask[0], 1

            if start != 0:  # Spare the ugly generated code if unneccesary (occurs often)
                name = sregistry.make_name(prefix='%s_ofs' % n.function.name)
                wildcard = Wildcard(name=name, dtype=np.int32, is_const=True)

                symsect = PragmaLangBB._make_symbolic_sections_from_imask(n.function,
                                                                          imask)
                assert len(symsect) == 1
                start, _ = symsect[0]
                exprs.append(DummyExpr(wildcard, start, init=True))

                imask = [(wildcard, size)]

        rebuilt = n._rebuild(imask=imask)

        if exprs:
            mapper[n] = List(body=exprs + [rebuilt])
        else:
            mapper[n] = rebuilt

    iet = Transformer(mapper).visit(iet)

    return iet

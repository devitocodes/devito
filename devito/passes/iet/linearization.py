from functools import singledispatch

import numpy as np

from devito.data import FULL
from devito.ir import (BlankLine, Call, DummyExpr, Dereference, List, PointerCast,
                       Transfer, FindNodes, FindSymbols, Transformer, Uxreplace)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import DefFunction, MacroArgument, ccode
from devito.tools import Bunch, DefaultOrderedDict, filter_ordered, prod
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
    track = DefaultOrderedDict(lambda: Bunch(stmts0=[], stmts1=[], held=set(), cbk=None))

    linearization(graph, track=track, **kwargs)

    # Sanity check
    assert all(not v.held for v in track.values())


@iet_pass
def linearization(iet, **kwargs):
    """
    Carry out the actual work of `linearize`.
    """
    mode = kwargs['mode']
    sregistry = kwargs['sregistry']
    track = kwargs['track']

    # Pre-process the `mode` opt option
    # `mode` may be a callback describing what Function types, and under what
    # conditions, should linearization be applied
    if not mode:
        return iet, {}
    elif callable(mode):
        key = lambda f: mode(f) and f.ndim > 1
    else:
        # Default
        key = lambda f: (f.is_DiscreteFunction or f.is_Array) and f.ndim > 1

    iet, headers = linearize_accesses(iet, key, track, sregistry)
    iet = linearize_pointers(iet, key)
    iet = linearize_transfers(iet, sregistry)

    return iet, {'headers': headers}


def linearize_accesses(iet, key, track, sregistry):
    """
    Turn Indexeds into FIndexeds and create the necessary access Macros.
    """
    # The `candidates` are all Functions that may be linearized inside `iet`
    indexeds = FindSymbols('indexeds').visit(iet)
    candidates = filter_ordered(i.function for i in indexeds if key(i.function))
    candidates = sorted(candidates, key=lambda f: len(f.dimensions), reverse=True)

    # For some of these candidates, a linearization may have already been
    # produced in another IET
    selected = [f for f in candidates if f not in track]

    # Find unique sizes (unique -> minimize necessary registers)
    mapper = DefaultOrderedDict(list)
    for f in selected:
        # NOTE: the outermost dimension is unnecessary
        for d in f.dimensions[1:]:
            # TODO: same grid + same halo => same padding, however this is
            # never asserted throughout the compiler yet... maybe should do
            # it when in debug mode at `prepare_arguments` time, ie right
            # before jumping to C?
            mapper[(d, f._size_halo[d], getattr(f, 'grid', None))].append(f)

    # For all unseen Functions, build the size exprs. For example:
    # `x_fsz0 = u_vec->size[1]`
    imapper = DefaultOrderedDict(dict)
    for (d, halo, _), v in mapper.items():
        expr = _generate_fsz(v[0], d, sregistry)
        if expr:
            for f in v:
                imapper[f][d] = expr.write
                track[f].stmts0.append(expr)

    # For all unseen Functions, build the stride exprs. For example:
    # `y_stride0 = y_fsz0*z_fsz0`
    built = {}
    mapper = DefaultOrderedDict(dict)
    for f, v in imapper.items():
        for d in v:
            n = f.dimensions.index(d)
            expr = prod(v[i] for i in f.dimensions[n:])
            try:
                stmt = built[expr]
            except KeyError:
                name = sregistry.make_name(prefix='%s_stride' % d.name)
                s = Symbol(name=name, dtype=np.int64, is_const=True)
                stmt = built[expr] = DummyExpr(s, expr, init=True)
            mapper[f][d] = stmt.write
            track[f].stmts1.append(stmt)

    # For all unseen Functions, build the access macros. For example:
    # `#define uL(t, x, y, z) u[(t)*t_stride0 + (x)*x_stride0 + (y)*y_stride0 + (z)]`
    headers = []
    for f in selected:
        if track[f].cbk is None:
            header, track[f].cbk = _generate_macro(f, mapper[f], sregistry)
            headers.append(header)

    # Turn all Indexeds into "functional" Indexeds. For example:
    # `u[t2, x+8, y+9, z+7] => uL(t2, x+8, y+9, z+7)`
    mapper = {i: track[i.function].cbk(i) for i in indexeds if i.function in track}
    iet = Uxreplace(mapper).visit(iet)

    # All Functions that can actually be linearized in `iet`
    defines = FindSymbols('defines-aliases').visit(iet)

    # All Callables for which `iet` may produce a linearization
    calls = {i.name for i in FindNodes(Call).visit(iet)}

    # Place the linearization expressions or delegate to ancestor efunc
    stmts0 = []
    stmts1 = []
    for f, v in track.items():
        release = calls & v.held
        v.held.difference_update(release)
        if f in candidates or release:
            if f in defines:
                stmts0.extend(v.stmts0)
                stmts1.extend(v.stmts1)
            else:
                v.held.add(iet.name)
    if stmts0:
        assert len(stmts1) > 0
        stmts0 = filter_ordered(stmts0) + [BlankLine]
        stmts1 = filter_ordered(stmts1) + [BlankLine]
        body = iet.body._rebuild(body=tuple(stmts0) + tuple(stmts1) + iet.body.body)
        iet = iet._rebuild(body=body)
    else:
        assert len(stmts1) == 0

    return iet, headers


@singledispatch
def _generate_fsz(f, d, sregistry):
    return


@_generate_fsz.register(DiscreteFunction)
def _(f, d, sregistry):
    name = sregistry.make_name(prefix='%s_fsz' % d.name)
    s = Symbol(name=name, dtype=np.int64, is_const=True)
    return DummyExpr(s, f._C_get_field(FULL, d).size, init=True)


@_generate_fsz.register(Array)
def _(f, d, sregistry):
    name = sregistry.make_name(prefix='%s_fsz' % d.name)
    s = Symbol(name=name, dtype=np.int64, is_const=True)
    return DummyExpr(s, f.symbolic_shape[d], init=True)


@singledispatch
def _generate_macro(f, szs, sregistry):
    return


@_generate_macro.register(DiscreteFunction)
@_generate_macro.register(Array)
def _(f, szs, sregistry):
    assert len(szs) == len(f.dimensions) - 1

    pname = sregistry.make_name(prefix='%sL' % f.name)
    cbk = lambda i, pname=pname: FIndexed(i, pname, strides=tuple(szs.values()))

    expr = sum([MacroArgument(d0.name)*szs[d1]
                for d0, d1 in zip(f.dimensions, f.dimensions[1:])])
    expr += MacroArgument(f.dimensions[-1].name)
    expr = Indexed(IndexedData(f.name, None, f), expr)
    define = DefFunction(pname, f.dimensions)
    header = (ccode(define), ccode(expr))

    return header, cbk


def linearize_pointers(iet, key):
    """
    Flatten n-dimensional PointerCasts/Dereferences.
    """
    candidates = {f for f in FindSymbols().visit(iet) if key(f)}

    mapper = {}

    # Linearize casts, e.g. `float *u = (float*) u_vec->data`
    mapper.update({n: n._rebuild(flat=True)
                   for n in FindNodes(PointerCast).visit(iet)
                   if n.function in candidates})

    # Linearize array dereferences, e.g. `float *r1 = (float*) pr1[tid]`
    mapper.update({n: n._rebuild(flat=True)
                   for n in FindNodes(Dereference).visit(iet)
                   if n.pointer.is_PointerArray and n.pointee in candidates})

    iet = Transformer(mapper).visit(iet)

    return iet


def linearize_transfers(iet, sregistry):
    casts = FindNodes(PointerCast).visit(iet)
    candidates = {i.function for i in casts if i.flat is not None}

    mapper = {}
    for n in FindNodes(Transfer).visit(iet):
        if n.function not in candidates:
            continue

        imask0 = n.imask or []

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

                symsect = n._rebuild(imask=imask).sections
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

from collections import defaultdict

from functools import singledispatch

import numpy as np
from sympy import Add

from devito.data import FULL
from devito.ir import (BlankLine, Call, DummyExpr, Dereference, List, PointerCast,
                       Transfer, FindNodes, FindSymbols, Transformer, Uxreplace)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import DefFunction, MacroArgument, ccode
from devito.tools import Bunch, filter_ordered, prod
from devito.types import Array, Symbol, FIndexed, Indexed, Wildcard
from devito.types.basic import IndexedData
from devito.types.dense import DiscreteFunction


__all__ = ['linearize']


def linearize(graph, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j] -> a[i*n + j]`. The row-major format
    of the underlying Function objects is honored.
    """
    mode = kwargs['mode']
    if not mode:
        return

    # All information that need to be propagated across callables
    tracker = Bunch(sizes={}, strides={}, undef=defaultdict(set), cbks={})

    linearization(graph, tracker=tracker, **kwargs)

    # Sanity check
    assert not tracker.undef


@iet_pass
def linearization(iet, mode=None, tracker=None, sregistry=None, **kwargs):
    """
    Carry out the actual work of `linearize`.
    """
    # `mode` may be a callback describing what Function types, and under what
    # conditions, should linearization be applied
    if callable(mode):
        key0 = lambda f: mode(f) and f.ndim > 1
    else:
        # Default
        key0 = lambda f: (f.is_DiscreteFunction or f.is_Array) and f.ndim > 1
    key1 = lambda f: not f._mem_stack
    key = lambda f: key0(f) and key1(f)

    iet, headers = linearize_accesses(iet, key, tracker, sregistry)
    iet = linearize_pointers(iet, key)
    iet = linearize_transfers(iet, sregistry)

    return iet, {'headers': headers}


def key1(f, d):
    """
    A key for Function dimensions. The key is:

        * False if not statically linearizable, that is not linearizable via
          constant symbolic sizes and strides;
        * A 3-tuple `(Dimension, halo size, grid)` otherwise.
    """
    if f.is_regular and f.is_compact:
        # TODO: same grid + same halo => same padding, however this is not asserted
        # during compilation... so maybe we should do it at `prepare_args` time?
        return (d, f._size_halo[d], getattr(f, 'grid', None))
    else:
        return False


def linearize_accesses(iet, key0, tracker, sregistry):
    """
    Turn Indexeds into FIndexeds and create the necessary access Macros.
    """
    # 1) What `iet` *needs*
    indexeds = FindSymbols('indexeds').visit(iet)
    needs = filter_ordered(i.function for i in indexeds if key0(i.function))
    needs = sorted(needs, key=lambda f: len(f.dimensions), reverse=True)

    # Update unique sizes table
    # E.g. `{(x, 8, grid) -> x_fsz0}`
    for f in needs:
        # NOTE: the outermost dimension is unnecessary
        for d in f.dimensions[1:]:
            k = key1(f, d)
            if k and k not in tracker.sizes:
                name = sregistry.make_name(prefix='%s_fsz' % d.name)
                fsz = Symbol(name=name, dtype=np.int64, is_const=True)
                tracker.sizes[k] = fsz

    # Update unique strides table
    # E.g., `{x_stride0 -> (y_fsz0, z_fsz0)}`
    mapper = defaultdict(dict)
    for f in needs:
        for n, d in enumerate(f.dimensions[1:], 1):
            try:
                k = tuple(tracker.sizes[key1(f, d1)] for d1 in f.dimensions[n:])
            except KeyError:
                continue
            if k not in tracker.strides:
                name = sregistry.make_name(prefix='%s_stride' % d.name)
                stride = Symbol(name=name, dtype=np.int64, is_const=True)
                tracker.strides[k] = stride
            mapper[f][d] = tracker.strides[k]

    # Update unique access macros
    # E.g. `{f(x, y) -> foo}`, `foo(Indexed) -> f[(x)*y_stride0 + (y)]`
    headers = []
    for f in needs:
        if f in tracker.cbks:
            continue
        header, tracker.cbks[f] = _generate_macro(f, mapper[f], sregistry)
        headers.append(header)

    # Turn Indexeds into FIndexeds
    # E.g. `u[t2, x+8, y+9, z+7] -> uL(t2, x+8, y+9, z+7)`
    subs = {i: tracker.cbks[i.function](i) for i in indexeds if i.function in needs}
    iet = Uxreplace(subs).visit(iet)

    # 2) What `iet` *offers*
    # E.g. `{x_fsz0 -> u_vec->size[1]}`
    defines = FindSymbols('defines-aliases').visit(iet)
    offers = filter_ordered(f for f in defines if key0(f))
    instances = {}
    for f in offers:
        for d in f.dimensions[1:]:
            k = key1(f, d)
            try:
                fsz = tracker.sizes[k]
                v = _generate_fsz(f, d, fsz, sregistry)
            except KeyError:
                v = None
            if v is not None:
                instances[fsz] = v

    # 3) What we need to construct are `iet`'s needs plus the callee delegations
    candidates = set().union(*[v.values() for v in mapper.values()])
    for n in FindNodes(Call).visit(iet):
        candidates.update(tracker.undef.pop(n.name, []))

    # 4) Construct what needs to *and* can be constructed
    mapper = dict(zip(tracker.strides.values(), tracker.strides.keys()))
    stmts0, stmts1 = [], []
    for stride, sizes in mapper.items():
        if stride not in candidates:
            continue
        try:
            stmts0.extend([instances[size] for size in sizes])
        except KeyError:
            # `stride` would be needed by `iet`, but we don't have the means
            # to define it, hence we delegate to the caller
            tracker.undef[iet.name].add(stride)
            continue
        stmts1.append(DummyExpr(stride, prod(sizes), init=True))

    # 5) Attach `stmts0` and `stmts1` to `iet`
    if stmts0:
        assert len(stmts1) > 0
        stmts = filter_ordered(stmts0) + [BlankLine] + stmts1 + [BlankLine]
        body = iet.body._rebuild(body=tuple(stmts) + iet.body.body)
        iet = iet._rebuild(body=body)
    else:
        assert len(stmts1) == 0

    return iet, headers


@singledispatch
def _generate_fsz(f, d, fsz, sregistry):
    return


@_generate_fsz.register(DiscreteFunction)
def _(f, d, fsz, sregistry):
    return DummyExpr(fsz, f._C_get_field(FULL, d).size, init=True)


@_generate_fsz.register(Array)
def _(f, d, fsz, sregistry):
    return DummyExpr(fsz, f.symbolic_shape[d], init=True)


@singledispatch
def _generate_macro(f, strides, sregistry):
    return None, None


@_generate_macro.register(DiscreteFunction)
@_generate_macro.register(Array)
def _(f, strides, sregistry):
    pname = sregistry.make_name(prefix='%sL' % f.name)

    # Generate e.g. `usave[(time)*xi_slc0 + (xi)*yi_slc0 + (yi)]`
    assert len(strides) == len(f.dimensions) - 1
    macroargnames = [d.name for d in f.dimensions]
    macroargs = [MacroArgument(i) for i in macroargnames]

    items = [m*strides[d] for m, d in zip(macroargs, f.dimensions[1:])]
    items.append(MacroArgument(f.dimensions[-1].name))

    value = Add(*items, evaluate=False)
    expr = Indexed(IndexedData(f.name, None, f), value)
    define = DefFunction(pname, macroargnames)
    header = (ccode(define), ccode(expr))

    def cbk(i):
        return FIndexed(i, pname, strides=tuple(strides.values()))

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

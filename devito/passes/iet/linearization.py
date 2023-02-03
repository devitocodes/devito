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
from devito.types import Array, Bundle, Symbol, FIndexed, Indexed, Wildcard
from devito.types.basic import IndexedData
from devito.types.dense import DiscreteFunction


__all__ = ['linearize']


def linearize(graph, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j] -> a[i*n + j]`. The row-major format
    of the underlying Function objects is honored.
    """
    options = kwargs.get('options', {})
    lmode = kwargs.pop('lmode', options.get('linearize'))
    if not lmode:
        return

    # All information that need to be propagated across callables
    tracker = Bunch(
        sizes={},
        strides={},
        undef=defaultdict(set),
        subs={},
        headers={}
    )

    linearization(graph, lmode=lmode, tracker=tracker, **kwargs)

    # Sanity check
    assert not tracker.undef


@iet_pass
def linearization(iet, lmode=None, tracker=None, **kwargs):
    """
    Carry out the actual work of `linearize`.
    """
    # `lmode` may be a callback describing what Function types, and under what
    # conditions, should linearization be applied
    if callable(lmode):
        key0 = lambda f: lmode(f) and f.ndim > 1
    else:
        # Default
        key0 = lambda f: f.is_AbstractFunction and f.ndim > 1
    key = lambda f: key0(f) and not f._mem_stack

    iet = linearize_accesses(iet, key, tracker, **kwargs)
    iet = linearize_pointers(iet, key)
    iet = linearize_transfers(iet, **kwargs)

    # Postprocess headers
    headers = [(ccode(define), ccode(expr)) for define, expr in tracker.headers.values()]

    return iet, {'headers': headers}


def key1(f, d):
    """
    A key for Function dimensions. The key is:

        * False if not statically linearizable, that is not linearizable via
          constant symbolic sizes and strides;
        * A 3-tuple `(Dimension, halo size, grid)` otherwise.
    """
    if f.is_regular:
        # TODO: same grid + same halo => same padding, however this is not asserted
        # during compilation... so maybe we should do it at `prepare_args` time?
        return (d, f._size_halo[d], getattr(f, 'grid', None))
    else:
        return False


def linearize_accesses(iet, key0, tracker=None, sregistry=None, options=None,
                       **kwargs):
    """
    Turn Indexeds into FIndexeds and create the necessary access Macros.
    """
    if options['index-mode'] == 'int32':
        dtype = np.int32
    else:
        dtype = np.int64

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
                fsz = Symbol(name=name, dtype=dtype, is_const=True)
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
                stride = Symbol(name=name, dtype=dtype, is_const=True)
                tracker.strides[k] = stride
            mapper[f][d] = tracker.strides[k]

    # Update unique access macros
    # E.g. `{f(x, y) -> foo}`, `foo(Indexed) -> f[(x)*y_stride0 + (y)]`
    for f in needs:
        # All the Indexeds demanding an access macro. For a given `f`, more than
        # one access macros may be required -- in particular, it will be more
        # than one if `f` was optimized by the `split_pointers` pass
        v = [i for i in indexeds if i.function is f]

        _generate_macro(f, v, tracker, mapper[f], sregistry)

    # Turn Indexeds into FIndexeds
    # E.g. `u[t2, x+8, y+9, z+7] -> uL(t2, x+8, y+9, z+7)`
    iet = Uxreplace(tracker.subs).visit(iet)

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
                val = _generate_fsz(f, d)
                if val.free_symbols.issubset(f.bound_symbols):
                    v = DummyExpr(fsz, val, init=True)
                else:
                    v = None
            except (AttributeError, KeyError):
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

    return iet


@singledispatch
def _generate_fsz(f, d):
    return


@_generate_fsz.register(DiscreteFunction)
def _(f, d):
    return f._C_get_field(FULL, d).size


@_generate_fsz.register(Array)
def _(f, d):
    return f.symbolic_shape[d]


@_generate_fsz.register(Bundle)
def _(f, d):
    if f.is_DiscreteFunction:
        return _generate_fsz.registry[DiscreteFunction](f, d)
    else:
        return _generate_fsz.registry[Array](f, d)


@singledispatch
def _generate_macro(f, indexeds, tracker, strides, sregistry):
    pass


@_generate_macro.register(DiscreteFunction)
@_generate_macro.register(Array)
@_generate_macro.register(Bundle)
def _(f, indexeds, tracker, strides, sregistry):
    # Generate e.g. `usave[(time)*xi_slc0 + (xi)*yi_slc0 + (yi)]`
    assert len(strides) == len(f.dimensions) - 1
    macroargnames = [d.name for d in f.dimensions]
    macroargs = [MacroArgument(i) for i in macroargnames]

    items = [m*strides[d] for m, d in zip(macroargs, f.dimensions[1:])]
    items.append(MacroArgument(f.dimensions[-1].name))

    headers = tracker.headers
    subs = tracker.subs
    for i in indexeds:
        if i in subs:
            continue

        n = len(i.indices)
        if n == 1:
            continue

        if i.base not in headers:
            pname = sregistry.make_name(prefix='%sL' % f.name)

            value = Add(*items[-n:], evaluate=False)
            expr = Indexed(IndexedData(i.base, None, f), value)
            define = DefFunction(pname, macroargnames[-n:])

            headers[i.base] = (define, expr)
        else:
            define, _ = headers[i.base]
            pname = str(define.name)

        if len(i.indices) == i.function.ndim:
            v = tuple(strides.values())[-n:]
            subs[i] = FIndexed(i, pname, strides=v)
        else:
            # Honour custom indexing
            subs[i] = i.base[sum(i.indices)]


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


def linearize_transfers(iet, sregistry=None, **kwargs):
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

from collections import defaultdict

from functools import singledispatch

import numpy as np
from sympy import Add

from devito.data import FULL
from devito.ir import (BlankLine, Call, DummyExpr, Dereference, List, PointerCast,
                       Transfer, FindNodes, FindSymbols, Transformer, Uxreplace,
                       IMask)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.parpragma import PragmaIteration
from devito.symbolics import DefFunction, MacroArgument, ccode
from devito.tools import Bunch, filter_ordered, flatten, prod
from devito.types import Array, Bundle, Symbol, FIndexed, Indexed, Wildcard
from devito.types.dense import DiscreteFunction

__all__ = ['linearize']


def linearize(graph, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j] -> a[i*n + j]`. The row-major format
    of the underlying Function objects is honored.
    """
    sregistry = kwargs.get('sregistry')
    options = kwargs.get('options', {})

    mode = options.get('linearize')
    maybe_callback = kwargs.pop('callback', mode)

    if not maybe_callback:
        return
    elif callable(maybe_callback):
        key = lambda f: maybe_callback(f) and f.ndim > 1 and not f._mem_stack
    else:
        key = lambda f: f.is_AbstractFunction and f.ndim > 1 and not f._mem_stack

    if options['index-mode'] == 'int32':
        dtype = np.int32
    else:
        dtype = np.int64

    # NOTE: Even if `mode=False`, `key` may still want to enforce linearization
    # of some Functions, so it takes precedence and we then attempt to linearize
    if not mode or isinstance(mode, bool):
        mode = 'basic'

    tracker = Tracker(mode, dtype, sregistry)

    linearization(graph, key=key, tracker=tracker, **kwargs)

    # Sanity check
    assert not tracker.undef


@iet_pass
def linearization(iet, key=None, tracker=None, **kwargs):
    """
    Carry out the actual work of `linearize`.
    """
    iet = linearize_accesses(iet, key, tracker)
    iet = linearize_pointers(iet, key)
    iet = linearize_transfers(iet, **kwargs)
    iet = linearize_clauses(iet, **kwargs)

    # Postprocess headers
    headers = sorted((ccode(i), ccode(e)) for i, e in tracker.headers.values())

    return iet, {'headers': headers}


def key1(f, d):
    """
    A key for Function dimensions. The key is:

        * False if not statically linearizable, that is not linearizable via
          constant symbolic sizes and strides;
        * A 3-tuple `(Dimension, halo size, grid)` otherwise.
    """
    if f.is_regular:
        # For paddable objects the following holds:
        # `same dim + same halo => same (auto-)padding`
        return (d, f._size_halo[d], f.is_autopaddable)
    else:
        return False


class Size(Symbol):

    """
    Symbol representing the size of a Function dimension.
    """

    pass


class Stride(Symbol):

    """
    Symbol representing the stride of a Function dimension.

    The stride is the the number of elements to skip over in the array's linear
    memory layout to move one step along that dimension. For example, consider
    a 2D array with rows and columns. The stride for the row dimension tells
    you how many elements you need to move forward in memory to get from one
    row to the next. Similarly, the stride for the column dimension tells you
    the number of elements you need to skip to move from one column to the next
    in memory. The stride, therefore, is the product of the size of the
    dimensions that come after the dimension in question.
    """

    pass


class Dist(Symbol):

    """
    Symbol representing the distance between a reference point and another point.
    """

    pass


class Tracker:

    def __init__(self, mode, dtype, sregistry):
        self.mode = mode
        self.dtype = dtype
        self.sregistry = sregistry

        self.sizes = {}
        self.strides = {}
        self.strides_dynamic = {}  # Strides varying regularly across iterations
        self.dists = {}
        self.undef = defaultdict(set)
        self.headers = {}

    def add(self, f):
        # Update unique sizes table
        for d in f.dimensions[1:]:
            k = key1(f, d)
            if not k or k in self.sizes:
                continue
            name = self.sregistry.make_name(prefix='%s_fsz' % d.name)
            self.sizes[k] = Size(name=name, dtype=self.dtype, is_const=True)

        # Update unique strides table
        for n, d in enumerate(f.dimensions[1:], 1):
            try:
                k = tuple(self.get_size(f, d1) for d1 in f.dimensions[n:])
            except KeyError:
                continue
            if k in self.strides:
                continue
            name = self.sregistry.make_name(prefix='%s_stride' % d.name)
            self.strides[k] = Stride(name=name, dtype=self.dtype, is_const=True)

    def update(self, functions):
        for f in functions:
            self.add(f)

    def add_strides_dynamic(self, f, strides):
        assert not f.is_regular
        assert len(strides) == f.ndim - 1
        self.strides_dynamic[f] = tuple(strides)

    def add_header(self, b, define, expr):
        # NOTE: the input must be an IndexedBase and not a Function because
        # we might require either the `.base` or the `.dmap`, or perhaps both
        v = Bunch(define=define, expr=expr)
        self.headers[b] = v
        return v

    def needs_header(self, b):
        return b.function.ndim > 1 and b not in self.headers

    def get_header(self, b):
        return self.headers.get(b)

    def get_size(self, f, d):
        return self.sizes[key1(f, d)]

    def get_sizes(self, f):
        return tuple(self.get_size(f, d) for d in f.dimensions[1:])

    def map_strides(self, f):
        dims = list(f.dimensions[1:])
        if f.is_regular:
            sizes = self.get_sizes(f)
            return {d: self.strides[sizes[n:]] for n, d in enumerate(dims)}
        elif f in self.strides_dynamic:
            return {d: i for d, i in zip(dims, self.strides_dynamic[f])}
        else:
            return {}

    def make_dist(self, v):
        try:
            dist = self.dists[v]
        except KeyError:
            name = self.sregistry.make_name(prefix='dst')
            dist = self.dists[v] = Dist(name=name, dtype=self.dtype, is_const=True)
        return dist


def linearize_accesses(iet, key0, tracker=None):
    """
    Turn Indexeds into FIndexeds and create the necessary access Macros.
    """
    # 1) What `iet` *needs*
    indexeds = FindSymbols('indexeds').visit(iet)
    needs = filter_ordered(i.function for i in indexeds if key0(i.function))
    needs = sorted(needs, key=lambda f: len(f.dimensions), reverse=True)

    # Update unique sizes and strides
    tracker.update(needs)

    # Pick the chosen linearization mode
    generate_linearization = linearization_registry[tracker.mode]

    # Generate unique access macros
    # E.g. `{f(x, y) -> foo}`, `foo(Indexed) -> f[(x)*y_stride0 + (y)]`
    # And linearize Indexeds
    # E.g. `u[t2, x+8, y+9, z+7] -> uL(t2, x+8, y+9, z+7)`
    subs = {}
    for i in indexeds:
        f = i.function
        if f not in needs:
            continue

        v = generate_linearization(f, i, tracker)
        if v is not None:
            subs[i] = v

    iet = Uxreplace(subs).visit(iet)

    # 2) What `iet` *offers*
    # E.g. `{x_fsz0 -> u_vec->size[1]}`
    defines = FindSymbols('defines-aliases').visit(iet)
    offers = filter_ordered(f for f in defines if key0(f))
    instances = {}
    for f in offers:
        for d in f.dimensions[1:]:
            try:
                fsz = tracker.get_size(f, d)
            except KeyError:
                continue
            try:
                val = _generate_fsz(f, d)
                if val.free_symbols.issubset(f.bound_symbols):
                    instances[fsz] = DummyExpr(fsz, val, init=True)
            except AttributeError:
                pass

    # 3) Which symbols (Strides, Dists) does `iet` need?
    symbols = flatten(i.free_symbols for i in subs.values())
    candidates = {i for i in symbols if isinstance(i, (Stride, Dist))}
    for n in FindNodes(Call).visit(iet):
        # Also consider the strides needed by the callee
        candidates.update(tracker.undef.pop(n.name, []))

    # 4) What `strides` can indeed be constructed?
    mapper = {}
    for sizes, stride in tracker.strides.items():
        if stride in candidates:
            if set(sizes).issubset(instances):
                mapper[stride] = sizes
            else:
                tracker.undef[iet.name].add(stride)

    # 5) Construct what needs to *and* can be constructed
    stmts, stmts1 = [], []
    for stride, sizes in mapper.items():
        stmts.extend([instances[size] for size in sizes])
        stmts1.append(DummyExpr(stride, prod(sizes), init=True))
    stmts = filter_ordered(stmts)

    stmts2 = []
    for v, dist in tracker.dists.items():
        if dist in candidates:
            stmts2.append(DummyExpr(dist, v, init=True))

    for i in [stmts1, stmts2]:
        if i and stmts:
            stmts.append(BlankLine)
        stmts.extend(i)

    if stmts:
        body = iet.body._rebuild(strides=stmts)
        iet = iet._rebuild(body=body)

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
def _generate_header_basic(f, tracker):
    return


@_generate_header_basic.register(DiscreteFunction)
@_generate_header_basic.register(Array)
@_generate_header_basic.register(Bundle)
def _(f, i, tracker):
    b = i.base

    if not tracker.needs_header(b):
        return tracker.get_header(b)

    # Generate e.g. `usave[(time)*xi_slc0 + (xi)*yi_slc0 + (yi)]`

    strides = tracker.map_strides(f)

    macroargnames = [d.name for d in f.dimensions]
    macroargs = [MacroArgument(i) for i in macroargnames]

    items = [m*strides[d] for m, d in zip(macroargs, f.dimensions[1:])]
    items.append(MacroArgument(f.dimensions[-1].name))

    pname = tracker.sregistry.make_name(prefix='%sL' % f.name)
    define = DefFunction(pname, macroargnames)
    expr = Indexed(b, Add(*items, evaluate=False))

    return tracker.add_header(b, define, expr)


@singledispatch
def _generate_linearization_basic(f, i, tracker):
    assert False


@_generate_linearization_basic.register(DiscreteFunction)
@_generate_linearization_basic.register(Array)
@_generate_linearization_basic.register(Bundle)
def _(f, i, tracker):
    header = _generate_header_basic(f, i, tracker)

    n = len(i.indices)

    if header and n == i.function.ndim:
        pname = header.define.name.value
        strides = tuple(tracker.map_strides(f).values())[-n:]
        return FIndexed.from_indexed(i, pname, strides=strides)
    else:
        # Honour custom indexing
        return i.base[sum(i.indices)]


linearization_registry = {
    'basic': _generate_linearization_basic,
}


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


def linearize_clauses(iet, **kwargs):
    iters = FindNodes(PragmaIteration).visit(iet)
    mapper = {}
    for i in iters:
        # Linearize reduction clauses, e.g.:
        # `reduction(+:f[0:f_vec->size[1][0:f_vec->size[2]]])`
        # ->
        # `reduction(+:f[0:f_vec->size[1]*f_vec->size[2]])
        if not i.reduction:
            continue
        reductions = []
        for output, imask, op in i.reduction:
            f = output.function

            # Support partial reductions
            try:
                idx = imask.index(FULL)
            except ValueError:
                idx = len(imask)

            m = np.prod(imask[:idx] or [0])
            size = prod([f._C_get_field(FULL, d).size for d in f.dimensions[idx:]])

            reductions.append((output, IMask((m*size, size)), op))

        mapper[i] = i._rebuild(reduction=reductions)

    iet = Transformer(mapper).visit(iet)

    return iet

from functools import singledispatch

import cgen
import numpy as np
import sympy

from devito.finite_differences import Max, Min
from devito.finite_differences.differentiable import SafeInv
from devito.logger import warning
from devito.ir import (Any, Forward, DummyExpr, Iteration, EmptyList, Prodder,
                       FindApplications, FindNodes, FindSymbols, Transformer,
                       Uxreplace, filter_iterations, retrieve_iteration_tree,
                       pull_dims)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.languages.C import CPrinter
from devito.ir.iet.efunc import DeviceFunction, EntryFunction
from devito.symbolics import (ValueLimit, evalrel, has_integer_args, limits_mapper, Cast)
from devito.tools import Bunch, as_mapper, filter_ordered, split, as_tuple
from devito.types import FIndexed

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions',
           'generate_macros', 'minimize_symbols']


@iet_pass
def avoid_denormals(iet, platform=None, **kwargs):
    """
    Introduce nodes in the Iteration/Expression tree that will expand to C
    macros telling the CPU to flush denormal numbers in hardware. Denormals
    are normally flushed when using SSE-based instruction sets, except when
    compiling shared objects.
    """
    # There is unfortunately no known portable way of flushing denormal to zero.
    # See for example: https://stackoverflow.com/questions/59546406/\
    #                       a-robust-portable-way-to-set-flush-denormals-to-zero
    try:
        if 'sse' not in platform.known_isas:
            return iet, {}
    except AttributeError:
        return iet, {}

    if iet.is_ElementalFunction:
        return iet, {}

    header = (cgen.Comment('Flush denormal numbers to zero in hardware'),
              cgen.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)'),
              cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'),
              cgen.Line())

    body = iet.body._rebuild(body=(EmptyList(header=header),) + iet.body.body)
    iet = iet._rebuild(body=body)

    return iet, {'includes': ('xmmintrin.h', 'pmmintrin.h')}


@iet_pass
def hoist_prodders(iet):
    """
    Move Prodders within the outer levels of an Iteration tree.
    """
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        for prodder in FindNodes(Prodder).visit(tree.root):
            if prodder._periodic:
                try:
                    key = lambda i: i.dim.is_Block and i.dim.step != 1
                    candidate = filter_iterations(tree, key)[-1]
                except IndexError:
                    # Fallback: use the outermost Iteration
                    candidate = tree.root
                mapper[candidate] = candidate._rebuild(nodes=(candidate.nodes +
                                                              (prodder._rebuild(),)))
                mapper[prodder] = None

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}


@iet_pass
def relax_incr_dimensions(iet, options=None, **kwargs):
    """
    This pass adjusts the bounds of blocked Iterations in order to include the "remainder
    regions".  Without the relaxation that occurs in this pass, the only way to iterate
    over the entire iteration space is to have step increments that are perfect divisors
    of the iteration space (e.g. in case of an iteration space of size 67 and block size
    8 only 64 iterations would be computed, as `67 - 67mod8 = 64`.

    A simple 1D example: nested Iterations are transformed from:

    <Iteration x0_blk0; (x_m, x_M, x0_blk0_size)>
        <Iteration x; (x0_blk0, x0_blk0 + x0_blk0_size - 1, 1)>

    to:

    <Iteration x0_blk0; (x_m, x_M, x0_blk0_size)>
        <Iteration x; (x0_blk0, Min(x_M, x0_blk0 + x0_blk0_size - 1)), 1)>

    """
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Block]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        assert all(i.direction in (Forward, Any) for i in iterations)
        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Block)

        # Get root's `symbolic_max` out of each outer Dimension
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # Process inner iterations and adjust their bounds
        for n, i in enumerate(inner):
            # If definitely in-bounds, as ensured by a prior compiler pass, then
            # we can skip this step
            if i.is_Inbound:
                continue

            # The Iteration's maximum is the Min of (a) the `symbolic_max` of current
            # Iteration e.g. `x0_blk0 + x0_blk0_size - 1` and (b) the `symbolic_max`
            # of the current Iteration's root Dimension e.g. `x_M`. The generated
            # maximum will be `Min(x0_blk0 + x0_blk0_size - 1, x_M)

            # In some corner cases an offset may be added (e.g. after CIRE passes)
            # E.g. assume `i.symbolic_max = x0_blk0 + x0_blk0_size + 1` and
            # `i.dim.symbolic_max = x0_blk0 + x0_blk0_size - 1` then the generated
            # maximum will be `Min(x0_blk0 + x0_blk0_size + 1, x_M + 2)`

            root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max
            iter_max = evalrel(min, [i.symbolic_max, root_max])
            mapper[i] = i._rebuild(limits=(i.symbolic_min, iter_max, i.step))

    if mapper:
        iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}


def generate_macros(graph, **kwargs):
    _generate_macros(graph, tracker={}, **kwargs)


@iet_pass
def _generate_macros(iet, tracker=None, langbb=None, printer=CPrinter, **kwargs):
    # Derive the Macros necessary for the FIndexeds
    iet = _generate_macros_findexeds(iet, tracker=tracker, **kwargs)

    # NOTE: sorting is necessary to ensure deterministic code generation
    headers = [i.header for i in tracker.values()]
    headers = sorted((printer()._print(define), printer()._print(expr))
                     for define, expr in headers)

    # Generate Macros from higher-level SymPy objects
    mheaders, includes = _generate_macros_math(iet, langbb=langbb)
    includes = sorted(includes, key=str)
    headers.extend(sorted(mheaders, key=str))

    # Remove redundancies while preserving the order
    headers = filter_ordered(headers)

    # Some special Symbols may represent Macros defined in standard libraries,
    # so we need to include the respective includes
    limits = FindApplications(ValueLimit).visit(iet)
    if limits & (set(limits_mapper[np.int32]) | set(limits_mapper[np.int64])):
        includes.append('limits.h')
    elif limits & (set(limits_mapper[np.float32]) | set(limits_mapper[np.float64])):
        includes.append('float.h')

    return iet, {'headers': headers, 'includes': includes}


def _generate_macros_findexeds(iet, sregistry=None, tracker=None, **kwargs):
    indexeds = FindSymbols('indexeds').visit(iet)
    findexeds = [i for i in indexeds if isinstance(i, FIndexed)]
    if not findexeds:
        return iet

    subs = {}
    for i in findexeds:
        try:
            v = tracker[i.base].v
            subs[i] = v.func(v.base, *i.indices)
            continue
        except KeyError:
            pass

        pname = sregistry.make_name(prefix='%sL' % i.name)
        header, v = i.bind(pname)

        subs[i] = v
        tracker[i.base] = Bunch(header=header, v=v)

    iet = Uxreplace(subs).visit(iet)

    return iet


def _generate_macros_math(iet, langbb=None):
    headers = []
    includes = []
    for i in FindApplications().visit(iet):
        header, include = _lower_macro_math(i, langbb)
        headers.extend(header)
        includes.extend(include)

    return headers, set(includes) - {None}


@singledispatch
def _lower_macro_math(expr, langbb):
    return (), {}


@_lower_macro_math.register(Min)
@_lower_macro_math.register(sympy.Min)
def _(expr, langbb):
    if has_integer_args(*expr.args):
        return (('MIN(a,b)', ('(((a) < (b)) ? (a) : (b))')),), {}
    else:
        return (), as_tuple(langbb.get('header-math'))


@_lower_macro_math.register(Max)
@_lower_macro_math.register(sympy.Max)
def _(expr, langbb):
    if has_integer_args(*expr.args):
        return (('MAX(a,b)', ('(((a) > (b)) ? (a) : (b))')),), {}
    else:
        return (), as_tuple(langbb.get('header-math'))


@_lower_macro_math.register(SafeInv)
def _(expr, langbb):
    try:
        eps = np.finfo(expr.base.dtype).resolution**2
    except ValueError:
        warning(f"dtype not recognized in SafeInv for {expr.base}, assuming float32")
        eps = np.finfo(np.float32).resolution**2
    b = Cast('b', dtype=np.float32)
    return (('SAFEINV(a, b)',
             f'(((a) < {eps}F || ({b}) < {eps}F) ? (0.0F) : ((1.0F) / (a)))'),), {}


@iet_pass
def minimize_symbols(iet):
    """
    Remove unneccesary symbols. Currently applied sub-passes:

        * Remove redundant ModuloDimensions (e.g., due to using the
          `save=Buffer(2)` API)
        * Simplify Iteration headers (e.g., ModuloDimensions with identical
          starting point and step)
        * Abridge SubDimension names where possible to declutter generated
          loop nests and shrink indices
    """
    iet = remove_redundant_moddims(iet)
    iet = simplify_iteration_headers(iet)
    iet = abridge_dim_names(iet)

    return iet, {}


def remove_redundant_moddims(iet):
    key = lambda d: d.is_Modulo and d.origin is not None
    mds = [d for d in FindSymbols('dimensions').visit(iet) if key(d)]
    if not mds:
        return iet

    degenerates, others = split(mds, lambda d: d.modulo == 1)
    subs = {d: sympy.S.Zero for d in degenerates}

    redundants = as_mapper(others, key=lambda d: d.offset % d.modulo)
    for k, v in redundants.items():
        chosen = v.pop(0)
        subs.update({d: chosen for d in v})

    # Transform the `body`, rather than `iet`, to avoid applying substitutions
    # to `iet.parameters`, so e.g. `..., t0, t1, t2, ...` remains unchanged
    # instead of becoming `..., t0, t1, t1, ...`. The IET `engine` will then
    # take care of cleaning up the `parameters` list
    body = Uxreplace(subs).visit(iet.body)
    iet = iet._rebuild(body=body)

    return iet


def simplify_iteration_headers(iet):
    mapper = {}
    for i in FindNodes(Iteration).visit(iet):
        candidates = [d for d in i.uindices
                      if d.is_Modulo and d.symbolic_min == d.symbolic_incr]

        # Don't touch `t0, t1, ...` for codegen aesthetics and to avoid
        # massive changes in the test suite
        candidates = [d for d in candidates
                      if not any(dd.is_Time for dd in d._defines)]

        if not candidates:
            continue

        uindices = [d for d in i.uindices if d not in candidates]
        stmts = [DummyExpr(d, d.symbolic_incr, init=True) for d in candidates]

        mapper[i] = i._rebuild(nodes=tuple(stmts) + i.nodes, uindices=uindices)

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


@singledispatch
def abridge_dim_names(iet):
    return iet


@abridge_dim_names.register(DeviceFunction)
def _(iet):
    # Catch SubDimensions not in EntryFunction
    mapper = _rename_subdims(iet, FindSymbols('dimensions').visit(iet))
    return Uxreplace(mapper, nested=True).visit(iet)


@abridge_dim_names.register(EntryFunction)
def _(iet):
    # SubDimensions in the main loop nests
    mapper = {}
    # Build a mapper replacing SubDimension names with respective root dimension
    # names where possible
    for tree in retrieve_iteration_tree(iet):
        # Rename SubDimensions present as indices in innermost loop
        mapper.update(_rename_subdims(tree.inner, tree.dimensions))

        # Update unbound index parents with renamed SubDimensions
        dims = set().union(*[i.uindices for i in tree])
        dims = [d for d in dims if d.is_Incr and d.parent in mapper]
        mapper.update({d: d._rebuild(parent=mapper[d.parent]) for d in dims})

        # Update parents of CIRE-generated ModuloDimensions
        dims = FindSymbols('dimensions').visit(tree)
        dims = [d for d in dims if d.is_Modulo and d.parent in mapper]
        mapper.update({d: d._rebuild(parent=mapper[d.parent]) for d in dims})

    return Uxreplace(mapper, nested=True).visit(iet)


def _rename_subdims(target, dimensions):
    # Find SubDimensions or SubDimension-derived dimensions used as indices in
    # the expression
    indexeds = FindSymbols('indexeds').visit(target)
    dims = pull_dims(indexeds, flag=False)
    dims = [d for d in dims if any(dim.is_AbstractSub for dim in d._defines)]
    dims = [d for d in dims if not d.is_SubIterator]
    names = [d.root.name for d in dims]

    # Rename them to use the name of their root dimension if this will not cause a
    # clash with Dimensions or other renamed SubDimensions
    return {d: d._rebuild(d.root.name) for d in dims
            if d.root not in dimensions
            and names.count(d.root.name) < 2}

from collections import defaultdict

import cgen
import numpy as np

from devito.data import FULL
from devito.ir import (BlankLine, DummyEq, Expression, Forward, List, LocalExpression,
                       Prodder, FindNodes, FindSymbols, Transformer, filter_iterations,
                       retrieve_iteration_tree)
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (MIN, MAX, DefFunction, MacroArgument, ccode,
                              retrieve_indexed, uxreplace)
from devito.tools import (Bunch, DefaultOrderedDict, filter_ordered, flatten,
                          is_integer, prod, split)
from devito.types import Symbol, FIndexed, Indexed
from devito.types.basic import IndexedData

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions', 'linearize',
           'is_on_device']


@iet_pass
def avoid_denormals(iet):
    """
    Introduce nodes in the Iteration/Expression tree that will expand to C
    macros telling the CPU to flush denormal numbers in hardware. Denormals
    are normally flushed when using SSE-based instruction sets, except when
    compiling shared objects.
    """
    if iet.is_ElementalFunction:
        return iet, {}

    header = (cgen.Comment('Flush denormal numbers to zero in hardware'),
              cgen.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)'),
              cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'),
              cgen.Line())

    body = iet.body._rebuild(body=(List(header=header),) + iet.body.body)
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
                    key = lambda i: i.dim.is_Incr and i.dim.step != 1
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
def relax_incr_dimensions(iet, **kwargs):
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
        <Iteration x; (x0_blk0, MIN(x_M, x0_blk0 + x0_blk0_size - 1)), 1)>

    """

    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Incr]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        assert all(i.direction is Forward for i in iterations)
        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Incr)

        # Get root's `symbolic_max` out of each outer Dimension
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # A dictionary to map maximum of processed parent dimensions. Helps to neatly
        # handle bounds in hierarchical blocking and SubDimensions
        proc_parents_max = {}

        # Process inner iterations and adjust their bounds
        for n, i in enumerate(inner):
            if i.dim.parent in proc_parents_max and i.symbolic_size == i.dim.parent.step:
                # Use parent's Iteration max in hierarchical blocking
                iter_max = proc_parents_max[i.dim.parent]
            else:
                # The Iteration's maximum is the MIN of (a) the `symbolic_max` of current
                # Iteration e.g. `x0_blk0 + x0_blk0_size - 1` and (b) the `symbolic_max`
                # of the current Iteration's root Dimension e.g. `x_M`. The generated
                # maximum will be `MIN(x0_blk0 + x0_blk0_size - 1, x_M)

                # In some corner cases an offset may be added (e.g. after CIRE passes)
                # E.g. assume `i.symbolic_max = x0_blk0 + x0_blk0_size + 1` and
                # `i.dim.symbolic_max = x0_blk0 + x0_blk0_size - 1` then the generated
                # maximum will be `MIN(x0_blk0 + x0_blk0_size + 1, x_M + 2)`

                root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max

                try:
                    iter_max = (min(i.symbolic_max, root_max))
                    bool(iter_max)  # Can it be evaluated?
                except TypeError:
                    iter_max = MIN(i.symbolic_max, root_max)

            proc_parents_max[i.dim] = iter_max

            mapper[i] = i._rebuild(limits=(i.symbolic_min, iter_max, i.step))

    iet = Transformer(mapper, nested=True).visit(iet)

    headers = [('%s(a,b)' % MIN.name, ('(((a) < (b)) ? (a) : (b))')),
               ('%s(a,b)' % MAX.name, ('(((a) > (b)) ? (a) : (b))'))]

    return iet, {'headers': headers}


def linearize(iet, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j]` -> `a[i*n + j]`.
    """
    # Simple data structure to avoid generation of duplicated code
    cache = defaultdict(lambda: Bunch(stmts0=[], stmts1=[], cbk=None))

    linearize_accesses(iet, cache=cache, **kwargs)


@iet_pass
def linearize_accesses(iet, **kwargs):
    """
    Actually implement linearize()
    """
    sregistry = kwargs['sregistry']
    cache = kwargs['cache']

    # Find unique sizes (unique -> minimize necessary registers)
    symbol_names = {i.name for i in FindSymbols('indexeds').visit(iet)}
    functions = [f for f in FindSymbols().visit(iet)
                 if f.is_AbstractFunction and f.name in symbol_names]
    functions = sorted(functions, key=lambda f: len(f.dimensions), reverse=True)
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
        name = sregistry.make_name(prefix='%s_fsz' % d.name)
        s = Symbol(name=name, dtype=np.int32, is_const=True)
        try:
            expr = LocalExpression(DummyEq(s, v[0]._C_get_field(FULL, d).size))
        except AttributeError:
            assert v[0].is_Array
            expr = LocalExpression(DummyEq(s, v[0].symbolic_shape[d]))
        for f in v:
            imapper[f].append((d, s))
            cache[f].stmts0.append(expr)

    # Build all exprs such as `y_slc0 = y_fsz0*z_fsz0`
    built = {}
    mapper = DefaultOrderedDict(list)
    for f, v in imapper.items():
        for n, (d, _) in enumerate(v):
            expr = prod(list(zip(*v[n:]))[1])
            try:
                s = built[expr]
            except KeyError:
                name = sregistry.make_name(prefix='%s_slc' % d.name)
                s = built[expr] = Symbol(name=name, dtype=np.int32, is_const=True)
                cache[f].stmts1.append(LocalExpression(DummyEq(s, expr)))
            mapper[f].append(s)
    mapper.update([(f, []) for f in functions if f not in mapper])

    # Build defines. For example:
    # `define uL(t, x, y, z) ul[(t)*t_slice_sz + (x)*x_slice_sz + (y)*y_slice_sz + (z)]`
    headers = []
    findexeds = {}
    for f, szs in mapper.items():
        if cache[f].cbk is not None:
            # Perhaps we've already built an access macro for `f` through another efunc
            findexeds[f] = cache[f].cbk
        else:
            assert len(szs) == len(f.dimensions) - 1
            pname = sregistry.make_name(prefix='%sL' % f.name)
            sname = sregistry.make_name(prefix='%sl' % f.name)

            expr = sum([MacroArgument(d.name)*s for d, s in zip(f.dimensions, szs)])
            expr += MacroArgument(f.dimensions[-1].name)
            expr = Indexed(IndexedData(sname, None, f), expr)
            define = DefFunction(pname, f.dimensions)
            headers.append((ccode(define), ccode(expr)))

            cache[f].cbk = findexeds[f] = lambda i, p=pname, s=sname: FIndexed(i, p, s)

    # Build "functional" Indexeds. For example:
    # `u[t2, x+8, y+9, z+7] => uL(t2, x+8, y+9, z+7)`
    mapper = {}
    for n in FindNodes(Expression).visit(iet):
        subs = {i: findexeds[i.function](i) for i in retrieve_indexed(n.expr)}
        mapper[n] = n._rebuild(expr=uxreplace(n.expr, subs))

    # Put together all of the necessary exprs for `y_fsz0`, ..., `y_slc0`, ...
    stmts0 = filter_ordered(flatten(cache[f].stmts0 for f in functions))
    if stmts0:
        stmts0.append(BlankLine)
    stmts1 = filter_ordered(flatten(cache[f].stmts1 for f in functions))
    if stmts1:
        stmts1.append(BlankLine)

    iet = Transformer(mapper).visit(iet)
    iet = iet._rebuild(body=List(body=stmts0 + stmts1 + list(iet.body)))

    return iet, {'headers': headers}


def is_on_device(obj, gpu_fit):
    """
    True if the given object is allocated in the device memory, False otherwise.

    Parameters
    ----------
    obj : Indexed or Function
        The target object.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    """
    functions = (obj.function,)
    fsave = [f for f in functions if f.is_TimeFunction and is_integer(f.save)]
    if 'all-fallback' in gpu_fit and fsave:
        warning("TimeFunction %s assumed to fit the GPU memory" % fsave)
        return True

    return all(f in gpu_fit for f in fsave)

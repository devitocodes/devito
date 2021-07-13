from itertools import product

import cgen
import numpy as np

from devito.data import FULL
from devito.ir import DummyEq
from devito.ir.iet import (BlankLine, Expression, List, LocalExpression, Prodder,
                           FindNodes, FindSymbols, Transformer, make_efunc, compose_nodes,
                           filter_iterations, retrieve_iteration_tree)
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (DefFunction, MacroArgument, ccode, retrieve_indexed,
                              uxreplace)
from devito.tools import DefaultOrderedDict, flatten, is_integer, prod, split
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
    iet = iet._rebuild(body=(List(header=header),) + iet.body)
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
    Recast Iterations over IncrDimensions as ElementalFunctions; insert
    ElementalCalls to iterate over the "main" and "remainder" regions induced
    by the IncrDimensions.
    """
    sregistry = kwargs['sregistry']

    efuncs = []
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Incr]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Incr)

        # Compute the iteration ranges
        ranges = []
        for i in outer:
            maxb = i.symbolic_max - (i.symbolic_size % i.dim.step)
            ranges.append(((i.symbolic_min, maxb, i.dim.step),
                           (maxb + 1, i.symbolic_max, i.symbolic_max - maxb)))

        # Remove any offsets
        # E.g., `x = x_m + 2 to x_M - 2` --> `x = x_m to x_M`
        outer = [i._rebuild(limits=(i.dim.root.symbolic_min, i.dim.root.symbolic_max,
                                    i.step))
                 for i in outer]

        # Create the ElementalFunction
        name = sregistry.make_name(prefix="bf")
        body = compose_nodes(outer)
        dynamic_parameters = flatten((i.symbolic_bounds, i.step) for i in outer)
        dynamic_parameters.extend([i.step for i in inner if not is_integer(i.step)])
        efunc = make_efunc(name, body, dynamic_parameters)

        efuncs.append(efunc)

        # Create the ElementalCalls
        calls = []
        for p in product(*ranges):
            dynamic_args_mapper = {}
            for i, (m, M, b) in zip(outer, p):
                dynamic_args_mapper[i.symbolic_min] = m
                dynamic_args_mapper[i.symbolic_max] = M
                dynamic_args_mapper[i.step] = b
                for j in inner:
                    if j.dim.root is i.dim.root and not is_integer(j.step):
                        value = j.step if b is i.step else b
                        dynamic_args_mapper[j.step] = (value,)
            calls.append(efunc.make_call(dynamic_args_mapper))

        mapper[root] = List(body=calls)

    iet = Transformer(mapper).visit(iet)

    return iet, {'efuncs': efuncs}


@iet_pass
def linearize(iet, **kwargs):
    """
    Turn n-dimensional Indexeds into 1-dimensional Indexed with suitable index
    access function, such as `a[i, j]` -> `a[i*n + j]`.
    """
    sregistry = kwargs['sregistry']

    # Find unique sizes (unique -> minimize necessary registers)
    functions = [f for f in FindSymbols().visit(iet) if f.is_AbstractFunction]
    functions = sorted(functions, key=lambda f: len(f.dimensions), reverse=True)
    mapper = DefaultOrderedDict(list)
    for f in functions:
        for d in f.dimensions[1:]:  # NOTE: the outermost dimension is unnecessary
            #TODO: THIS SHOULD TAKE PADDING INTO ACCOUNT TOO
            #TODO: need to provide option "assume-same-padding" ??
            mapper[(d, f._size_halo[d], getattr(f, 'grid', None))].append(f)

    # Build all exprs such as `xs = u_vec->size[1]`
    imapper = DefaultOrderedDict(list)
    stmts = []
    for (d, halo, _), v in mapper.items():
        name = sregistry.make_name(prefix='%s_fsz' % d.name)
        s = Symbol(name=name, dtype=np.int32, is_const=True)
        try:
            expr = DummyEq(s, v[0]._C_get_field(FULL, d).size)
        except AttributeError:
            assert v[0].is_Array
            expr = DummyEq(s, v[0].symbolic_shape[d])
        stmts.append(LocalExpression(expr))
        for f in v:
            imapper[f].append((d, s))
    stmts.append(BlankLine)

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
                stmts.append(LocalExpression(DummyEq(s, expr)))
            mapper[f].append(s)
    mapper.update([(f, []) for f in functions if f not in mapper])
    stmts.append(BlankLine)

    # Build defines. For example:
    # `define uL(t, x, y, z) ul[(t)*t_slice_sz + (x)*x_slice_sz + (y)*y_slice_sz + (z)]`
    headers = []
    findexeds = {}
    for f, szs in mapper.items():
        assert len(szs) == len(f.dimensions) - 1
        pname = sregistry.make_name(prefix='%sL' % f.name)
        sname = sregistry.make_name(prefix='%sl' % f.name)

        expr = sum([MacroArgument(d.name)*s for d, s in zip(f.dimensions, szs)])
        expr += MacroArgument(f.dimensions[-1].name)
        expr = Indexed(IndexedData(sname, None, f), expr)
        define = DefFunction(pname, f.dimensions)
        headers.append((ccode(define), ccode(expr)))

        findexeds[f] = lambda i, pname=pname, sname=sname: FIndexed(i, pname, sname)

    # Build "functional" Indexeds. For example:
    # `u[t2, x+8, y+9, z+7] => uL(t2, x+8, y+9, z+7)`
    mapper = {}
    for n in FindNodes(Expression).visit(iet):
        subs = {i: findexeds[i.function](i) for i in retrieve_indexed(n.expr)}
        mapper[n] = n._rebuild(expr=uxreplace(n.expr, subs))

    iet = Transformer(mapper).visit(iet)
    iet = iet._rebuild(body=List(body=list(stmts) + list(iet.body)))

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

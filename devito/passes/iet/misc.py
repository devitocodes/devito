from itertools import product

import cgen

from devito.ir.iet import (Expression, List, Prodder, FindNodes, FindSymbols,
                           Transformer, make_efunc, compose_nodes, filter_iterations,
                           retrieve_iteration_tree, Section)
from devito.ir.equations import DummyEq
from devito.passes.iet.engine import iet_pass
from devito.symbolics import INT
from devito.tools import flatten, is_integer, split
from devito.logger import warning
from sympy import Min, Max
from devito.types import Scalar, Symbol
import numpy as np

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions', 'is_on_device']


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
    #import pdb;pdb.set_trace()
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
        new_outer = [i._rebuild(limits=(i.dim.root.symbolic_min, i.dim.root.symbolic_max,
                                    i.step))
                 for i in outer]


#[((x_m, x_M - Mod(x_M - x_m + 1, x0_blk0_size), x0_blk0_size), (x_M - Mod(x_M - x_m + 1, x0_blk0_size) + 1, x_M, Mod(x_M - x_m + 1, x0_blk0_size))), ((y_m, y_M - Mod(y_M - y_m + 1, y0_blk0_size), y0_blk0_size), (y_M - Mod(y_M - y_m + 1, y0_blk0_size) + 1, y_M, Mod(y_M - y_m + 1, y0_blk0_size)))]

        new_body = compose_nodes(outer)
        #inner[0].dim.root.symbolic_max + inner[0].symbolic_size - inner[0].symbolic_max + inner[0].symbolic_min


# if - i.dim.symbolic_max + i.dim.symbolic_size + i.dim.symbolic_min > 1
# b1+1
# i.dim.symbolic_max - i.symbolic_min > i.dim.symbolic_size

        nodes1 = []
        nodes2 = []
        for i in inner:
            
            # import pdb;pdb.set_trace()
            offset = i.symbolic_size - i.symbolic_max + i.symbolic_min
            b0a = i.dim.parent.symbolic_max + i.symbolic_max - i.dim.parent.step - i.dim.symbolic_min + offset

            b0b = i.dim.parent.symbolic_max
            b1 = Max( b0a, b0b)

            ub = INT(Min(i.symbolic_max, b1))
            #inner_start = Symbol(name="%s_lb" % i.dim.name, dtype=np.int32)
            #lb_expr = Expression(DummyEq(inner_start, lb))
            new_inner = i._rebuild(limits=(i.symbolic_min, ub, i.step))
            #nodes1.append(lb_expr)
            nodes2.append(new_inner)


        # nodes1body = compose_nodes(nodes1)

        root = inner[0]
        #expr_body = compose_nodes(nodes1)
        inner_body = compose_nodes(nodes2)
        #nodes1.append(inner_body)



        test_1 = List(body= inner_body)
        mapper[root] = test_1

        # import pdb;pdb.set_trace()        


    #import pdb;pdb.set_trace()

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {'efuncs': efuncs}


def is_on_device(maybe_symbol, gpu_fit, only_writes=False):
    """
    True if all given Functions are allocated in the device memory, False otherwise.

    Parameters
    ----------
    maybe_symbol : Indexed or Function or Node
        The inspected object. May be a single Indexed or Function, or even an
        entire piece of IET.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    only_writes : bool, optional
        Only makes sense if `maybe_symbol` is an IET. If True, ignore all Function's
        that do not appear on the LHS of at least one Expression. Defaults to False.
    """
    try:
        functions = (maybe_symbol.function,)
    except AttributeError:
        assert maybe_symbol.is_Node
        iet = maybe_symbol
        functions = set(FindSymbols().visit(iet))
        if only_writes:
            expressions = FindNodes(Expression).visit(iet)
            functions &= {i.write for i in expressions}

    fsave = [f for f in functions if f.is_TimeFunction and f.save is not None]
    if 'all-fallback' in gpu_fit and fsave:
        warning("TimeFunction %s assumed to fit the GPU memory" % fsave)
        return True

    return all(f in gpu_fit for f in fsave)

"""Misc Target passes."""

import cgen

from devito.ir.iet import (Iteration, List, Prodder, FindSymbols, FindNodes,
                           Transformer, filter_iterations, retrieve_iteration_tree)
from devito.logger import perf_adv
from devito.targets.common.blocking import BlockDimension
from devito.targets.common.engine import target_pass

__all__ = ['avoid_denormals', 'loop_wrapping', 'minimize_remainders', 'hoist_prodders']


@target_pass
def avoid_denormals(iet):
    """
    Introduce nodes in the Iteration/Expression tree that will expand to C
    macros telling the CPU to flush denormal numbers in hardware. Denormals
    are normally flushed when using SSE-based instruction sets, except when
    compiling shared objects.
    """
    header = (cgen.Comment('Flush denormal numbers to zero in hardware'),
              cgen.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)'),
              cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'))
    iet = iet._rebuild(body=(List(header=header),) + iet.body)
    return iet, {'includes': ('xmmintrin.h', 'pmmintrin.h')}


@target_pass
def loop_wrapping(iet):
    """
    Emit a performance message if WRAPPABLE Iterations are found,
    as these are a symptom that unnecessary memory is being allocated.
    """
    for i in FindNodes(Iteration).visit(iet):
        if not i.is_Wrappable:
            continue
        perf_adv("Functions using modulo iteration along Dimension `%s` "
                 "may safely allocate a one slot smaller buffer" % i.dim)
    return iet, {}


@target_pass
def minimize_remainders(iet, **kwargs):
    """
    Adjust ROUNDABLE Iteration bounds so as to avoid the insertion of remainder
    loops by the backend compiler.
    """
    simd_items_per_reg = kwargs.pop('simd_items_per_reg')

    roundable = [i for i in FindNodes(Iteration).visit(iet) if i.is_Roundable]

    mapper = {}
    for i in roundable:
        functions = FindSymbols().visit(i)

        # Get the SIMD vector length
        dtypes = {f.dtype for f in functions if f.is_Tensor}
        assert len(dtypes) == 1
        vl = simd_items_per_reg(dtypes.pop())

        # Round up `i`'s max point so that at runtime only vector iterations
        # will be performed (i.e., remainder loops won't be necessary)
        m, M, step = i.limits
        limits = (m, M + (i.symbolic_size % vl), step)

        mapper[i] = i._rebuild(limits=limits)

    iet = Transformer(mapper).visit(iet)

    return iet, {}


@target_pass
def hoist_prodders(iet):
    """
    Move Prodders within the outer levels of an Iteration tree.
    """
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        for prodder in FindNodes(Prodder).visit(tree.root):
            if prodder._periodic:
                try:
                    key = lambda i: isinstance(i.dim, BlockDimension)
                    candidate = filter_iterations(tree, key)[-1]
                except IndexError:
                    # Fallback: use the outermost Iteration
                    candidate = tree.root
                mapper[candidate] = candidate._rebuild(nodes=(candidate.nodes +
                                                              (prodder._rebuild(),)))
                mapper[prodder] = None

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}

from itertools import product

import cgen

from devito.ir.iet import (Iteration, List, Prodder, FindNodes, Transformer, make_efunc,
                           compose_nodes, filter_iterations, retrieve_iteration_tree)
from devito.logger import perf_adv
from devito.passes.iet.engine import iet_pass
from devito.tools import flatten, is_integer, split

__all__ = ['avoid_denormals', 'loop_wrapping', 'hoist_prodders', 'relax_incr_dimensions']


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
              cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'))
    iet = iet._rebuild(body=(List(header=header),) + iet.body)
    return iet, {'includes': ('xmmintrin.h', 'pmmintrin.h')}


@iet_pass
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
                    key = lambda i: i.dim.is_Incr
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
def relax_incr_dimensions(iet):
    """
    Recast Iterations over IncrDimensions as ElementalFunctions; insert
    ElementalCalls to iterate over the "main" and "remainder" regions induced
    by the IncrDimensions.
    """
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
        name = "bf%d" % len(mapper)
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

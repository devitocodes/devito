import cgen

from devito.ir.iet import (List, Prodder, FindNodes, Transformer, filter_iterations,
                           retrieve_iteration_tree)
from devito.ir.support import Forward
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import MIN, MAX
from devito.tools import split, is_integer

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
    This pass adjusts the bounds of blocked Iterations in order to include the "remainder
    regions".  Without the relaxation that occurs in this pass, the only way to iterate
    over the entire iteration space is to have step increments that are perfect divisors
    of the iteration space (e.g. in case of an iteration space of size 67 and block size
    8 only 64 iterations would be computed as `67 - 67mod8 = 64`

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
        # handle bounds in hierarchical blocking and subdimensions
        proc_parents_max = {}

        # Process inner iterations and adjust their bounds
        for n, i in enumerate(inner):
            if i.dim.parent in proc_parents_max and i.symbolic_size == i.dim.parent.step:
                # Case A: An Iteration has a parent Dimension that has already been
                # processed as an 'inner' Iteration. Mainly encountered in hierarchical
                # blocking (BLOCKLEVELS > 1). In hierarchical blocking block sizes in
                # lower levels perfectly divide block sizes of upper levels so we can
                # use the parent's Iteration maximum
                iter_max = proc_parents_max[i.dim.parent]
            else:
                # Case B: General case, most of the cases pass through here. There are
                # two candidates for the Iteration's maximum (C1 and C2), C1 denotes the
                # upper bound of the "main" blocked area and C2 the upper bound of the
                # "remainder area". The Iteration's maximum is MIN(C1, C2)

                # Candidate 1 (C1): `symbolic_max` of current Iteration, the maximum of
                # step increments e.g. `i.symbolic_max = x0_blk0 + x0_blk0_size`

                # Candidate 2 (C2): maximum of the root Iteration. It is usually the max
                # of root Dimension, e.g. `x_M` though there are some corner cases where
                # the maximum needs to be exceeded (e.g. after CIRE passes). We get this
                # additional margin by checking the the `symbolic_size` of the Iteration
                # compared to the size of a parent's block size  e.g.
                # `i.dim.parent.step = x0_blk1_size` while
                # `i.symbolic_size = x0_blk1_size + 4`
                # This margin of `4` needs to be added to the "remainder" maximum in
                # order not to omit iterations That may result in using `x_M + 1` or
                # `x_M + 4` instead of the expected `x_M`.

                root_max = roots_max[i.dim.root]
                lb_margin = i.symbolic_min - i.dim.symbolic_min  # lower bound margin
                size_margin = i.symbolic_size - i.dim.parent.step  # symbolic size margin

                domain_max = root_max + size_margin + lb_margin  # root max extended

                # Finally the Iteration's maximum is the minimum of C1 and C2
                # e.g. `iter_max = MIN(x0_blk0 + x0_blk0_size, domain_max)``
                try:
                    bool(min(i.symbolic_max, domain_max))
                    iter_max = (min(i.symbolic_max, domain_max))
                except TypeError:
                    iter_max = MIN(i.symbolic_max, domain_max)

            proc_parents_max[i.dim] = iter_max

            mapper[i] = i._rebuild(limits=(i.symbolic_min, iter_max, i.step))

    iet = Transformer(mapper, nested=True).visit(iet)

    headers = [('%s(a,b)' % MIN.name, ('(((a) < (b)) ? (a) : (b))')),
               ('%s(a,b)' % MAX.name, ('(((a) > (b)) ? (a) : (b))'))]

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

import cgen

from devito.ir.iet import (List, Prodder, FindNodes, Transformer, filter_iterations,
                           retrieve_iteration_tree)
from devito.ir.support import Forward
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import MIN, MAX
from devito.tools import split, is_integer

__all__ = ['avoid_denormals', 'hoist_prodders', 'finalize_loop_bounds', 'is_on_device']


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
def finalize_loop_bounds(iet, **kwargs):
    """
    This pass is adjusting the bounds of space-blocked loops in order to
    include the "remainder regions". The iterations in the input IET span
    only to the extent of the last space block that can be fully iterated,
    thus missing the "remainder" part.

    A simple example (blocking only), nested Iterations are
    transformed from:

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

        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Incr)

        # Get root symbolic_max out of each outer dimension
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # A dictionary to map maximum of processed parent dimensions
        # Helps to neatly handle bounds in hierarchical blocking
        proc_parents_max = {}

        # Process inner iterations
        for n, i in enumerate(inner):
            assert i.direction is Forward
            if i.dim.parent in proc_parents_max and i.symbolic_size == i.dim.parent.step:
                # In hierarchical blocking (BLOCKLEVELS > 1) a parent dimension may
                # have already been processed as an 'inner' iteration. Since in
                # hierarchical blocking, block sizes in lower levels always perfectly
                # divide block sizes of upper levels we can use parent's iteration
                # maximum.
                iter_max = proc_parents_max[i.dim.parent]
            else:
                # Candidate 1: symbolic_max of current iteration
                # Most of the cases pass though this code:
                # Candidates for iteration's maximum are:
                # Candidate 1: symbolic_max of current iteration
                # e.g. `i.symbolic_max = x0_blk0 + x0_blk0_size`
                symbolic_max = i.symbolic_max

                # Candidate 2: maximum of the root iteration
                # Candidate 2a: Usualy it is the max of parent/root
                # dimension, e.g. `x_M`
                root_max = roots_max[i.dim.root]

                # Candidate 2b:
                # the symbolic_size of an Iteration may exceed the size of a parent's
                # block size (e.g. after CIRE passes)
                # e.g.
                # `i.dim.parent.step = x0_blk1_size`
                # `i.symbolic_size = x0_blk1_size + 4`
                # For this case, proper margin should be allowed
                # in order not to omit loop iterations.
                lb_margin = i.symbolic_min - i.dim.symbolic_min
                size_margin = i.symbolic_size - i.dim.parent.step
                ub_margin = i.dim.parent.symbolic_max + size_margin + lb_margin

                # So Candidate 2 is the maximum of the
                # root's max (2a) and the current iteration's required max (2b)
                # and instead of `x_M` we may need `x_M + 1` or `x_M + 2`

                # So candidate 2 is
                # e.g. `domain_max = Max(x_M + 1, x_M)``
                domain_max = MAX(ub_margin, root_max)

                # Finally the iteration's maximum is the minimum of the
                # candidates 1 and 2
                # e.g. `iter_max = Min(x0_blk0 + x0_blk0_size, domain_max)``
                iter_max = MIN(symbolic_max, domain_max)

            # Store the selected maximum of this iteration's dimension for
            # possible reference in case of children iterations
            # Usually encountered in subdims and hierarchical blocking
            proc_parents_max[i.dim] = iter_max

            mapper[i] = i._rebuild(limits=(i.symbolic_min, iter_max, i.step))

    iet = Transformer(mapper, nested=True).visit(iet)

    headers = [('MIN(a,b)', ('(((a) < (b)) ? (a) : (b))')),
               ('MAX(a,b)', ('(((a) > (b)) ? (a) : (b))'))]
    return iet, {'headers': headers}

    # return iet, {}


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

import cgen

from devito.ir.iet import (List, Prodder, FindNodes, Transformer, filter_iterations,
                           retrieve_iteration_tree)
from devito.ir.support import Forward
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import MIN, MAX
from devito.tools import is_integer, split

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
        roots_min = {i.dim.root: i.symbolic_min for i in outer}

        # A dictionary to map maximum of processed parent dimensions. Helps to neatly
        # handle bounds in hierarchical blocking and SubDimensions
        proc_parents_max = {}
        proc_parents_min = {}

        skew_dim = 0
        if inner[0].dim.is_Time:
            skew_dim = inner[0].dim

        # Process inner iterations and adjust their bounds
        for n, i in enumerate(inner):
            assert i.direction is Forward

            if i.dim.parent in proc_parents_max and i.symbolic_size == i.dim.parent.step:
                iter_max = MIN(proc_parents_max[i.dim.parent], i.dim.symbolic_max)

                if skew_dim and not i.dim.is_Time:
                    # import pdb;pdb.set_trace()
                    # symbolic_min = i.symbolic_min - skew_dim
                    symbolic_min = i.dim.symbolic_min  # proc_parents_min[i.dim.parent]
                else:
                    symbolic_min = i.symbolic_min
            else:
                # Most of the cases pass though this code:
                # Candidates for upper bound calculation are:
                # Candidate 1: symbolic_max of current iteration
                # e.g.
                # i.symbolic_max = x0_blk0 + x0_blk0_size
                # symbolic_max = i.symbolic_max

                # Candidate 2: The domain max. Usualy it is the max of parent/root
                # dimension.
                # e.g. x_M
                # This may not always be true as the symbolic_size of an Iteration may
                # exceed the size of a parent's block size (e.g. after CIRE passes)
                # e.g.
                # i.dim.parent.step = x0_blk1_size
                # i.symbolic_size = x0_blk1_size + 4

                # For this case, proper margin should be allowed
                # in order not to drop iterations. So Candidate 2 is the maximum of the
                # root's max and the current iteration's required max
                # and instead of `x_M` we may need `x_M + 1` or `x_M + 2`
                root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max

                if skew_dim and not i.dim.is_Time:
                    iter_max = MIN(i.symbolic_max, root_max + skew_dim)
                    # domain_max = MAX(upper_margin, root_max + skew_dim)
                else:
                    iter_max = MIN(i.symbolic_max, root_max)
                    # Finally our upper bound is the minimum of upper bound candidates
                # e.g. upper_bound = Min(x0_blk0 + x0_blk0_size, domain_max)
                # iter_max = MIN(i.symbolic_max, root_max)

                # Min(symbolic_max, domain_max)

                if skew_dim and not i.dim.is_Time:
                    symbolic_min = MAX(i.symbolic_min, roots_min[i.dim.root] + skew_dim)
                else:
                    symbolic_min = i.symbolic_min

            # Store the selected maximum of this iteration's dimension for
            # possible reference in case of children iterations
            # Usually encountered in subdims and hierarchical blocking
            proc_parents_max[i.dim] = iter_max
            proc_parents_min[i.dim] = symbolic_min

            mapper[i] = i._rebuild(limits=(symbolic_min, iter_max, i.step))

        for n, i in enumerate(outer):
            assert i.direction is Forward

            if skew_dim and not i.dim.is_Time:
                time_size = skew_dim.root.symbolic_max - skew_dim.root.symbolic_min
                mapper[i] = i._rebuild(limits=(i.symbolic_min,
                                               i.symbolic_max + time_size, i.step))

    if mapper:
        iet = Transformer(mapper, nested=True).visit(iet)

        headers = [('%s(a,b)' % MIN.name, ('(((a) < (b)) ? (a) : (b))')),
                   ('%s(a,b)' % MAX.name, ('(((a) > (b)) ? (a) : (b))'))]
    else:
        headers = []

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

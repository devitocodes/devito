import cgen

from devito.ir import (Forward, List, Prodder, FindNodes, Transformer,
                       filter_iterations, retrieve_iteration_tree)
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.passes.clusters.utils import level
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

        # Get the skew dimension. 0 if not applicable
        skew_dim = (inner[0].dim if inner[0].dim.is_Time else 0)

        # Process inner iterations and adjust their bounds
        for i in inner:
            # The Iteration's maximum is the MIN of (a) the `symbolic_max` of current
            # Iteration e.g. `x0_blk0 + x0_blk0_size - 1` and (b) the `symbolic_max`
            # of the current Iteration's root Dimension e.g. `x_M`. The generated
            # maximum will be `MIN(x0_blk0 + x0_blk0_size - 1, x_M)

            # In some corner cases an offset may be added (e.g. after CIRE passes)
            # E.g. assume `i.symbolic_max = x0_blk0 + x0_blk0_size + 1` and
            # `i.dim.symbolic_max = x0_blk0 + x0_blk0_size - 1` then the generated
            # maximum will be `MIN(x0_blk0 + x0_blk0_size + 1, x_M + 2)`

            root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max
            symbolic_max = i.symbolic_max
            symbolic_min = i.symbolic_min

            if skew_dim and skew_dim is not i.dim:
                root_max = roots_max[i.dim.root] + skew_dim
                if level(i.dim) == 2:  # At skewing level
                    symbolic_max = i.dim.symbolic_max
                    root_min = roots_min[i.dim.root] + i.symbolic_min - i.dim.symbolic_min
                    symbolic_min = evalmax(root_min, i.dim.symbolic_min)
                elif level(i.dim) > 2:  # In TB, multiple levels need parent symbolic_max
                    symbolic_max = evalmin(proc_parents_max[i.dim.parent], symbolic_max)

            proc_parents_max[i.dim] = symbolic_max

            iter_max = evalmin(symbolic_max, root_max)
            mapper[i] = i._rebuild(limits=(symbolic_min, iter_max, i.step))

        for i in outer:
            if skew_dim and skew_dim.parent is not i.dim:
                mapper[i] = i._rebuild(limits=(i.symbolic_min, i.symbolic_max +
                                       skew_dim.parent.symbolic_size, i.step))

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


def evalmin(a, b):
    """
    """
    try:
        bool(min(a, b))  # Can it be evaluated?
        return min(a, b)
    except TypeError:
        return MIN(a, b)


def evalmax(a, b):
    """
    """
    try:
        bool(max(a, b))  # Can it be evaluated?
        return max(a, b)
    except TypeError:
        return MAX(a, b)

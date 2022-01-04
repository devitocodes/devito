import cgen

from collections import OrderedDict
from itertools import product, chain
from devito.ir import (Forward, List, Prodder, FindNodes, Transformer,
                       filter_iterations, retrieve_iteration_tree)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import MIN, MAX, reduce_relation, rfunc
from devito.types.relational import Gt
from devito.tools import is_integer, split, as_list

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions']


@iet_pass
def avoid_denormals(iet, platform=None):
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
        iterations = [i for i in tree if i.dim.is_Block]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        assert all(i.direction is Forward for i in iterations)
        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Block)

        # Get root's `symbolic_max` out of each outer Dimension
        roots_min = {i.dim.root: i.symbolic_min for i in outer}
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # Process inner iterations and adjust their bounds
        for i in inner:
            # This part of the code aims to collect the data necessary to build the
            # MIN/MAX bounds of an Iteration. The main challenge is to correctly
            # take care of interval offsets (incurred by several Devito optimizations),
            # subdimension offsets and blocking loops (main and remainder areas).

            # The Iteration's maximum is the MIN of (a) the `symbolic_max` of current
            # Iteration e.g. `x0_blk0 + x0_blk0_size - 1` and (b) the `symbolic_max`
            # of the current Iteration's root Dimension e.g. `x_M`. The generated
            # maximum will be `MIN(x0_blk0 + x0_blk0_size - 1, x_M)

            # In some corner cases an offset may be added (e.g. after CIRE passes)
            # E.g. assume `i.symbolic_max = x0_blk0 + x0_blk0_size + 1` and
            # `i.dim.symbolic_max = x0_blk0 + x0_blk0_size - 1` then the generated
            # maximum will be `MIN(x0_blk0 + x0_blk0_size + 1, x_M + 2)`

            # Step 1: Interval offset care
            # Here we create two dictionaries to keep track of offsets incurring in
            # 'i' and 'i.root' iteration
            root_min = roots_min[i.dim.root] + i.symbolic_min - i.dim.symbolic_min
            min_map = {
                i.dim.root.symbolic_min: root_min,
                i.dim.symbolic_min: i.symbolic_min
            }

            root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max
            max_map = {
                i.dim.root.symbolic_max: root_max,
                i.dim.symbolic_max: i.symbolic_max
            }

            # Step 2: Generate assumptions deriving from hierarchical blocking. Assuming
            # lower blocking levels perfectly fit within outer levels, we use depth of
            # IncrDimensions as a rule to generate assumptions such as:
            # e.g. for 2-level blocking [x (inner Incr), x0_blk1, x0_blk0, x (outer root)]
            # [x0_blk1 > x0_blk0, x0_blk1_size > x0_blk0_size]
            # `defines` keeps all the ancestors and descendants of `i.dim`.
            defines = sorted(i.dim._defines, key=lambda x: (not x.is_Incr or -x._depth))

            # Collect assumptions based on hierarchy
            acs = [j for j in defines if j is not i.dim and j.is_Incr]
            assumptions = as_list(chain.from_iterable((Gt(j.step, k.step), Gt(j, k))
                                  for j, k in product(acs, acs) if j._depth > k._depth))

            # Step 3: Reduce defines min/max candidates using assumptions on `defines`
            defmin = reduce_relation(max, defines, assumptions)
            defmax = reduce_relation(min, defines, assumptions)

            # Collect symbolic_min, symbolic_max from defmin/defmax lists
            defmin = [j.symbolic_min for j in defmin]
            defmax = [j.symbolic_max for j in defmax]

            # Drop duplicates from defmin/defmax and keep them Ordered
            defmin = list(OrderedDict.fromkeys(defmin))
            defmax = list(OrderedDict.fromkeys(defmax))

            # Step 4: Drop a candidate if it is the symbolic_min/max of symbol candidates
            for k in defmin.copy():  # TOFIX: Should not use a copy I guess?
                try:
                    if k.symbolic_min in defmin:
                        defmin.remove(k.symbolic_min)
                except:
                    pass

            for k in defmax.copy():
                try:
                    if k.symbolic_max in defmax:
                        defmax.remove(k.symbolic_max)
                except:
                    pass

            # At this point the candidates for the start and end of the `Iteration`
            # comprise of candidates stemming (i) from `i.dim` and (ii) candidates
            # stemming from other dims, (in space blocking other dims is usually root.)
            # The canidates stemming from 'i.dim' can be compared and for those we need
            # the minimum of them for the start and the maximum of them for the end so
            # as to span the whole extent of the Iteration's Dimension (i.dim iteration
            # extent).
            # We split on comparable and uncomparable. Comparable ones can be further
            # simplfied, (subdimensions, offsets incurred) and non-comparable ones
            # include candidates that cannot be simplified more (no clear rule of
            # further simplification exists)
            # For the start: MAX(uncomparable, MIN(comparable)) --> MAX(uncomparable)
            # For the end: MIN(uncomparable, MAX(comparable)) --> MIN(uncomparable)
            defmin = reduce_relation(min, defmin, assumptions=None)
            defmax = reduce_relation(max, defmax, assumptions=None)

            # Apply offset if needed, else leave as it is
            defmin = [j.subs(min_map) if j in min_map else j for j in as_list(defmin)]
            defmax = [j.subs(max_map) if j in max_map else j for j in as_list(defmax)]

            # Final MIN/MAX processing
            iter_min = rfunc(max, *defmin)
            iter_max = rfunc(min, *defmax)

            # Final iteration form will look like:
            # Iteration i <starting from MAX(uncomparable, MIN(comparable))
            # to MIN(uncomparable, MAX(comparable))>
            mapper[i] = i._rebuild(limits=(iter_min, iter_max, i.step))

    if mapper:
        iet = Transformer(mapper, nested=True).visit(iet)

        headers = [('%s(a,b)' % MIN.name, ('(((a) < (b)) ? (a) : (b))')),
                   ('%s(a,b)' % MAX.name, ('(((a) > (b)) ? (a) : (b))'))]
    else:
        headers = []

    return iet, {'headers': headers}

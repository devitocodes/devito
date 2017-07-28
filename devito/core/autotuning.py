from __future__ import absolute_import

from collections import OrderedDict
from itertools import combinations
from functools import reduce
from operator import mul
import resource

from devito.logger import info, info_at
from devito.nodes import Iteration
from devito.visitors import FindNodes, FindSymbols

__all__ = ['autotune']


def autotune(operator, arguments, tunable, mode='basic'):
    """
    Acting as a high-order function, take as input an operator and a list of
    operator arguments to perform empirical autotuning. Some of the operator
    arguments are marked as tunable.
    """
    at_arguments = arguments.copy()

    # User-provided output data must not be altered
    output = [i.name for i in operator.output]
    for k, v in arguments.items():
        if k in output:
            at_arguments[k] = v.copy()

    # Squeeze dimensions to minimize auto-tuning time
    iterations = FindNodes(Iteration).visit(operator.body)
    dim_mapper = {i.dim.name: i.dim for i in iterations}
    squeezable = [i.dim.parent.symbolic_size.name for i in iterations
                  if i.is_Sequential and i.dim.is_Buffered]

    # Attempted block sizes
    mapper = OrderedDict([(i.argument.symbolic_size.name, i) for i in tunable])
    blocksizes = [OrderedDict([(i, v) for i in mapper])
                  for v in options['at_blocksize']]
    if mode == 'aggressive':
        blocksizes = more_heuristic_attempts(blocksizes)

    # How many temporaries are allocated on the stack?
    # Will drop block sizes that might lead to a stack overflow
    functions = FindSymbols('symbolics').visit(operator.body +
                                               operator.elemental_functions)
    stack_shapes = [i.shape for i in functions if i.is_TensorFunction and i._mem_stack]
    stack_space = sum(reduce(mul, i, 1) for i in stack_shapes)*operator.dtype().itemsize

    # Note: there is only a single loop over 'blocksize' because only
    # square blocks are tested
    timings = OrderedDict()
    for bs in blocksizes:
        illegal = False
        for k, v in at_arguments.items():
            if k in bs:
                val = bs[k]
                handle = at_arguments.get(mapper[k].original_dim.symbolic_size.name)
                if val <= mapper[k].iteration.end(handle):
                    at_arguments[k] = val
                else:
                    # Block size cannot be larger than actual dimension
                    illegal = True
                    break
            elif k in squeezable:
                at_arguments[k] = options['at_squeezer']
        if illegal:
            continue

        # Make sure we remain within stack bounds, otherwise skip block size
        dim_sizes = {}
        for k, v in at_arguments.items():
            if k in bs:
                dim_sizes[mapper[k].argument.symbolic_size] = bs[k]
            elif k in dim_mapper:
                dim_sizes[dim_mapper[k].symbolic_size] = v
        try:
            bs_stack_space = stack_space.xreplace(dim_sizes)
        except AttributeError:
            bs_stack_space = stack_space
        try:
            if int(bs_stack_space) > options['at_stack_limit']:
                continue
        except TypeError:
            # We should never get here
            info_at("Couldn't determine stack size, skipping block size %s" % str(bs))
            continue

        # Add profiler structs
        at_arguments.update(operator._extra_arguments())

        operator.cfunction(*list(at_arguments.values()))
        elapsed = sum(operator.profiler.timings.values())
        timings[tuple(bs.items())] = elapsed
        info_at("<%s>: %f" % (','.join('%d' % i for i in bs.values()), elapsed))

    try:
        best = dict(min(timings, key=timings.get))
        info("Auto-tuned block shape: %s" % best)
    except ValueError:
        info("Auto-tuning request, but couldn't find legal block sizes")
        return arguments

    # Build the new argument list
    tuned = OrderedDict()
    for k, v in arguments.items():
        tuned[k] = best[k] if k in mapper else v

    return tuned


def more_heuristic_attempts(blocksizes):
    handle = []

    for blocksize in blocksizes[:3]:
        for i in blocksizes:
            handle.append(OrderedDict(list(blocksize.items())[:-1] +
                                      [list(i.items())[-1]]))

    for blocksize in list(blocksizes):
        ncombs = len(blocksize)
        for i in range(ncombs):
            for j in combinations(blocksize, i+1):
                item = [(k, blocksize[k]*2 if k in j else v)
                        for k, v in blocksize.items()]
                handle.append(OrderedDict(item))

    unique = []
    for i in blocksizes + handle:
        if i not in unique:
            unique.append(i)

    return unique


options = {
    'at_squeezer': 3,
    'at_blocksize': [8, 16, 24, 32, 40, 64, 128],
    'at_stack_limit': resource.getrlimit(resource.RLIMIT_STACK)[0] / 4
}
"""Autotuning options."""

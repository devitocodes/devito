from collections import OrderedDict
from itertools import combinations

from devito.logger import info, info_at
from devito.nodes import Iteration
from devito.visitors import FindNodes

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
    squeezable = [i.dim.parent.name for i in iterations
                  if i.is_Sequential and i.dim.is_Buffered]

    # Attempted block sizes
    mapper = OrderedDict([(i.argument.name, i) for i in tunable])
    blocksizes = [OrderedDict([(i, v) for i in mapper])
                  for v in options['at_blocksize']]
    if mode == 'aggressive':
        blocksizes = more_heuristic_attempts(blocksizes)

    # Note: there is only a single loop over 'blocksize' because only
    # square blocks are tested
    timings = OrderedDict()
    for blocksize in blocksizes:
        illegal = False
        for k, v in at_arguments.items():
            if k in blocksize:
                val = blocksize[k]
                handle = at_arguments.get(mapper[k].original_dim.name)
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

        # Add profiler structs
        at_arguments.update(operator._extra_arguments())

        operator.cfunction(*list(at_arguments.values()))
        elapsed = sum(operator.profiler.timings.values())
        timings[tuple(blocksize.items())] = elapsed
        info_at("<%s>: %f" %
                (','.join('%d' % i for i in blocksize.values()), elapsed))

    best = dict(min(timings, key=timings.get))
    info('Auto-tuned block shape: %s' % best)

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
    'at_blocksize': [8, 16, 24, 32, 40, 64, 128]
}
"""Autotuning options."""

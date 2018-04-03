from __future__ import absolute_import

from collections import OrderedDict
from itertools import combinations
from functools import reduce
from operator import mul
import resource

from devito.ir.iet import Iteration, FindNodes, FindSymbols
from devito.logger import info, info_at
from devito.parameters import configuration

__all__ = ['autotune']


def autotune(operator, arguments, tunable):
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

    iterations = FindNodes(Iteration).visit(operator.body)
    dim_mapper = {i.dim.name: i.dim for i in iterations}

    # Shrink the iteration space of time-stepping dimension so that auto-tuner
    # runs will finish quickly
    steppers = [i for i in iterations if i.dim.is_Time]
    if len(steppers) == 0:
        timesteps = 1
    elif len(steppers) == 1:
        stepper = steppers[0]
        start = at_arguments[stepper.dim.min_name]
        timesteps = stepper.extent(start=start, finish=options['at_squeezer']) - 1
        if timesteps < 0:
            timesteps = options['at_squeezer'] - timesteps
            info_at("Adjusted auto-tuning timestep to %d" % timesteps)
        at_arguments[stepper.dim.min_name] = start
        at_arguments[stepper.dim.max_name] = timesteps
        if stepper.dim.is_Stepping:
            at_arguments[stepper.dim.parent.min_name] = start
            at_arguments[stepper.dim.parent.max_name] = timesteps
    else:
        info_at("Couldn't understand loop structure, giving up auto-tuning")
        return arguments

    # Attempted block sizes ...
    mapper = OrderedDict([(i.argument.symbolic_size.name, i) for i in tunable])
    # ... Defaults (basic mode)
    blocksizes = [OrderedDict([(i, v) for i in mapper]) for v in options['at_blocksize']]
    # ... Always try the entire iteration space (degenerate block)
    datashape = [at_arguments[mapper[i].original_dim.symbolic_end.name] -
                 at_arguments[mapper[i].original_dim.symbolic_start.name] for i in mapper]
    blocksizes.append(OrderedDict([(i, mapper[i].iteration.extent(0, j))
                      for i, j in zip(mapper, datashape)]))
    # ... More attempts if auto-tuning in aggressive mode
    if configuration.core['autotuning'] == 'aggressive':
        blocksizes = more_heuristic_attempts(blocksizes)

    # How many temporaries are allocated on the stack?
    # Will drop block sizes that might lead to a stack overflow
    functions = FindSymbols('symbolics').visit(operator.body +
                                               operator.elemental_functions)
    stack_shapes = [i.symbolic_shape for i in functions if i.is_Array and i._mem_stack]
    stack_space = sum(reduce(mul, i, 1) for i in stack_shapes)*operator._dtype().itemsize

    # Note: there is only a single loop over 'blocksize' because only
    # square blocks are tested
    timings = OrderedDict()
    for bs in blocksizes:
        illegal = False
        for k, v in at_arguments.items():
            if k in bs:
                val = bs[k]
                start = at_arguments[mapper[k].original_dim.symbolic_start.name]
                end = at_arguments[mapper[k].original_dim.symbolic_end.name]
                if val <= mapper[k].iteration.extent(start, end):
                    at_arguments[k] = val
                else:
                    # Block size cannot be larger than actual dimension
                    illegal = True
                    break
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

        # Use AT-specific profiler structs
        timer = operator.profiler.new()
        at_arguments[operator.profiler.name] = timer

        operator.cfunction(*list(at_arguments.values()))
        elapsed = sum(getattr(timer._obj, i) for i, _ in timer._obj._fields_)
        timings[tuple(bs.items())] = elapsed
        info_at("Block shape <%s> took %f (s) in %d time steps" %
                (','.join('%d' % i for i in bs.values()), elapsed, timesteps))

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

    # Reset the profiling struct
    assert operator.profiler.name in tuned
    tuned[operator.profiler.name] = operator.profiler.new()

    return tuned


def more_heuristic_attempts(blocksizes):
    # Ramp up to higher block sizes
    handle = OrderedDict([(i, options['at_blocksize'][-1]) for i in blocksizes[0]])
    for i in range(3):
        new_bs = OrderedDict([(k, v*2) for k, v in handle.items()])
        blocksizes.insert(blocksizes.index(handle) + 1, new_bs)
        handle = new_bs

    handle = []
    # Extended shuffling for the smaller block sizes
    for bs in blocksizes[:4]:
        for i in blocksizes:
            handle.append(OrderedDict(list(bs.items())[:-1] + [list(i.items())[-1]]))
    # Some more shuffling for all block sizes
    for bs in list(blocksizes):
        ncombs = len(bs)
        for i in range(ncombs):
            for j in combinations(bs, i+1):
                item = [(k, bs[k]*2 if k in j else v) for k, v in bs.items()]
                handle.append(OrderedDict(item))

    unique = []
    for i in blocksizes + handle:
        if i not in unique:
            unique.append(i)

    return unique


options = {
    'at_squeezer': 4,
    'at_blocksize': sorted({8, 16, 24, 32, 40, 64, 128}),
    'at_stack_limit': resource.getrlimit(resource.RLIMIT_STACK)[0] / 4
}
"""Autotuning options."""

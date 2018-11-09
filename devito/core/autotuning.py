from __future__ import absolute_import

from collections import OrderedDict
from itertools import combinations
from functools import reduce
from operator import mul
import resource

from devito.ir import Backward, Iteration, FindNodes, FindSymbols
from devito.logger import perf, warning
from devito.parameters import configuration

__all__ = ['autotune']


def autotune(operator, arguments, level, mode):
    """
    Operator autotuning.

    Parameters
    ----------
    operator : Operator
        Input Operator.
    arguments : dict_like
        The runtime arguments with which `operator` is run.
    level : str
        The autotuning aggressiveness (basic, aggressive). A more aggressive
        autotuning might eventually result in higher performance, though in
        some circumstances it might instead increase the actual runtime.
    mode : str
        The autotuning mode (preemptive, runtime). In preemptive mode, the
        output runtime values supplied by the user to `operator.apply` are
        replaced with shadow copies.
    """
    key = [level, mode]
    accepted = configuration._accepted['autotuning']
    if key not in accepted:
        raise ValueError("The accepted `(level, mode)` combinations are `%s`; "
                         "provided `%s` instead" % (accepted, key))

    parameters = operator.parameters
    tunable = operator._dle_args

    # We get passed all the arguments, but the cfunction only requires a subset
    at_arguments = OrderedDict([(p.name, arguments[p.name]) for p in parameters])

    # User-provided output data won't be altered in `preemptive` mode
    if mode == 'preemptive':
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
        stepper = None
        timesteps = 1
    elif len(steppers) == 1:
        stepper = steppers[0]
        timesteps = init_time_bounds(stepper, at_arguments)
        if timesteps is None:
            return arguments
    else:
        warning("AutoTuner: Couldn't understand loop structure; giving up")
        return arguments

    # Attempted block sizes ...
    mapper = OrderedDict([(i.argument.symbolic_size.name, i) for i in tunable])
    # ... Defaults (basic mode)
    blocksizes = [OrderedDict([(i, v) for i in mapper]) for v in options['at_blocksize']]
    # ... Always try the entire iteration space (degenerate block)
    itershape = [mapper[i].iteration.symbolic_extent.subs(arguments) for i in mapper]
    blocksizes.append(OrderedDict([(i, mapper[i].iteration.extent(0, j-1))
                      for i, j in zip(mapper, itershape)]))
    # ... More attempts if auto-tuning in aggressive mode
    if level == 'aggressive':
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
        # Can we safely autotune over the given time range?
        check_time_bounds(stepper, at_arguments, arguments)

        illegal = False
        for k, v in at_arguments.items():
            if k in bs:
                val = bs[k]
                start = mapper[k].original_dim.symbolic_start.subs(arguments)
                end = mapper[k].original_dim.symbolic_end.subs(arguments)

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
            warning("AutoTuner: Couldn't determine stack size; skipping block shape %s"
                    % str(bs))
            continue

        # Use AutoTuner-specific profiler structs
        timer = operator.profiler.timer.reset()
        at_arguments[operator.profiler.name] = timer

        operator.cfunction(*list(at_arguments.values()))
        elapsed = sum(getattr(timer._obj, i) for i, _ in timer._obj._fields_)
        timings[tuple(bs.items())] = elapsed
        perf("AutoTuner: Block shape <%s> took %f (s) in %d timesteps" %
             (','.join('%d' % i for i in bs.values()), elapsed, timesteps))

        # Prepare for the next autotuning run
        update_time_bounds(stepper, at_arguments, timesteps, mode)

    try:
        best = dict(min(timings, key=timings.get))
        perf("AutoTuner: Selected block shape %s" % best)
    except ValueError:
        warning("AutoTuner: Couldn't find legal block shapes")
        return arguments

    # Build the new argument list
    arguments = {k: best.get(k, v) for k, v in arguments.items()}

    # In `runtime` mode, some timesteps have been executed already, so we
    # get to adjust the time range
    finalize_time_bounds(stepper, at_arguments, arguments, mode)

    # Reset profiling data
    assert operator.profiler.name in arguments
    arguments[operator.profiler.name] = operator.profiler.timer.reset()

    return arguments


def init_time_bounds(stepper, at_arguments):
    if stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        at_arguments[dim.max_name] = at_arguments[dim.max_name]
        at_arguments[dim.min_name] = at_arguments[dim.max_name] - options['at_squeezer']
        if at_arguments[dim.max_name] < at_arguments[dim.min_name]:
            warning("AutoTuner: too few time iterations; giving up")
            return False
    else:
        at_arguments[dim.min_name] = at_arguments[dim.min_name]
        at_arguments[dim.max_name] = at_arguments[dim.min_name] + options['at_squeezer']
        if at_arguments[dim.min_name] > at_arguments[dim.max_name]:
            warning("AutoTuner: too few time iterations; giving up")
            return False

    return stepper.extent(start=at_arguments[dim.min_name], finish=at_arguments[dim.max_name])


def check_time_bounds(stepper, at_arguments, arguments):
    if stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        if at_arguments[dim.min_name] < arguments[dim.min_name]:
            raise ValueError("Too few time iterations")

    else:
        if at_arguments[dim.max_name] > arguments[dim.max_name]:
            raise ValueError("Too few time iterations")


def update_time_bounds(stepper, at_arguments, timesteps, mode):
    if mode != 'runtime' or stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        at_arguments[dim.max_name] -= timesteps
        at_arguments[dim.min_name] -= timesteps
    else:
        at_arguments[dim.min_name] += timesteps
        at_arguments[dim.max_name] += timesteps


def finalize_time_bounds(stepper, at_arguments, arguments, mode):
    if mode != 'runtime' or stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        arguments[dim.max_name] = at_arguments[dim.max_name]
        arguments[dim.min_name] = arguments[dim.min_name]
    else:
        arguments[dim.min_name] = at_arguments[dim.min_name]
        arguments[dim.max_name] = arguments[dim.max_name]


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

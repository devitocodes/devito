from collections import OrderedDict
from itertools import chain, combinations, product
import resource
import psutil

from devito.dle import BlockDimension, NThreads
from devito.ir import Backward, retrieve_iteration_tree
from devito.logger import perf, warning as _warning
from devito.mpi import MPI
from devito.parameters import configuration
from devito.symbolics import evaluate
from devito.tools import filter_ordered, flatten, prod

__all__ = ['autotune']


def autotune(operator, args, level, mode):
    """
    Operator autotuning.

    Parameters
    ----------
    operator : Operator
        Input Operator.
    args : dict_like
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

    # Tunable objects
    blockable = [i for i in operator.dimensions if isinstance(i, BlockDimension)]
    nthreads = [i for i in operator.input if isinstance(i, NThreads)]

    if len(nthreads + blockable) == 0:
        # Nothing to tune for
        return args, {}

    # We get passed all the arguments, but the cfunction only requires a subset
    at_args = OrderedDict([(p.name, args[p.name]) for p in operator.parameters])

    # User-provided output data won't be altered in `preemptive` mode
    if mode == 'preemptive':
        output = [i.name for i in operator.output]
        for k, v in args.items():
            if k in output:
                at_args[k] = v.copy()

    # Disable halo exchanges as the number of autotuning steps performed on each
    # rank may be different. Also, this makes the autotuning runtimes reliable
    # regardless of whether the timed regions include the halo exchanges or not,
    # as now the halo exchanges become a no-op.
    try:
        nb = []
        if mode != 'runtime':
            for i, _ in at_args['nb']._obj._fields_:
                nb.append((i, getattr(at_args['nb']._obj, i)))
                setattr(at_args['nb']._obj, i, MPI.PROC_NULL)
    except KeyError:
        assert not configuration['mpi']

    trees = retrieve_iteration_tree(operator.body)

    # Shrink the time dimension's iteration range for quick autotuning
    steppers = {i for i in flatten(trees) if i.dim.is_Time}
    if len(steppers) == 0:
        stepper = None
        timesteps = 1
    elif len(steppers) == 1:
        stepper = steppers.pop()
        timesteps = init_time_bounds(stepper, at_args)
        if timesteps is None:
            return args, {}
    else:
        warning("cannot perform autotuning unless there is one time loop; skipping")
        return args, {}

    # Formula to calculate the number of parallel blocks given block shape,
    # number of threads, and extent of the parallel iteration space
    calculate_parblocks = make_calculate_parblocks(trees, blockable, nthreads)

    # Generated loop-blocking attempts
    block_shapes = generate_block_shapes(blockable, args, level)

    # Generate nthreads attempts
    nthreads = generate_nthreads(nthreads, args, level)

    generators = [i for i in [block_shapes, nthreads] if i]

    timings = OrderedDict()
    for i in product(*generators):
        run = tuple(chain(*i))
        mapper = OrderedDict(run)

        # Can we safely autotune over the given time range?
        if not check_time_bounds(stepper, at_args, args, mode):
            break

        # Update `at_args` to use the new tunable values
        at_args = {k: mapper.get(k, v) for k, v in at_args.items()}

        if heuristically_discard_run(calculate_parblocks, at_args):
            continue

        # Make sure we remain within stack bounds, otherwise skip run
        try:
            stack_footprint = operator._mem_summary['stack']
            if int(evaluate(stack_footprint, **at_args)) > options['stack_limit']:
                continue
        except TypeError:
            warning("couldn't determine stack size; skipping run %s" % str(i))
            continue
        except AttributeError:
            assert stack_footprint == 0

        # Use fresh profiling data
        timer = operator._profiler.timer.reset()
        at_args[operator._profiler.name] = timer

        operator.cfunction(*list(at_args.values()))
        elapsed = sum(getattr(timer._obj, k) for k, _ in timer._obj._fields_)
        timings[run] = elapsed
        log("run <%s> took %f (s) in %d timesteps" %
            (','.join('%s=%s' % (k, v) for k, v in mapper.items()), elapsed, timesteps))

        # Prepare for the next autotuning run
        update_time_bounds(stepper, at_args, timesteps, mode)

    try:
        best = dict(min(timings, key=timings.get))
        log("selected best: %s" % best)
    except ValueError:
        warning("couldn't perform any runs")
        return args, {}

    # Build the new argument list
    args = {k: best.get(k, v) for k, v in args.items()}

    # In `runtime` mode, some timesteps have been executed already, so we
    # get to adjust the time range
    finalize_time_bounds(stepper, at_args, args, mode)

    # Reset profiling data
    assert operator._profiler.name in args
    args[operator._profiler.name] = operator._profiler.timer.reset()

    # Reinstate MPI neighbourhood
    for i, v in nb:
        setattr(args['nb']._obj, i, v)

    # Autotuning summary
    summary = {}
    summary['runs'] = len(timings)
    summary['tpr'] = timesteps  # tpr -> timesteps per run
    summary['tuned'] = dict(best)

    return args, summary


def init_time_bounds(stepper, at_args):
    if stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        at_args[dim.min_name] = at_args[dim.max_name] - options['squeezer']
        if at_args[dim.max_name] < at_args[dim.min_name]:
            warning("too few time iterations; skipping")
            return False
    else:
        at_args[dim.max_name] = at_args[dim.min_name] + options['squeezer']
        if at_args[dim.min_name] > at_args[dim.max_name]:
            warning("too few time iterations; skipping")
            return False

    return stepper.extent(start=at_args[dim.min_name], finish=at_args[dim.max_name])


def check_time_bounds(stepper, at_args, args, mode):
    if mode != 'runtime' or stepper is None:
        return True
    dim = stepper.dim.root
    if stepper.direction is Backward:
        if at_args[dim.min_name] < args[dim.min_name]:
            warning("too few time iterations; stopping")
            return False
    else:
        if at_args[dim.max_name] > args[dim.max_name]:
            warning("too few time iterations; stopping")
            return False
    return True


def update_time_bounds(stepper, at_args, timesteps, mode):
    if mode != 'runtime' or stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        at_args[dim.max_name] -= timesteps
        at_args[dim.min_name] -= timesteps
    else:
        at_args[dim.min_name] += timesteps
        at_args[dim.max_name] += timesteps


def finalize_time_bounds(stepper, at_args, args, mode):
    if mode != 'runtime' or stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        args[dim.max_name] = at_args[dim.max_name]
        args[dim.min_name] = args[dim.min_name]
    else:
        args[dim.min_name] = at_args[dim.min_name]
        args[dim.max_name] = args[dim.max_name]


def make_calculate_parblocks(trees, blockable, nthreads):
    blocks_per_threads = []
    main_block_trees = [i for i in trees if set(blockable) < set(i.dimensions)]
    for tree, nt in product(main_block_trees, nthreads):
        block_iters = [i for i in tree if i.dim in blockable]
        par_block_iters = block_iters[:block_iters[0].ncollapsed]
        niterations = prod(i.extent() for i in par_block_iters)
        block_size = prod(i.dim.step for i in par_block_iters)
        blocks_per_threads.append((niterations / block_size) / nt)
    return blocks_per_threads


def generate_block_shapes(blockable, args, level):
    # Max attemptable block shape
    max_bs = tuple((d.step.name, d.max_step.subs(args)) for d in blockable)

    # Attempted block shapes:
    # 1) Defaults (basic mode)
    ret = [tuple((d.step.name, v) for d in blockable) for v in options['blocksize']]
    # 2) Always try the entire iteration space (degenerate block)
    ret.append(max_bs)
    # 3) More attempts if auto-tuning in aggressive mode
    if level == 'aggressive':
        # Ramp up to larger block shapes
        handle = tuple((i, options['blocksize'][-1]) for i, _ in ret[0])
        for i in range(3):
            new_bs = tuple((b, v*2) for b, v in handle)
            ret.insert(ret.index(handle) + 1, new_bs)
            handle = new_bs

        handle = []
        # Extended shuffling for the smaller block shapes
        for bs in ret[:4]:
            for i in ret:
                handle.append(bs[:-1] + (i[-1],))
        # Some more shuffling for all block shapes
        for bs in list(ret):
            ncombs = len(bs)
            for i in range(ncombs):
                for j in combinations(dict(bs), i+1):
                    handle.append(tuple((b, v*2 if b in j else v) for b, v in bs))
        ret.extend(handle)

    # Drop unnecessary attempts:
    # 1) Block shapes exceeding the iteration space extent
    ret = [i for i in ret if all(dict(i)[k] <= v for k, v in max_bs)]
    # 2) Redundant block shapes
    ret = filter_ordered(ret)

    return ret


def generate_nthreads(nthreads, args, level):
    ret = [((i.name, args[i.name]),) for i in nthreads]

    # On the KNL, also try running with a different number of hyperthreads
    if level == 'aggressive' and configuration['platform'] == 'knl':
        ret.extend([((i.name, psutil.cpu_count()),) for i in nthreads])
        ret.extend([((i.name, psutil.cpu_count() // 2),) for i in nthreads])
        ret.extend([((i.name, psutil.cpu_count() // 4),) for i in nthreads])

    return filter_ordered(ret)


def heuristically_discard_run(calculate_parblocks, at_args):
    if configuration['develop-mode']:
        return False
    # Drop run if not at least one block per thread
    return all(i.subs(at_args) < 1 for i in calculate_parblocks)


options = {
    'squeezer': 4,
    'blocksize': sorted({8, 16, 24, 32, 40, 64, 128}),
    'stack_limit': resource.getrlimit(resource.RLIMIT_STACK)[0] / 4
}
"""Autotuning options."""


def log(msg):
    perf("AutoTuner: %s" % msg)


def warning(msg):
    _warning("AutoTuner: %s" % msg)

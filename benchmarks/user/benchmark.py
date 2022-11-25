import numpy as np
import click
import os

from devito import Device, configuration, info, warning, set_log_level, switchconfig, norm
from devito.arch.compiler import IntelCompiler
from devito.mpi import MPI
from devito.operator.profiling import PerformanceSummary
from devito.tools import all_equal, as_tuple, sweep
from devito.types.dense import DiscreteFunction

from examples.seismic.acoustic.acoustic_example import run as acoustic_run, acoustic_setup
from examples.seismic.tti.tti_example import run as tti_run, tti_setup
from examples.seismic.elastic.elastic_example import run as elastic_run, elastic_setup
from examples.seismic.self_adjoint.example_iso import run as acoustic_sa_run, \
    acoustic_sa_setup
from examples.seismic.viscoelastic.viscoelastic_example import run as viscoelastic_run, \
    viscoelastic_setup


model_type = {
    'viscoelastic': {
        'run': viscoelastic_run,
        'setup': viscoelastic_setup,
        'default-section': 'global'
    },
    'elastic': {
        'run': elastic_run,
        'setup': elastic_setup,
        'default-section': 'global'
    },
    'tti': {
        'run': tti_run,
        'setup': tti_setup,
        'default-section': 'global'
    },
    'acoustic': {
        'run': acoustic_run,
        'setup': acoustic_setup,
        'default-section': 'global'
    },
    'acoustic_sa': {
        'run': acoustic_sa_run,
        'setup': acoustic_sa_setup,
        'default-section': 'global'
    }
}


class NTuple(click.Tuple):
    """
    A floating subtype of click's Tuple that allows inputs with fewer elements.
    Instead of accepting only tuples of exact length, this accepts tuples
    of length up to the definition size.
    For example, NTuple([int, int, int]) accepts (1,), (1, 2) and (1, 2, 3) as inputs.
    """
    def convert(self, value, param, ctx):
        n_value = len(value)
        n_type = len(self.types)
        if n_value <= n_type:
            warning(f"Processing {n_value} out of expected up to {n_type}")
        else:
            super().convert(value, param, ctx)
        return tuple(self.types[i](value[i], param, ctx) for i in range(n_value))


def run_op(solver, operator, **options):
    """
    Initialize any necessary input and run the operator associated with the solver.
    """
    # Get the operator if exist
    try:
        op = getattr(solver, operator)
    except AttributeError:
        raise AttributeError("Operator %s not implemented for %s" % (operator, solver))

    # This is a bit ugly but not sure how to make clean input creation for different op
    if operator == "forward":
        return op(**options)
    elif operator == "adjoint":
        rec = solver.geometry.adj_src
        return op(rec, **options)
    elif operator == "jacobian":
        dm = solver.model.dm
        # Because sometime dm is zero, artificially add a non zero slice
        if dm.data.min() == 0 and dm.data.max() == 0:
            dm.data[..., np.min([25, dm.shape_global[-1]//4])] = .1
        return op(dm, **options)
    elif operator == "jacobian_adjoint":
        # I think we want the forward + gradient call, need to merge retvals
        args = solver.forward(save=True, **options)
        assert isinstance(args[-1], PerformanceSummary)
        args = args[:-1]
        return op(*args, **options)
    else:
        raise ValueError("Unrecognized operator %s" % operator)


@click.group()
def benchmark():
    """
    Benchmarking script for seismic operators.

    \b
    There are three main 'execution modes':
    run: a single run with given optimization level
    run-jit-backdoor: a single run using the DEVITO_JIT_BACKDOOR to
                      experiment with manual customizations
    test: tests numerical correctness with different parameters

    Further, this script can generate a roofline plot from a benchmark
    """
    pass


def option_simulation(f):
    def default_list(ctx, param, value):
        return list(value if len(value) > 0 else (2, ))

    options = [
        click.option('-P', '--problem', help='Problem name',
                     type=click.Choice(['acoustic', 'tti',
                                        'elastic', 'acoustic_sa', 'viscoelastic'])),
        click.option('-d', '--shape', default=(50, 50, 50), type=NTuple([int, int, int]),
                     help='Number of grid points along each axis'),
        click.option('-s', '--spacing', default=(20., 20., 20.),
                     type=NTuple([float, float, float]),
                     help='Spacing between grid sizes in meters'),
        click.option('-n', '--nbl', default=10,
                     help='Number of boundary layers'),
        click.option('-so', '--space-order', type=int, multiple=True,
                     callback=default_list, help='Space order of the simulation'),
        click.option('-to', '--time-order', type=int, multiple=True,
                     callback=default_list, help='Time order of the simulation'),
        click.option('-t', '--tn', default=250,
                     help='End time of the simulation in ms'),
        click.option('-op', '--operator', default='forward', help='Operator to run',
                     type=click.Choice(['forward', 'adjoint',
                                        'jacobian', 'jacobian_adjoint']))]
    for option in reversed(options):
        f = option(f)
    return f


def option_performance(f):
    """Defines options for all aspects of performance tuning"""

    def from_value(ctx, param, value):
        """Prefer preset values and warn for competing values."""
        return ctx.params[param.name] or value

    def from_opt(ctx, param, value):
        """Process the opt argument."""
        try:
            # E.g., `('advanced', {'par-tile': True})`
            value = eval(value)
            if not isinstance(value, tuple) and len(value) >= 1:
                raise click.BadParameter("Invalid choice `%s` (`opt` must be "
                                         "either str or tuple)" % str(value))
            opt = value[0]
        except NameError:
            # E.g. `'advanced'`
            opt = value
        if opt not in configuration._accepted['opt']:
            raise click.BadParameter("Invalid choice `%s` (choose from %s)"
                                     % (opt, str(configuration._accepted['opt'])))
        return value

    def config_blockshape(ctx, param, value):
        if isinstance(configuration['platform'], Device):
            normalized_value = []
        elif value:
            # Block innermost loops if a full block shape is provided
            # Note: see https://github.com/devitocodes/devito/issues/320 for why
            # we use blockinner=True only if the backend compiler is Intel
            flag = isinstance(configuration['compiler'], IntelCompiler)
            configuration['opt-options']['blockinner'] = flag
            # Normalize value:
            # 1. integers, not strings
            # 2. sanity check the (hierarchical) blocking shape
            normalized_value = []
            for i, block_shape in enumerate(value):
                # If hierarchical blocking is activated, say with N levels, here in
                # `bs` we expect to see 3*N entries
                bs = [int(x) for x in block_shape.split()]
                levels = [bs[x:x+3] for x in range(0, len(bs), 3)]
                if any(len(level) != 3 for level in levels):
                    raise ValueError("Expected 3 entries per block shape level, but got "
                                     "one level with less than 3 entries (`%s`)" % levels)
                normalized_value.append(levels)
            if not all_equal(len(i) for i in normalized_value):
                raise ValueError("Found different block shapes with incompatible "
                                 "number of levels (`%s`)" % normalized_value)
            configuration['opt-options']['blocklevels'] = len(normalized_value[0])
        else:
            normalized_value = []
        return tuple(normalized_value)

    def config_autotuning(ctx, param, value):
        """Setup auto-tuning to run in ``{basic,aggressive,...}+preemptive`` mode."""
        if isinstance(configuration['platform'], Device):
            level = False
        elif value != 'off':
            # Sneak-peek at the `block-shape` -- if provided, keep auto-tuning off
            if ctx.params['block_shape']:
                warning("Skipping autotuning (using explicit block-shape `%s`)"
                        % str(ctx.params['block_shape']))
                level = False
            else:
                # Make sure to always run in preemptive mode
                configuration['autotuning'] = [value, 'preemptive']
                # We apply blocking to all parallel loops, including the innermost ones
                # Note: see https://github.com/devitocodes/devito/issues/320 for why
                # we use blockinner=True only if the backend compiler is Intel
                flag = isinstance(configuration['compiler'], IntelCompiler)
                configuration['opt-options']['blockinner'] = flag
                level = value
        else:
            level = False
        return level

    options = [
        click.option('--arch', default='unknown',
                     help='Architecture on which the simulation is/was run'),
        click.option('--opt', callback=from_opt, default='advanced',
                     help='Performance optimization level'),
        click.option('-bs', '--block-shape', callback=config_blockshape, multiple=True,
                     is_eager=True, help='Loop-blocking shape, bypass autotuning'),
        click.option('-a', '--autotune', default='aggressive', callback=config_autotuning,
                     type=click.Choice([str(tuple(i)) if type(i) is list else i
                                        for i in configuration._accepted['autotuning']]),
                     help='Select autotuning mode')
    ]
    for option in reversed(options):
        f = option(f)
    return f


@benchmark.command(name='run')
@option_simulation
@option_performance
@click.option('--warmup', is_flag=True, default=False,
              help='Perform a preliminary run to warm up the system')
@click.option('--dump-summary', default=False,
              help='File where the performance results are saved')
@click.option('--dump-norms', default=False,
              help='File where the output norms are saved')
def cli_run(problem, **kwargs):
    """`click` interface for the `run` mode."""
    configuration['develop-mode'] = False

    run(problem, **kwargs)


def run(problem, **kwargs):
    """
    A single run with a specific set of performance parameters.
    """
    setup = model_type[problem]['setup']
    options = {}

    time_order = kwargs.pop('time_order')[0]
    space_order = kwargs.pop('space_order')[0]
    autotune = kwargs.pop('autotune')
    options['autotune'] = autotune
    block_shapes = as_tuple(kwargs.pop('block_shape'))
    operator = kwargs.pop('operator', 'forward')
    warmup = kwargs.pop('warmup')

    # Should a specific block-shape be used? Useful if one wants to skip
    # the autotuning pass as a good block-shape is already known
    # Note: the following piece of code is horribly *hacky*, but it works for now
    for i, block_shape in enumerate(block_shapes):
        for n, level in enumerate(block_shape):
            for d, s in zip(['x', 'y', 'z'], level):
                options['%s%d_blk%d_size' % (d, i, n)] = s

    solver = setup(space_order=space_order, time_order=time_order, **kwargs)
    if warmup:
        info("Performing warm-up run ...")
        set_log_level('ERROR', comm=MPI.COMM_WORLD)
        run_op(solver, operator, **options)
        set_log_level('DEBUG', comm=MPI.COMM_WORLD)
        info("DONE!")
    retval = run_op(solver, operator, **options)

    try:
        rank = MPI.COMM_WORLD.rank
    except AttributeError:
        # MPI not available
        rank = 0

    dumpfile = kwargs.pop('dump_summary')
    if dumpfile:
        if configuration['profiling'] != 'advanced':
            raise RuntimeError("Must set DEVITO_PROFILING=advanced (or, alternatively, "
                               "DEVITO_LOGGING=PERF) with --dump-summary")
        if rank == 0:
            with open(dumpfile, 'w') as f:
                summary = retval[-1]
                assert isinstance(summary, PerformanceSummary)
                f.write(str(summary.globals_all))

    dumpfile = kwargs.pop('dump_norms')
    if dumpfile:
        norms = ["'%s': %f" % (i.name, norm(i)) for i in retval[:-1]
                 if isinstance(i, DiscreteFunction)]
        if rank == 0:
            with open(dumpfile, 'w') as f:
                f.write("{%s}" % ', '.join(norms))

    return retval


@benchmark.command(name='run-jit-backdoor')
@option_simulation
@option_performance
@click.option('--dump-norms', is_flag=True, default=False,
              help='Display norms of written fields')
def cli_run_jit_backdoor(problem, **kwargs):
    """`click` interface for the `run_jit_backdoor` mode."""
    run_jit_backdoor(problem, **kwargs)


def run_jit_backdoor(problem, **kwargs):
    """
    A single run using the DEVITO_JIT_BACKDOOR to test kernel customization.
    """
    configuration['develop-mode'] = False

    setup = model_type[problem]['setup']

    time_order = kwargs.pop('time_order')[0]
    space_order = kwargs.pop('space_order')[0]
    autotune = kwargs.pop('autotune')

    info("Preparing simulation...")
    solver = setup(space_order=space_order, time_order=time_order, **kwargs)

    # Generate code (but do not JIT yet)
    op = solver.op_fwd()

    # Get the filename in the JIT cache
    cfile = "%s.c" % str(op._compiler.get_jit_dir().joinpath(op._soname))

    if not os.path.exists(cfile):
        # First time we run this problem, let's generate and jit-compile code
        op.cfunction
        info("You may now edit the generated code in `%s`. "
             "Then save the file, and re-run this benchmark." % cfile)
        return

    info("Running wave propagation Operator...")

    @switchconfig(jit_backdoor=True)
    def _run_jit_backdoor():
        return run_op(solver, 'forward', autotune=autotune)

    retval = _run_jit_backdoor()

    dumpnorms = kwargs.pop('dump_norms')
    if dumpnorms:
        for i in retval[:-1]:
            if isinstance(i, DiscreteFunction):
                info("'%s': %f" % (i.name, norm(i)))

    return retval


@benchmark.command(name='test')
@option_simulation
@option_performance
def cli_test(problem, **kwargs):
    """`click` interface for the `test` mode."""
    set_log_level('ERROR')

    test(problem, **kwargs)


def test(problem, **kwargs):
    """
    Test numerical correctness with different parameters.
    """
    run = model_type[problem]['run']
    sweep_options = ('space_order', 'time_order', 'opt', 'autotune')

    last_res = None
    for params in sweep(kwargs, keys=sweep_options):
        kwargs.update(params)
        _, _, _, res = run(**kwargs)

        if last_res is None:
            last_res = res
        else:
            for i in range(len(res)):
                assert np.isclose(res[i], last_res[i])


if __name__ == "__main__":
    # If running with MPI, we emit logging messages from rank0 only
    try:
        MPI.Init()  # Devito starts off with MPI disabled!
        set_log_level('DEBUG', comm=MPI.COMM_WORLD)

        if MPI.COMM_WORLD.size > 1 and not configuration['mpi']:
            warning("It seems that you're running over MPI with %d processes, but "
                    "DEVITO_MPI is unset. Setting `DEVITO_MPI=basic`..."
                    % MPI.COMM_WORLD.size)
            configuration['mpi'] = 'basic'
    except TypeError:
        # MPI not available
        pass

    # Benchmarking cannot be done at basic level
    if configuration['profiling'] == 'basic':
        configuration['profiling'] = 'advanced'

    benchmark(standalone_mode=False)

    try:
        MPI.Finalize()
    except TypeError:
        # MPI not available
        pass

from collections import OrderedDict
import sys

import numpy as np
import click
import os
from devito import clear_cache, configuration, warning, set_log_level
from devito.mpi import MPI
from devito.tools import as_tuple, sweep
from examples.seismic.acoustic.acoustic_example import run as acoustic_run, acoustic_setup
from examples.seismic.tti.tti_example import run as tti_run, tti_setup
from examples.seismic.elastic.elastic_example import run as elastic_run, elastic_setup
from examples.seismic.viscoelastic.viscoelastic_example import run as viscoelastic_run, \
    viscoelastic_setup


model_type = {
    'viscoelastic': {
        'run': viscoelastic_run,
        'setup': viscoelastic_setup,
        'default-section': 'section0'
    },
    'elastic': {
        'run': elastic_run,
        'setup': elastic_setup,
        'default-section': 'section0'
    },
    'tti': {
        'run': tti_run,
        'setup': tti_setup,
        'default-section': 'section1'
    },
    'acoustic': {
        'run': acoustic_run,
        'setup': acoustic_setup,
        'default-section': 'section0'
    }
}


@click.group()
def benchmark():
    """
    Benchmarking script for seismic forward operators.

    \b
    There are three main 'execution modes':
    run: a single run with given DSE/DLE levels
    bench: complete benchmark with multiple DSE/DLE levels
    test: tests numerical correctness with different parameters

    Further, this script can generate a roofline plot from a benchmark
    """
    pass


def option_simulation(f):
    def default_list(ctx, param, value):
        return list(value if len(value) > 0 else (2, ))

    options = [
        click.option('-P', '--problem', type=click.Choice(['acoustic', 'tti',
                                                           'elastic', 'viscoelastic']),
                     help='Problem name'),
        click.option('-d', '--shape', default=(50, 50, 50),
                     help='Number of grid points along each axis'),
        click.option('-s', '--spacing', default=(20., 20., 20.),
                     help='Spacing between grid sizes in meters'),
        click.option('-n', '--nbpml', default=10,
                     help='Number of PML layers'),
        click.option('-so', '--space-order', type=int, multiple=True,
                     callback=default_list, help='Space order of the simulation'),
        click.option('-to', '--time-order', type=int, multiple=True,
                     callback=default_list, help='Time order of the simulation'),
        click.option('-t', '--tn', default=250,
                     help='End time of the simulation in ms')
    ]
    for option in reversed(options):
        f = option(f)
    return f


def option_performance(f):
    """Defines options for all aspects of performance tuning"""

    _preset = {
        # Fixed
        'O1': {'dse': 'basic', 'dle': 'advanced'},
        'O2': {'dse': 'advanced', 'dle': 'advanced'},
        'O3': {'dse': 'aggressive', 'dle': 'advanced'},
        # Parametric
        'dse': {'dse': ['basic', 'advanced', 'aggressive'], 'dle': 'advanced'},
    }

    def from_preset(ctx, param, value):
        """Set all performance options according to bench-mode preset"""
        ctx.params.update(_preset[value])
        return value

    def from_value(ctx, param, value):
        """Prefer preset values and warn for competing values."""
        return ctx.params[param.name] or value

    def config_blockshape(ctx, param, value):
        if value:
            # Block innermost loops if a full block shape is provided
            configuration['dle-options']['blockinner'] = True
        return value

    def config_autotuning(ctx, param, value):
        """Setup auto-tuning to run in ``{basic,aggressive,...}+preemptive`` mode."""
        if value != 'off':
            # Sneak-peek at the `block-shape` -- if provided, keep auto-tuning off
            if ctx.params['block_shape']:
                warning("Skipping autotuning (using explicit block-shape `%s`)"
                        % str(ctx.params['block_shape']))
                level = False
            else:
                # Make sure to always run in preemptive mode
                configuration['autotuning'] = [value, 'preemptive']
                # We apply blocking to all parallel loops, including the innermost ones
                configuration['dle-options']['blockinner'] = True
                level = value
        else:
            level = False
        return level

    options = [
        click.option('-bm', '--bench-mode', is_eager=True,
                     callback=from_preset, expose_value=False, default='O2',
                     type=click.Choice(['O1', 'O2', 'O3', 'dse']),
                     help='Choose what to benchmark; ignored if execmode=run'),
        click.option('--arch', default='unknown',
                     help='Architecture on which the simulation is/was run'),
        click.option('--dse', callback=from_value,
                     type=click.Choice(['noop'] + configuration._accepted['dse']),
                     help='Devito symbolic engine (DSE) mode'),
        click.option('--dle', callback=from_value,
                     type=click.Choice(['noop'] + configuration._accepted['dle']),
                     help='Devito loop engine (DLE) mode'),
        click.option('-bs', '--block-shape', callback=config_blockshape, multiple=True,
                     is_eager=True, help='Loop-blocking shape, bypass autotuning'),
        click.option('-a', '--autotune', default='aggressive', callback=config_autotuning,
                     type=click.Choice(configuration._accepted['autotuning']),
                     help='Select autotuning mode')
    ]
    for option in reversed(options):
        f = option(f)
    return f


@benchmark.command(name='run')
@option_simulation
@option_performance
def cli_run(problem, **kwargs):
    """
    A single run with a specific set of performance parameters.
    """
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
    block_shapes = as_tuple(kwargs.pop('block_shape'))

    # Should a specific block-shape be used? Useful if one wants to skip
    # the autotuning pass as a good block-shape is already known
    if block_shapes:
        # The following piece of code is horribly hacky, but it works for now
        for i, block_shape in enumerate(block_shapes):
            bs = [int(x) for x in block_shape.split()]
            # If hierarchical blocking is activated, say with N levels, here in
            # `bs` we expect to see 3*N entries
            levels = [bs[x:x+3] for x in range(0, len(bs), 3)]
            for n, level in enumerate(levels):
                if len(level) != 3:
                    raise ValueError("Expected 3 entries per block shape level, "
                                     "but got one level with only %s entries (`%s`)"
                                     % (len(level), level))
                for d, s in zip(['x', 'y', 'z'], level):
                    options['%s%d_blk%d_size' % (d, i, n)] = s

    solver = setup(space_order=space_order, time_order=time_order, **kwargs)
    solver.forward(autotune=autotune, **options)


@benchmark.command(name='test')
@option_simulation
@option_performance
def cli_test(problem, **kwargs):
    """
    Test numerical correctness with different parameters.
    """
    set_log_level('ERROR')

    test(problem, **kwargs)


def test(problem, **kwargs):
    """
    Test numerical correctness with different parameters.
    """
    run = model_type[problem]['run']
    sweep_options = ('space_order', 'time_order', 'dse', 'dle', 'autotune')

    last_res = None
    for params in sweep(kwargs, keys=sweep_options):
        kwargs.update(params)
        _, _, _, res = run(**kwargs)

        if last_res is None:
            last_res = res
        else:
            for i in range(len(res)):
                assert np.isclose(res[i], last_res[i])


@benchmark.command(name='bench')
@click.option('-r', '--resultsdir', default='results',
              help='Directory containing results')
@click.option('-x', '--repeats', default=3,
              help='Number of test case repetitions')
@option_simulation
@option_performance
def cli_bench(problem, **kwargs):
    """
    Complete benchmark with multiple simulation and performance parameters.
    """
    configuration['develop-mode'] = False

    bench(problem, **kwargs)


def bench(problem, **kwargs):
    """
    Complete benchmark with multiple simulation and performance parameters.
    """
    run = model_type[problem]['run']
    resultsdir = kwargs.pop('resultsdir')
    repeats = kwargs.pop('repeats')

    bench = get_ob_bench(problem, resultsdir, kwargs)
    bench.execute(get_ob_exec(run), warmups=0, repeats=repeats)
    bench.save()

    # Final clean up, just in case the benchmarker is used from external Python modules
    clear_cache()


@benchmark.command(name='plot')
@click.option('--backend', default='core',
              type=click.Choice(configuration._accepted['backend']),
              help='Used execution backend (e.g., core, yask)')
@click.option('-r', '--resultsdir', default='results',
              help='Directory containing results')
@click.option('--max-bw', type=float,
              help='Max GB/s of the DRAM')
@click.option('--flop-ceil', type=(float, str), multiple=True,
              help='Max GFLOPS/s of the CPU. A 2-tuple (float, str)'
                   'is expected, where the float is the performance'
                   'ceil (GFLOPS/s) and the str indicates how the'
                   'ceil was obtained (ideal peak, linpack, ...)')
@click.option('--point-runtime', is_flag=True, default=True,
              help='Annotate points with runtime values')
@click.option('--section', default=None,
              help='Code section for which the roofline is plotted')
@option_simulation
@option_performance
def cli_plot(problem, **kwargs):
    """
    Plotting mode to generate plots for performance analysis.
    """
    plot(problem, **kwargs)


def plot(problem, **kwargs):
    """
    Plotting mode to generate plots for performance analysis.
    """
    backend = kwargs.pop('backend')
    resultsdir = kwargs.pop('resultsdir')
    max_bw = kwargs.pop('max_bw')
    flop_ceils = kwargs.pop('flop_ceil')
    point_runtime = kwargs.pop('point_runtime')
    autotune = kwargs['autotune']
    arch = kwargs['arch']
    space_order = "[%s]" % ",".join(str(i) for i in kwargs['space_order'])
    time_order = kwargs['time_order']
    shape = "[%s]" % ",".join(str(i) for i in kwargs['shape'])

    section = kwargs.pop('section')
    if not section:
        warning("No `section` provided. Using `%s`'s default `%s`"
                % (problem, model_type[problem]['default-section']))
        section = model_type[problem]['default-section']

    RooflinePlotter = get_ob_plotter()
    bench = get_ob_bench(problem, resultsdir, kwargs)

    bench.load()
    if not bench.loaded:
        warning("Could not load any results, nothing to plot. Exiting...")
        sys.exit(0)

    gflopss = bench.lookup(params=kwargs, measure="gflopss", event=section)
    oi = bench.lookup(params=kwargs, measure="oi", event=section)
    time = bench.lookup(params=kwargs, measure="timings", event=section)

    # What plot am I?
    modes = [i for i in ['dse', 'dle', 'autotune']
             if len(set(dict(j)[i] for j in gflopss)) > 1]

    # Filename
    figname = "%s_arch[%s]_shape%s_so%s_to%s_bkend[%s]_at[%s]" % (
        problem, arch, shape, space_order, time_order, backend, autotune
    )

    # Legend setup. Do not plot a legend if there's no variation in performance
    # options (dse, dle, autotune)
    if modes:
        legend = {'loc': 'upper left', 'fontsize': 7, 'ncol': 4}
    else:
        legend = 'drop'

    avail_colors = ['r', 'g', 'b', 'y', 'k', 'm']
    avail_markers = ['o', 'x', '^', 'v', '<', '>']

    used_colors = {}
    used_markers = {}

    # Find min and max runtimes for instances having the same OI
    min_max = {v: [0, sys.maxsize] for v in oi.values()}
    for k, v in time.items():
        i = oi[k]
        min_max[i][0] = v if min_max[i][0] == 0 else min(v, min_max[i][0])
        min_max[i][1] = v if min_max[i][1] == sys.maxsize else max(v, min_max[i][1])

    with RooflinePlotter(figname=figname, plotdir=resultsdir,
                         max_bw=max_bw, flop_ceils=flop_ceils,
                         fancycolor=True, legend=legend) as plot:
        for k, v in gflopss.items():
            so = dict(k)['space_order']

            oi_value = oi[k]
            time_value = time[k]

            run = tuple(dict(k)[i] for i in modes)
            label = ("<%s>" % ','.join(run)) if run else None

            color = used_colors[run] if run in used_colors else avail_colors.pop(0)
            used_colors.setdefault(run, color)
            marker = used_markers[so] if so in used_markers else avail_markers.pop(0)
            used_markers.setdefault(so, marker)

            oi_loc = 0.076 if len(str(so)) == 1 else 0.09
            oi_annotate = {'s': 'SO=%s' % so, 'size': 6, 'xy': (oi_value, oi_loc)}
            if time_value in min_max[oi_value] and point_runtime:
                # Only annotate min and max runtimes on each OI line, to avoid
                # polluting the plot too much
                point_annotate = {'s': "%.0fs" % time_value, 'xytext': (0.0, 5.5),
                                  'size': 6, 'rotation': 0}
            else:
                point_annotate = None
            oi_line = time_value == min_max[oi_value][0]
            if oi_line:
                perf_annotate = {'size': 6, 'xytext': (-4, 5)}

            plot.add_point(gflops=v, oi=oi_value, marker=marker, color=color,
                           oi_line=oi_line, label=label, perf_annotate=perf_annotate,
                           oi_annotate=oi_annotate, point_annotate=point_annotate)


def get_ob_bench(problem, resultsdir, parameters):
    """Return a special ``opescibench.Benchmark`` to manage performance runs."""
    try:
        from opescibench import Benchmark
    except:
        raise ImportError('Could not import opescibench utility package.\n'
                          'Please install https://github.com/opesci/opescibench')

    class DevitoBenchmark(Benchmark):

        def param_string(self, params):
            devito_params, params = OrderedDict(), dict(params)
            devito_params['arch'] = params['arch']
            devito_params['shape'] = ",".join(str(i) for i in params['shape'])
            devito_params['nbpml'] = params['nbpml']
            devito_params['tn'] = params['tn']
            devito_params['so'] = params['space_order']
            devito_params['to'] = params['time_order']
            devito_params['dse'] = params['dse']
            devito_params['dle'] = params['dle']
            devito_params['at'] = params['autotune']

            if configuration['openmp']:
                default_nthreads = configuration['platform'].cores_physical
                devito_params['nt'] = os.environ.get('OMP_NUM_THREADS', default_nthreads)
            else:
                devito_params['nt'] = 1

            devito_params['np'] = MPI.COMM_WORLD.size
            devito_params['rank'] = MPI.COMM_WORLD.rank
            devito_params['mpi'] = configuration['mpi']

            return '_'.join(['%s[%s]' % (k, v) for k, v in devito_params.items()])

    return DevitoBenchmark(name=problem, resultsdir=resultsdir, parameters=parameters)


def get_ob_exec(func):
    """Return a special ``opescibench.Executor`` to execute performance runs."""
    try:
        from opescibench import Executor
    except:
        raise ImportError('Could not import opescibench utility package.\n'
                          'Please install https://github.com/opesci/opescibench')

    class DevitoExecutor(Executor):

        def __init__(self, func):
            super(DevitoExecutor, self).__init__(comm=MPI.COMM_WORLD)
            self.func = func

        def run(self, *args, **kwargs):
            clear_cache()

            gflopss, oi, timings, _ = self.func(*args, **kwargs)

            for key in timings.keys():
                self.register(gflopss[key], measure="gflopss", event=key.name)
                self.register(oi[key], measure="oi", event=key.name)
                self.register(timings[key], measure="timings", event=key.name)

    return DevitoExecutor(func)


def get_ob_plotter():
    try:
        from opescibench import RooflinePlotter
    except:
        raise ImportError('Could not import opescibench utility package.\n'
                          'Please install https://github.com/opesci/opescibench'
                          'To plot performance results, make sure to have the'
                          'Matplotlib package installed')
    return RooflinePlotter


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

    # Profiling at max level
    configuration['profiling'] = 'advanced'

    benchmark()

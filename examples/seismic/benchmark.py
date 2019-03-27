from collections import OrderedDict
import sys

import numpy as np
import click

from devito import clear_cache, configuration, mode_develop, mode_benchmark, warning
from devito.tools import as_tuple, sweep
from examples.seismic.acoustic.acoustic_example import run as acoustic_run, acoustic_setup
from examples.seismic.tti.tti_example import run as tti_run, tti_setup


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
        click.option('-P', '--problem', type=click.Choice(['acoustic', 'tti']),
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
                     help='End time of the simulation in ms'),
    ]
    for option in reversed(options):
        f = option(f)
    return f


def option_performance(f):
    """Defines options for all aspects of performance tuning"""

    _preset = {
        # Fixed
        'O1': {'dse': 'basic', 'dle': 'basic'},
        'O2': {'dse': 'advanced', 'dle': 'advanced'},
        'O3': {'dse': 'aggressive', 'dle': 'advanced'},
        # Parametric
        'dse': {'dse': ['basic', 'advanced', 'aggressive'], 'dle': 'advanced'},
        'dle': {'dse': 'advanced', 'dle': ['basic', 'advanced']}
    }

    def from_preset(ctx, param, value):
        """Set all performance options according to bench-mode preset"""
        ctx.params.update(_preset[value])
        return value

    def from_value(ctx, param, value):
        """Prefer preset values and warn for competing values."""
        return ctx.params[param.name] or value

    options = [
        click.option('-bm', '--bench-mode', is_eager=True,
                     callback=from_preset, expose_value=False, default='O2',
                     type=click.Choice(['O1', 'O2', 'O3', 'dse', 'dle']),
                     help='Choose what to benchmark; ignored if execmode=run'),
        click.option('--arch', default='unknown',
                     help='Architecture on which the simulation is/was run'),
        click.option('--dse', callback=from_value,
                     type=click.Choice(['noop'] + configuration._accepted['dse']),
                     help='Devito symbolic engine (DSE) mode'),
        click.option('--dle', callback=from_value,
                     type=click.Choice(['noop'] + configuration._accepted['dle']),
                     help='Devito loop engine (DLE) mode'),
        click.option('-a', '--autotune', is_flag=True,
                     help='Switch auto tuning on/off')
    ]
    for option in reversed(options):
        f = option(f)
    return f


@benchmark.command(name='run')
@option_simulation
@option_performance
@click.option('-bs', '--block-shape', default=(0, 0, 0),
              help='Loop-blocking shape, bypass autotuning')
def cli_run(problem, **kwargs):
    """
    A single run with a specific set of performance parameters.
    """
    mode_benchmark()
    run(problem, **kwargs)


def run(problem, **kwargs):
    """
    A single run with a specific set of performance parameters.
    """
    setup = tti_setup if problem == 'tti' else acoustic_setup
    options = {}

    time_order = kwargs.pop('time_order')[0]
    space_order = kwargs.pop('space_order')[0]
    autotune = kwargs.pop('autotune')

    # Should a specific block-shape be used? Useful if one wants to skip
    # the autotuning pass as a good block-shape is already known
    block_shape = as_tuple(kwargs.pop('block_shape'))
    if all(block_shape):
        if autotune:
            warning("Skipping autotuning (using explicit block-shape `%s`)"
                    % str(block_shape))
            autotune = False
        # This is quite hacky, but it does the trick
        for d, bs in zip(['x', 'y', 'z'], block_shape):
            options['%s0_blk_size' % d] = bs

    solver = setup(space_order=space_order, time_order=time_order, **kwargs)
    solver.forward(autotune=autotune, **options)


@benchmark.command(name='test')
@option_simulation
@option_performance
def cli_test(problem, **kwargs):
    """
    Test numerical correctness with different parameters.
    """
    mode_develop()
    test(problem, **kwargs)


def test(problem, **kwargs):
    """
    Test numerical correctness with different parameters.
    """
    run = tti_run if problem == 'tti' else acoustic_run
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
@option_simulation
@option_performance
@click.option('-r', '--resultsdir', default='results',
              help='Directory containing results')
@click.option('-x', '--repeats', default=3,
              help='Number of test case repetitions')
def cli_bench(problem, **kwargs):
    """
    Complete benchmark with multiple simulation and performance parameters.
    """
    mode_benchmark()
    kwargs['autotune'] = configuration['autotuning'].level
    bench(problem, **kwargs)


def bench(problem, **kwargs):
    """
    Complete benchmark with multiple simulation and performance parameters.
    """
    run = tti_run if problem == 'tti' else acoustic_run
    resultsdir = kwargs.pop('resultsdir')
    repeats = kwargs.pop('repeats')

    bench = get_ob_bench(problem, resultsdir, kwargs)
    bench.execute(get_ob_exec(run), warmups=0, repeats=repeats)
    bench.save()

    # Final clean up, just in case the benchmarker is used from external Python modules
    clear_cache()


@benchmark.command(name='plot')
@option_simulation
@option_performance
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

    arch = kwargs['arch']
    space_order = "[%s]" % ",".join(str(i) for i in kwargs['space_order'])
    time_order = kwargs['time_order']
    shape = "[%s]" % ",".join(str(i) for i in kwargs['shape'])

    RooflinePlotter = get_ob_plotter()
    bench = get_ob_bench(problem, resultsdir, kwargs)

    bench.load()
    if not bench.loaded:
        warning("Could not load any results, nothing to plot. Exiting...")
        sys.exit(0)

    gflopss = bench.lookup(params=kwargs, measure="gflopss", event='main')
    oi = bench.lookup(params=kwargs, measure="oi", event='main')
    time = bench.lookup(params=kwargs, measure="timings", event='main')

    # What plot am I?
    modes = [i for i in ['dse', 'dle', 'autotune']
             if len(set(dict(j)[i] for j in gflopss)) > 1]

    # Filename
    figname = "%s_dim%s_so%s_to%s_arch[%s]_bkend[%s].pdf" % (
        problem, shape, space_order, time_order, arch, backend
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
    """Return a special :class:`opescibench.Benchmark` to manage performance runs."""
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
            return '_'.join(['%s[%s]' % (k, v) for k, v in devito_params.items()])

    return DevitoBenchmark(name=problem, resultsdir=resultsdir, parameters=parameters)


def get_ob_exec(func):
    """Return a special :class:`opescibench.Executor` to execute performance runs."""
    try:
        from opescibench import Executor
    except:
        raise ImportError('Could not import opescibench utility package.\n'
                          'Please install https://github.com/opesci/opescibench')

    class DevitoExecutor(Executor):

        def __init__(self, func):
            super(DevitoExecutor, self).__init__()
            self.func = func

        def run(self, *args, **kwargs):
            clear_cache()

            gflopss, oi, timings, _ = self.func(*args, **kwargs)

            for key in timings.keys():
                self.register(gflopss[key], measure="gflopss", event=key)
                self.register(oi[key], measure="oi", event=key)
                self.register(timings[key], measure="timings", event=key)

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
    benchmark()

import sys
import numpy as np
import click

from devito import clear_cache, configuration, sweep
from devito.logger import warning
from examples.seismic.acoustic.acoustic_example import run as acoustic_run
from examples.seismic.tti.tti_example import run as tti_run


@click.group()
def benchmark():
    """
    Benchmarking script for seismic forward operators.

    There are three main 'execution modes':
    run:   a single run with given DSE/DLE levels
    bench: complete benchmark with multiple DSE/DLE levels
    test:  tests numerical correctness with different parameters

    Further, this script can generate a roofline plot from a benchmark
    """

    # Make sure that with YASK we run in performance mode
    if configuration['backend'] == 'yask':
        configuration.yask['develop-mode'] = False


def option_simulation(f):
    def default_list(ctx, param, value):
        return list(value if len(value) > 0 else (2, ))

    options = [
        click.option('-P', '--problem', type=click.Choice(['acoustic', 'tti']),
                     help='Number of grid points along each axis'),
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

    _preset = {'maxperf': {
        'autotune': True,
        'dse': 'advanced',
        'dle': 'advanced'
    }, 'dse': {
        'autotune': True,
        'dse': ['basic', 'advanced', 'speculative', 'aggressive'],
        'dle': 'advanced',
    }, 'dle': {
        'autotune': True,
        'dse': 'advanced',
        'dle': ['basic', 'advanced'],
    }}

    def from_preset(ctx, param, value):
        """Set all performance options according to bench-mode preset"""
        ctx.params.update(_preset[value])
        return value

    def from_value(ctx, param, value):
        """Prefer preset values and warn for competing values."""
        return ctx.params[param.name] or value

    options = [
        click.option('-bm', '--bench-mode', is_eager=True,
                     callback=from_preset, expose_value=False,
                     type=click.Choice(['maxperf', 'dse', 'dle']), default='maxperf',
                     help='Choose what to benchmark; ignored if execmode=run'),
        click.option('--dse', callback=from_value,
                     type=click.Choice(['noop'] + configuration._accepted['dse']),
                     help='Devito symbolic engine (DSE) mode'),
        click.option('--dle', callback=from_value,
                     type=click.Choice(['noop'] + configuration._accepted['dle']),
                     help='Devito loop engine (DLE) mode'),
        click.option('-a', '--autotune', is_flag=True, callback=from_value,
                     help='Switch auto tuning on/off'),
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
    run(problem, **kwargs)


def run(problem, **kwargs):
    """
    A single run with a specific set of performance parameters.
    """
    run = tti_run if problem == 'tti' else acoustic_run
    time_order = kwargs.pop('time_order')[0]
    space_order = kwargs.pop('space_order')[0]
    run(space_order=space_order, time_order=time_order, **kwargs)


@benchmark.command(name='test')
@option_simulation
@option_performance
def cli_test(problem, **kwargs):
    """
    Test numerical correctness with different parameters.
    """
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
    bench(problem, **kwargs)


def bench(problem, **kwargs):
    """
    Complete benchmark with multiple simulation and performance parameters.
    """
    try:
        from opescibench import Benchmark, Executor
    except:
        raise ImportError('Could not import opescibench utility package.\n'
                          'Please install https://github.com/opesci/opescibench')

    run = tti_run if problem == 'tti' else acoustic_run
    resultsdir = kwargs.pop('resultsdir')
    repeats = kwargs.pop('repeats')

    class BenchExecutor(Executor):
        """Executor class that defines how to run the benchmark"""

        def run(self, *args, **kwargs):
            gflopss, oi, timings, _ = run(*args, **kwargs)

            for key in timings.keys():
                self.register(gflopss[key], measure="gflopss", event=key)
                self.register(oi[key], measure="oi", event=key)
                self.register(timings[key], measure="timings", event=key)

            clear_cache()

    bench = Benchmark(name=problem, resultsdir=resultsdir, parameters=kwargs)
    bench.execute(BenchExecutor(), warmups=0, repeats=repeats)
    bench.save()


@benchmark.command(name='plot')
@option_simulation
@option_performance
@click.option('-r', '--resultsdir', default='results',
              help='Directory containing results')
@click.option('-p', '--plotdir', default='plots',
              help='Directory containing plots')
@click.option('--arch', default='unknown',
              help='Architecture on which the simulation is/was run')
@click.option('--max_bw', type=float,
              help='Max GB/s of the DRAM')
@click.option('--max_flops', type=float,
              help='Max GFLOPS/s of the CPU')
@click.option('--point_runtime', is_flag=True,
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
    try:
        from opescibench import Benchmark, RooflinePlotter
    except:
        raise ImportError("Could not import opescibench utility package.\n"
                          "Please install https://github.com/opesci/opescibench "
                          "and Matplotlib to plot performance results")
    resultsdir = kwargs.pop('resultsdir')
    plotdir = kwargs.pop('plotdir')
    arch = kwargs.pop('arch')
    max_bw = kwargs.pop('max_bw')
    max_flops = kwargs.pop('max_flops')
    point_runtime = kwargs.pop('point_runtime')

    bench = Benchmark(name=problem, resultsdir=resultsdir, parameters=kwargs)
    bench.load()
    if not bench.loaded:
        warning("Could not load any results, nothing to plot. Exiting...")
        sys.exit(0)

    gflopss = bench.lookup(params=kwargs, measure="gflopss", event='main')
    oi = bench.lookup(params=kwargs, measure="oi", event='main')
    time = bench.lookup(params=kwargs, measure="timings", event='main')

    name = "%s_dim%s_so%s_to%s_arch[%s].pdf" % (problem,
                                                kwargs['shape'],
                                                kwargs['space_order'],
                                                kwargs['time_order'],
                                                arch)
    name = name.replace(' ', '')
    problem_styles = {'acoustic': 'Acoustic', 'tti': 'TTI'}
    title = "%s [grid=%s, TO=%s, duration=%sms], varying <DSE,DLE> on %s" %\
        (problem_styles[problem], list(kwargs['shape']), kwargs['time_order'],
         kwargs['tn'], arch)

    styles = {  # (marker, color)
        # DLE basic
        ('basic', 'basic'): ('D', 'r'),
        ('advanced', 'basic'): ('D', 'g'),
        ('speculative', 'basic'): ('D', 'y'),
        ('aggressive', 'basic'): ('D', 'b'),
        # DLE advanced
        ('basic', 'advanced'): ('o', 'r'),
        ('advanced', 'advanced'): ('o', 'g'),
        ('speculative', 'advanced'): ('o', 'y'),
        ('aggressive', 'advanced'): ('o', 'b'),
        # DLE speculative
        ('basic', 'speculative'): ('s', 'r'),
        ('advanced', 'speculative'): ('s', 'g'),
        ('speculative', 'speculative'): ('s', 'y'),
        ('aggressive', 'speculative'): ('s', 'b')
    }

    # Find min and max runtimes for instances having the same OI
    min_max = {v: [0, sys.maxsize] for v in oi.values()}
    for k, v in time.items():
        i = oi[k]
        min_max[i][0] = v if min_max[i][0] == 0 else min(v, min_max[i][0])
        min_max[i][1] = v if min_max[i][1] == sys.maxsize else max(v, min_max[i][1])

    with RooflinePlotter(title=title, figname=name, plotdir=plotdir,
                         max_bw=max_bw, max_flops=max_flops,
                         fancycolor=True, legend={'fontsize': 5, 'ncol': 4}) as plot:
        for key, gflopss in gflopss.items():
            oi_value = oi[key]
            time_value = time[key]
            key = dict(key)
            run = (key["dse"], key["dle"])
            label = "<%s,%s>" % run
            oi_loc = 0.05 if len(str(key["space_order"])) == 1 else 0.06
            oi_annotate = {'s': 'SO=%s' % key["space_order"],
                           'size': 4, 'xy': (oi_value, oi_loc)} if run[0] else None
            if time_value in min_max[oi_value] and point_runtime:
                # Only annotate min and max runtimes on each OI line, to avoid
                # polluting the plot too much
                point_annotate = {'s': "%.1f s" % time_value, 'xytext': (0, 5.2),
                                  'size': 3.5, 'weight': 'bold', 'rotation': 0}
            else:
                point_annotate = None
            oi_line = time_value == min_max[oi_value][0]
            if oi_line:
                perf_annotate = {'size': 4, 'xytext': (-4, 4)}
            plot.add_point(gflops=gflopss, oi=oi_value, marker=styles[run][0],
                           color=styles[run][1], oi_line=oi_line, label=label,
                           perf_annotate=perf_annotate, oi_annotate=oi_annotate,
                           point_annotate=point_annotate)


if __name__ == "__main__":
    benchmark()

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from itertools import product
from os import environ

import numpy as np

from devito import clear_cache
from devito.compiler import compiler_registry
from devito.logger import warning
from seismic.acoustic.acoustic_example import run as acoustic_run
from seismic.tti.tti_example3D import run as tti_run

if __name__ == "__main__":
    description = ("Benchmarking script for TTI example.\n\n" +
                   "Exec modes:\n" +
                   "\trun:   executes tti_example3D.py once " +
                   "with the provided parameters\n" +
                   "\ttest:  tests numerical correctness with different parameters\n"
                   "\tbench: runs a benchmark of tti_example3D.py\n" +
                   "\tplot:  plots a roofline plot using the results from the benchmark\n"
                   )
    parser = ArgumentParser(description=description,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(dest="execmode", nargs="?", default="run",
                        choices=["run", "test", "bench", "plot"],
                        help="Exec modes")
    parser.add_argument("--bench-mode", "-bm", dest="benchmode", default="maxperf",
                        choices=["maxperf", "dse", "dle"],
                        help="Choose what to benchmark (maxperf, dse, dle).")
    parser.add_argument(dest="compiler", nargs="?",
                        default=environ.get("DEVITO_ARCH", "gnu"),
                        choices=compiler_registry.keys(),
                        help="Compiler/architecture to use. Defaults to DEVITO_ARCH")
    parser.add_argument("--arch", default="unknown",
                        help="Architecture on which the simulation is/was run.")
    parser.add_argument("-P", "--problem", nargs="?", default="tti",
                        choices=["acoustic", "tti"], help="Problem")
    simulation = parser.add_argument_group("Simulation")
    simulation.add_argument("-o", "--omp", action="store_true",
                            help="Enable OpenMP")
    simulation.add_argument("-d", "--dimensions", nargs=3, default=[50, 50, 50],
                            type=int, help="Dimensions of the grid",
                            metavar=("dim1", "dim2", "dim3"))
    simulation.add_argument("-s", "--spacing", nargs=3, default=[20.0, 20.0, 20.0],
                            type=float,
                            help="Spacing between grid sizes in meters",
                            metavar=("spc1", "spc2", "spc3"))
    simulation.add_argument("-n", "--nbpml", default=10, type=int,
                            help="Number of PML points")
    simulation.add_argument("-so", "--space_order", nargs="*", default=[2],
                            type=int, help="Space order of the simulation")
    simulation.add_argument("-to", "--time_order", nargs="*", default=[2],
                            type=int, help="Time order of the simulation")
    simulation.add_argument("-t", "--tn", default=250,
                            type=int, help="End time of the simulation in ms")

    devito = parser.add_argument_group("Devito")
    devito.add_argument("-dse", default="advanced", nargs="*",
                        choices=["noop", "basic", "factorize", "approx-trigonometry",
                                 "glicm", "advanced"],
                        help="Devito symbolic engine (DSE) mode")
    devito.add_argument("-dle", default="advanced", nargs="*",
                        choices=["noop", "advanced", "3D-advanced", "speculative"],
                        help="Devito loop engine (DSE) mode")
    devito.add_argument("-a", "--autotune", action="store_true",
                        help=("Benchmark with auto tuning on and off. " +
                              "Enables auto tuning when execmode is run"))
    # devito.add_argument("-cb", "--cache_blocking", nargs=2, type=int,
    #                    default=None, metavar=("blockDim1", "blockDim2"),
    #                    help="User provided block sizes when auto-tuning is off")

    benchmarking = parser.add_argument_group("Benchmarking")
    benchmarking.add_argument("-r", "--resultsdir", default="results",
                              help="Directory containing results")

    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument("-p", "--plotdir", default="plots",
                          help="Directory containing plots")
    plotting.add_argument("--max_bw", type=float, help="Maximum bandwith of the system")
    plotting.add_argument("--max_flops", type=float, help="Maximum FLOPS of the system")
    plotting.add_argument("--point_runtime", action="store_true",
                          help="Annotate points with runtime values")

    args = parser.parse_args()

    if args.problem == "tti":
        run = tti_run
    else:
        run = acoustic_run

    parameters = vars(args).copy()
    del parameters["execmode"]
    del parameters["benchmode"]
    del parameters["problem"]
    del parameters["resultsdir"]
    del parameters["plotdir"]
    del parameters["max_bw"]
    del parameters["max_flops"]
    del parameters["omp"]
    del parameters["point_runtime"]
    del parameters["arch"]

    parameters["dimensions"] = tuple(parameters["dimensions"])
    parameters["spacing"] = tuple(parameters["spacing"])

    parameters["compiler"] = compiler_registry[args.compiler](openmp=args.omp)

    if args.execmode == "run":
        parameters["space_order"] = parameters["space_order"][0]
        parameters["time_order"] = parameters["time_order"][0]
        run(**parameters)
    else:
        if args.benchmode == 'maxperf':
            parameters["autotune"] = [True]
            parameters["dse"] = ["advanced"]
            parameters["dle"] = ["advanced"]
        elif args.benchmode == 'dse':
            parameters["autotune"] = [False]
            parameters["dse"] = ["basic",
                                 ('basic', 'glicm'),
                                 "advanced"]
            parameters["dle"] = ["basic"]
        else:
            # must be == 'dle'
            parameters["autotune"] = [True]
            parameters["dse"] = ["advanced"]
            parameters["dle"] = ["basic",
                                 "advanced"]

    if args.execmode == "test":
        values_sweep = [v if isinstance(v, list) else [v] for v in parameters.values()]
        params_sweep = [dict(zip(parameters.keys(), values))
                        for values in product(*values_sweep)]

        last_res = None

        for params in params_sweep:
            _, _, _, res = run(**params)

            if last_res is None:
                last_res = res
            else:
                for i in range(len(res)):
                    np.isclose(res[i], last_res[i])

    elif args.execmode == "bench":
        try:
            from opescibench import Benchmark, Executor
        except:
            raise ImportError("Could not import opescibench utility package.\n"
                              "Please install https://github.com/opesci/opescibench")

        class BenchExecutor(Executor):
            """Executor class that defines how to run the benchmark"""

            def run(self, *args, **kwargs):
                gflopss, oi, timings, _ = run(*args, **kwargs)

                for key in timings.keys():
                    self.register(gflopss[key], measure="gflopss", event=key)
                    self.register(oi[key], measure="oi", event=key)
                    self.register(timings[key], measure="timings", event=key)

                clear_cache()

        bench = Benchmark(
            name=args.problem, resultsdir=args.resultsdir, parameters=parameters
        )
        bench.execute(BenchExecutor(), warmups=0, repeats=3)
        bench.save()

    elif args.execmode == "plot":
        try:
            from opescibench import Benchmark, RooflinePlotter
        except:
            raise ImportError("Could not import opescibench utility package.\n"
                              "Please install https://github.com/opesci/opescibench "
                              "and Matplotlib to plot performance results")

        bench = Benchmark(
            name=args.problem, resultsdir=args.resultsdir, parameters=parameters
        )
        bench.load()
        if not bench.loaded:
            warning("Could not load any results, nothing to plot. Exiting...")
            sys.exit(0)

        gflopss = bench.lookup(params=parameters, measure="gflopss", event="main")
        oi = bench.lookup(params=parameters, measure="oi", event="main")
        time = bench.lookup(params=parameters, measure="timings", event="main")

        name = "%s_%s_dim%s_so%s_to%s_arch[%s].pdf" % (args.problem,
                                                       args.benchmode,
                                                       parameters["dimensions"],
                                                       parameters["space_order"],
                                                       parameters["time_order"],
                                                       args.arch)
        name = name.replace(' ', '')
        title = "%s[%s,TO=%s], with varying <DSE,DLE>, on %s" %\
            (args.problem.capitalize(),
             parameters["dimensions"],
             parameters["time_order"],
             args.arch)

        dse_runs = ["basic", ("basic", "factorize"), ("basic", "glicm"), "advanced"]
        dle_runs = ["basic", "advanced", "speculative"]
        runs = list(product(dse_runs, dle_runs))
        styles = {
            # DLE basic
            ('basic', 'basic'): 'or',
            (('basic', 'factorize'), 'basic'): 'og',
            (('basic', 'glicm'), 'basic'): 'oy',
            ('advanced', 'basic'): 'ob',
            # DLE advanced
            ('basic', 'advanced'): 'Dr',
            (('basic', 'factorize'), 'advanced'): 'Dg',
            (('basic', 'glicm'), 'advanced'): 'Dy',
            ('advanced', 'advanced'): 'Db',
            # DLE speculative
            ('basic', 'speculative'): 'sr',
            (('basic', 'factorize'), 'speculative'): 'sg',
            (('basic', 'glicm'), 'speculative'): 'sy',
            ('advanced', 'speculative'): 'sb',
        }

        # Find min and max runtimes for instances having the same OI
        min_max = {v: [0, sys.maxint] for v in oi.values()}
        for k, v in time.items():
            i = oi[k]
            min_max[i][0] = v if min_max[i][0] == 0 else min(v, min_max[i][0])
            min_max[i][1] = v if min_max[i][1] == sys.maxint else max(v, min_max[i][1])

        with RooflinePlotter(title=title, figname=name, plotdir=args.plotdir,
                             max_bw=args.max_bw, max_flops=args.max_flops,
                             legend={'fontsize': 7}) as plot:
            for key, gflopss in gflopss.items():
                oi_value = oi[key]
                time_value = time[key]
                key = dict(key)
                run = (key["dse"], key["dle"])
                label = "<%s,%s>" % run
                oi_loc = 0.06 if len(str(key["space_order"])) == 1 else 0.07
                oi_annotate = {'s': 'SO=%s' % key["space_order"],
                               'size': 5, 'xy': (oi_value, oi_loc)} if run[0] else None
                if time_value in min_max[oi_value]:
                    # Only annotate min and max runtimes on each OI line, to avoid
                    # polluting the plot too much
                    point_annotate = {'s': "%.1f s" % time_value,
                                      'xytext': (-16, 13), 'size': 4,
                                      'weight': 'bold'} if args.point_runtime else None
                else:
                    point_annotate = None
                oi_line = time_value == min_max[oi_value][0]
                perf_annotate = time_value == min_max[oi_value][0]
                plot.add_point(gflops=gflopss, oi=oi_value, style=styles[run],
                               oi_line=oi_line, label=label, perf_annotate=perf_annotate,
                               oi_annotate=oi_annotate, annotate=point_annotate)

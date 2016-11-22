import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from itertools import product
from os import environ

import numpy as np

from devito import clear_cache
from devito.compiler import compiler_registry
from devito.logger import warning
from examples.acoustic.acoustic_example import run as acoustic_run
from examples.tti.tti_example import run as tti_run

try:
    from opescibench import Benchmark, Executor, RooflinePlotter
except:
    Benchmark = None
    Executor = None
    RooflinePlotter = None


if __name__ == "__main__":
    description = ("Benchmarking script for TTI example.\n\n" +
                   "Exec modes:\n" +
                   "\trun:   executes tti_example.py once " +
                   "with the provided parameters\n" +
                   "\ttest:  tests numerical correctness with different parameters\n"
                   "\tbench: runs a benchmark of tti_example.py\n" +
                   "\tplot:  plots a roofline plot using the results from the benchmark\n"
                   )
    parser = ArgumentParser(description=description,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(dest="execmode", nargs="?", default="run",
                        choices=["run", "test", "bench", "plot"],
                        help="Exec modes")
    parser.add_argument(dest="compiler", nargs="?",
                        default=environ.get("DEVITO_ARCH", "gnu"),
                        choices=compiler_registry.keys(),
                        help="Compiler/architecture to use. Defaults to DEVITO_ARCH")
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
    devito.add_argument("-dse", default="advanced", choices=["basic", "advanced"],
                        help="Devito symbolic engine (DSE) mode")
    devito.add_argument("-a", "--auto_tuning", action="store_true",
                        help=("Benchmark with auto tuning on and off. " +
                              "Enables auto tuning when execmode is run"))
    devito.add_argument("-cb", "--cache_blocking", nargs=2, type=int,
                        default=None, metavar=("blockDim1", "blockDim2"),
                        help="Uses provided block sizes when AT is off")

    benchmarking = parser.add_argument_group("Benchmarking")
    benchmarking.add_argument("-r", "--resultsdir", default="results",
                              help="Directory containing results")

    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument("-p", "--plotdir", default="plots",
                          help="Directory containing plots")
    plotting.add_argument("--max_bw", type=float, help="Maximum bandwith of the system")
    plotting.add_argument("--max_flops", type=float, help="Maximum FLOPS of the system")

    args = parser.parse_args()

    if args.problem == "tti":
        run = tti_run
    else:
        run = acoustic_run

    parameters = vars(args).copy()
    del parameters["execmode"]
    del parameters["problem"]
    del parameters["resultsdir"]
    del parameters["plotdir"]
    del parameters["max_bw"]
    del parameters["max_flops"]
    del parameters["omp"]

    parameters["dimensions"] = tuple(parameters["dimensions"])
    parameters["spacing"] = tuple(parameters["spacing"])

    if parameters["cache_blocking"]:
        parameters["cache_blocking"] = parameters["cache_blocking"] + [None]

    parameters["compiler"] = compiler_registry[args.compiler](openmp=args.omp)

    if args.execmode == "run":
        parameters["space_order"] = parameters["space_order"][0]
        parameters["time_order"] = parameters["time_order"][0]
        run(**parameters)
    else:
        if Benchmark is None and args.execmode != "test":
            raise ImportError("Could not find opescibench utility package.\n"
                              "Please install from https://github.com/opesci/opescibench")

        if parameters["auto_tuning"]:
            parameters["auto_tuning"] = [True, False]

        if parameters["dse"]:
            parameters["dse"] = ["basic", "advanced"]

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
        bench.execute(BenchExecutor(), warmups=0)
        bench.save()

    elif args.execmode == "plot":
        bench = Benchmark(
            name=args.problem, resultsdir=args.resultsdir, parameters=parameters
        )
        bench.load()
        if not bench.loaded:
            warning("Could not load any results, nothing to plot. Exiting...")
            sys.exit(0)

        gflopss = bench.lookup(params=parameters, measure="gflopss", event="loop_body")
        oi = bench.lookup(params=parameters, measure="oi", event="loop_body")

        name = "%s_dim%s_so%s_to%s.pdf" % (args.problem, parameters["dimensions"],
                                           parameters["space_order"],
                                           parameters["time_order"])
        title = "%s - grid: %s, time order: %s" % (args.problem.capitalize(),
                                                   parameters["dimensions"],
                                                   parameters["time_order"])

        at_runs = [True, False]
        dse_runs = ["basic", "advanced"]
        runs = list(product(at_runs, dse_runs))

        with RooflinePlotter(figname=name, plotdir=args.plotdir,
                             max_bw=args.max_bw, max_flops=args.max_flops,
                             legend={'fontsize': 8} ) as plot:
            for key, gflopss in gflopss.items():
                oi_value = oi[key]
                key = dict(key)
                run = (key["auto_tuning"], key["dse"])
                index = runs.index(run)
                style = '%s%s' % (plot.color[index], plot.marker[index])
                label = "[AT=%r,DSE=%s]" % run
                annotation = {'s': 'SO=%s' % key["space_order"],
                              'size': 6} if run[0] else None
                plot.add_point(gflops=gflopss, oi=oi_value, style=style, oi_line=run[0],
                               label=label, oi_annotate=annotation)

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from itertools import product

import numpy as np

from devito import clear_cache
from devito.logger import warning
from examples.seismic.acoustic.acoustic_example import run as acoustic_run
from examples.seismic.tti.tti_example import run as tti_run

if __name__ == "__main__":
    description = ("Benchmarking script for seismic forward operators.\n\n" +
                   "There are three main 'execution modes':\n" +
                   "\trun:   a single run with given DSE/DLE levels\n" +
                   "\tbench: complete benchmark with multiple DSE/DLE levels\n" +
                   "\ttest:  tests numerical correctness with different parameters\n" +
                   "Further, this script can generate a roofline plot from a benchmark\n"
                   )
    parser = ArgumentParser(description=description,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(dest="execmode", default="run",
                        choices=["run", "bench", 'test', "plot"], help="Execution modes")
    parser.add_argument("--bench-mode", "-bm", dest="benchmode", default="maxperf",
                        choices=["maxperf", "dse", "dle"],
                        help="Choose what to benchmark; ignored if execmode=run")
    parser.add_argument("--arch", default="unknown",
                        help="Architecture on which the simulation is/was run")
    parser.add_argument("-P", "--problem", default="acoustic",
                        choices=["acoustic", "tti"], help="Problem")

    simulation = parser.add_argument_group("Simulation")
    simulation.add_argument("-d", "--shape", nargs=3, default=[50, 50, 50],
                            type=int, help="Number of grid points along each axis",
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
    devito.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced", "speculative",
                                 "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    devito.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DSE) mode")
    devito.add_argument("-a", "--autotune", action="store_true",
                        help=("Switch auto tuning on/off; ignored if execmode=bench"))

    benchmarking = parser.add_argument_group("Benchmarking")
    benchmarking.add_argument("-r", "--resultsdir", default="results",
                              help="Directory containing results")
    benchmarking.add_argument("-x", "--repeats", default="3", type=int,
                              help="Test case repetitions; ignored if execmode=run")

    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument("-p", "--plotdir", default="plots",
                          help="Directory containing plots")
    plotting.add_argument("--max_bw", type=float, help="Max GB/s of the DRAM")
    plotting.add_argument("--max_flops", type=float, help="Max GFLOPS/s of the CPU")
    plotting.add_argument("--point_runtime", action="store_true",
                          help="Annotate points with runtime values")

    args = parser.parse_args()

    if args.problem == "tti":
        run = tti_run
    else:
        run = acoustic_run

    parameters = vars(args).copy()
    # Drop what's unnecessary to run an operator
    del parameters["execmode"]
    del parameters["benchmode"]
    del parameters["problem"]
    del parameters["arch"]
    del parameters["resultsdir"]
    del parameters["repeats"]
    del parameters["plotdir"]
    del parameters["max_bw"]
    del parameters["max_flops"]
    del parameters["point_runtime"]

    parameters["shape"] = tuple(parameters["shape"])
    parameters["spacing"] = tuple(parameters["spacing"])

    if args.execmode == "run":
        parameters["space_order"] = parameters["space_order"][0]
        parameters["time_order"] = parameters["time_order"][0]
        run(**parameters)
    else:
        if args.benchmode == 'maxperf':
            parameters["autotune"] = [True]
            parameters["dse"] = ["aggressive"]
            parameters["dle"] = ["advanced"]
        elif args.benchmode == 'dse':
            parameters["autotune"] = [True]
            parameters["dse"] = ["basic", "advanced", "speculative", "aggressive"]
            parameters["dle"] = ["advanced"]
        else:
            # must be == 'dle'
            parameters["autotune"] = [True]
            parameters["dse"] = ["advanced"]
            parameters["dle"] = ["basic", "advanced"]

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
        bench.execute(BenchExecutor(), warmups=0, repeats=args.repeats)
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
        problem_styles = {'acoustic': 'Acoustic', 'tti': 'TTI'}
        title = "%s [grid=%s, TO=%s, duration=%sms], varying <DSE,DLE> on %s" %\
            (problem_styles[args.problem],
             list(parameters["dimensions"]),
             parameters["time_order"],
             args.tn,
             args.arch)

        dse_runs = ["basic", "advanced", "speculative", "aggressive"]
        dle_runs = ["basic", "advanced", "speculative"]
        runs = list(product(dse_runs, dle_runs))
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
        min_max = {v: [0, sys.maxint] for v in oi.values()}
        for k, v in time.items():
            i = oi[k]
            min_max[i][0] = v if min_max[i][0] == 0 else min(v, min_max[i][0])
            min_max[i][1] = v if min_max[i][1] == sys.maxint else max(v, min_max[i][1])

        with RooflinePlotter(title=title, figname=name, plotdir=args.plotdir,
                             max_bw=args.max_bw, max_flops=args.max_flops,
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
                if time_value in min_max[oi_value] and args.point_runtime:
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

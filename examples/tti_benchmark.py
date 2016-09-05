from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os import environ

import numpy as np

from devito.compiler import compiler_registry
from tti_example import run
try:
    from opescibench import Benchmark, Executor, Plotter
except:
    Benchmark = None
    Executor = None
    Plotter = None


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
    parser.add_argument("-o", "--omp", action="store_true",
                        help="Enable OpenMP")
    parser.add_argument("-d", "--dimensions", nargs=3, default=[50, 50, 50], type=int,
                        help="Dimension of the grid", metavar=("dim1", "dim2", "dim3"))
    parser.add_argument("-s", "--spacing", nargs=2, default=[20, 20], type=int,
                        help="Spacing on the grid", metavar=("spc1", "spc2"))
    parser.add_argument("-so", "--space_order", nargs="*", default=[2],
                        type=int, help="Space order of the simulation")
    parser.add_argument("-to", "--time_order", nargs="*", default=[2],
                        type=int, help="Time order of the simulation")
    parser.add_argument("-t", "--tn", default=250,
                        type=int, help="Number of timesteps")
    parser.add_argument("-c", "--cse", action="store_true",
                        help=("Benchmark with CSE on and off. " +
                              "Enables CSE when execmode is run"))
    parser.add_argument("-a", "--auto_tuning", action="store_true",
                        help=("Benchmark with auto tuning on and off. " +
                              "Enables auto tuning when execmode is run"))
    parser.add_argument("-cb", "--cache_blocking", nargs=2, type=int,
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

    parameters = vars(args).copy()
    del parameters["execmode"]
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
        if Benchmark is None:
            raise ImportError("Could not find opescibench utility package.\n"
                              "Please install from https://github.com/opesci/opescibench")

        if parameters["auto_tuning"]:
            parameters["auto_tuning"] = [True, False]
        if parameters["cse"]:
            parameters["cse"] = [True, False]

    if args.execmode == "test":
        class TTITester(Executor):
            """Executor class to test numerical correctness"""
            def __init__(self, *args, **kwargs):
                super(Executor, self).__init__(*args, **kwargs)
                self.last_rec = None
                self.last_u = None
                self.last_v = None

            def run(self, *args, **kwargs):
                _, _, rec, u, v = run(*args, **kwargs)

                if self.last_rec is not None:
                    np.isclose(rec, self.last_rec)
                else:
                    self.last_rec = rec

                if self.last_u is not None:
                    np.isclose(u, self.last_u)
                else:
                    self.last_u = u

                if self.last_v is not None:
                    np.isclose(v, self.last_v)
                else:
                    self.last_v = v

        test = Benchmark(name="TTItest", parameters=parameters)
        test.execute(TTITester(), warmups=0, repeats=1)

    elif args.execmode == "bench":
        class TTIExecutor(Executor):
            """Executor class that defines how to run TTI benchmark"""

            def run(self, *args, **kwargs):
                gflops, oi, _, _, _ = run(*args, **kwargs)

                self.register(gflops["kernel"], measure="gflops")
                self.register(oi["kernel"], measure="oi")

        bench = Benchmark(name="TTI", resultsdir=args.resultsdir, parameters=parameters)
        bench.execute(TTIExecutor(), warmups=0)
        bench.save()

    elif args.execmode == "plot":
        bench = Benchmark(name="TTI", resultsdir=args.resultsdir, parameters=parameters)
        bench.load()

        oi_dict = {}
        mflops_dict = {}

        gflops = bench.lookup(params=parameters, measure="gflops")
        oi = bench.lookup(params=parameters, measure="oi")

        for key, gflops in gflops.items():
            oi_value = oi[key]
            key = dict(key)
            label = "TTI, CSE: %s, AT: %s" % (key["cse"], key["auto_tuning"])
            mflops_dict[label] = gflops * 1000
            oi_dict[label] = oi_value

        name = ("TTI %s dimensions: %s - spacing: %s -"
                " space order: %s - time order: %s.pdf") % \
            (args.compiler, parameters["dimensions"], parameters["spacing"],
             parameters["space_order"], parameters["time_order"])
        name = name.replace(" ", "_")

        plotter = Plotter()
        plotter.plot_roofline(
            name, mflops_dict, oi_dict, args.max_bw, args.max_flops)

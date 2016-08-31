from argparse import ArgumentParser

import numpy as np

from containers import IGrid, IShot
from devito.compiler import compiler_registry
from TTI_codegen import TTI_cg

try:
    from opescibench import Benchmark, Executor, Plotter
except:
    Benchmark = None
    Executor = None
    Plotter = None


def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0), tn=250.0,
        cse=True, auto_tuning=False, compiler=None, cache_blocking=None):
    if auto_tuning:
        cache_blocking = None

    model = IGrid()
    model.shape = dimensions
    origin = (0., 0.)
    t_order = 2
    spc_order = 2

    # True velocity
    true_vp = np.ones(dimensions) + 1.0
    true_vp[:, :, int(dimensions[0] / 3):int(2*dimensions[0]/3)] = 3.0
    true_vp[:, :, int(2*dimensions[0] / 3):int(dimensions[0])] = 4.0

    model.create_model(
        origin, spacing, true_vp, .3*np.ones(dimensions), .2*np.ones(dimensions),
        np.pi/5*np.ones(dimensions), np.pi/5*np.ones(dimensions))

    # Define seismic data.
    data = IShot()

    f0 = .010
    dt = model.get_critical_dt()
    t0 = 0.0
    nt = int(1+(tn-t0)/dt)
    data.reinterpolate(dt)
    # Set up the source as Ricker wavelet for f0

    time_series = source(np.linspace(t0, tn, nt), f0)
    location = (origin[0] + dimensions[0] * spacing[0] * 0.5,
                origin[1] + dimensions[1] * spacing[1] * 0.5,
                origin[1] + 2 * spacing[1])
    data.set_source(time_series, dt, location)
    receiver_coords = np.zeros((101, 3))
    receiver_coords[:, 0] = np.linspace(50, 950, num=101)
    receiver_coords[:, 1] = 500
    receiver_coords[:, 2] = location[2]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)

    TTI = TTI_cg(model, data, None, t_order=t_order, s_order=spc_order, nbpml=10)
    rec, u, v, gflops, oi = TTI.Forward(
        cse=cse, auto_tuning=auto_tuning, cache_blocking=cache_blocking, compiler=compiler
    )
    return gflops, oi

if __name__ == "__main__":
    description = "Example script for TTI."
    parser = ArgumentParser(description=description)

    parser.add_argument(dest="execmode", nargs="?", default="run",
                        choices=["run", "bench", "plot"],
                        help="Script mode. Either 'run', 'bench' or 'plot'")
    parser.add_argument(dest="compiler", nargs="?", default="gnu",
                        choices=["gnu", "intel", "clang", "mic", "knl"])
    parser.add_argument("-o", "--omp", action="store_true",
                        help="Enable OpenMP")
    parser.add_argument("-d", "--dimensions", nargs=3, default=[50, 50, 50],
                        help="Dimension of the grid")
    parser.add_argument("-s", "--spacing", nargs=2, default=[20.0, 20.0],
                        help="Spacing on the grid")
    parser.add_argument("-t", "--tn", default=250,
                        type=int, help="Number of timesteps")
    parser.add_argument("-c", "--cse", action="store_true",
                        help=("Benchmark with CSE on and off. " +
                              "Enables CSE when execmode is run"))
    parser.add_argument("-a", "--auto_tuning", action="store_true",
                        help=("Benchmark with auto tuning on and off. " +
                              "Enables auto tuning when execmode is run"))
    parser.add_argument("-cb", "--cache_blocking", nargs=2, type=int,
                        default=None, help="Uses provided block sizes when AT is off")
    parser.add_argument("-r", "--resultsdir", default="results",
                        help="Directory containing results")
    parser.add_argument("-p", "--plotdir", default="plots",
                        help="Directory containing plots")
    parser.add_argument("--max_bw", type=float, help="Maximum bandwith of the system")
    parser.add_argument("--max_flops", type=float, help="Maximum FLOPS of the system")

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
        run(**parameters)

    if args.execmode == "bench":
        if Benchmark is None:
            raise ImportError("Could not find opescibench utility package.\n"
                              "Please install from https://github.com/opesci/opescibench")

        if parameters["auto_tuning"]:
            parameters["auto_tuning"] = [True, False]

        if parameters["cse"]:
            parameters["cse"] = [True, False]

        class TTIExecutor(Executor):
            """Executor class that defines how to run TTI benchmark"""

            def run(self, *args, **kwargs):
                gflops, oi = run(*args, **kwargs)

                self.register(gflops["kernel"], measure="gflops")
                self.register(oi["kernel"], measure="oi")

        bench = Benchmark(name="TTI", resultsdir=args.resultsdir, parameters=parameters)
        bench.execute(TTIExecutor(), warmups=0)
        bench.save()

    if args.execmode == "plot":
        if parameters["auto_tuning"]:
            parameters["auto_tuning"] = [True, False]

        if parameters["cse"]:
            parameters["cse"] = [True, False]

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

        name = "TTI - Dimensions: %s, Spacing: %s.pdf" % \
            (parameters["dimensions"], parameters["spacing"])

        plotter = Plotter()
        plotter.plot_roofline(
            name, mflops_dict, oi_dict, args.max_bw, args.max_flops)

import argparse

import numpy as np

from devito import Constant, Eq, Grid, Operator, TimeFunction, XDSLOperator, solve

from devito.operator.profiling import PerfEntry, PerfKey, PerformanceSummary


def equation_and_time_function(nx: int, ny: int, so: int, to: int, init_value: int):
    # Field initialization
    grid = Grid(shape=(nx, ny))
    u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
    # Create an equation with second-order derivatives
    a = Constant(name="a")
    eq = Eq(u.dt, a * u.laplace + 0.01)
    stencil = solve(eq, u.forward)
    eq0 = Eq(u.forward, stencil)

    return (grid, u, eq0)


def run_operator(op: Operator, nt: int, dt: float):
    res = op.apply(time_M=nt, a=0.1, dt=dt)
    assert isinstance(res, PerformanceSummary)
    o = res[PerfKey("section0", None)]
    assert isinstance(o, PerfEntry)
    return o.time


def main(bench_name: str):
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument(
        "-d",
        "--shape",
        default=(11, 11),
        type=int,
        nargs="+",
        help="Number of grid points along each axis",
    )
    parser.add_argument(
        "-so",
        "--space_order",
        default=2,
        type=int,
        help="Space order of the simulation",
    )
    parser.add_argument(
        "-to", "--time_order", default=1, type=int, help="Time order of the simulation"
    )
    parser.add_argument(
        "-nt", "--nt", default=10, type=int, help="Simulation time in millisecond"
    )
    parser.add_argument(
        "-bls", "--blevels", default=2, type=int, nargs="+", help="Block levels"
    )
    parser.add_argument("-xdsl", "--xdsl", default=False, action="store_true")
    parser.add_argument("--mpi", default=False, action="store_true", help="run in MPI mode")
    parser.add_argument("-nd", "--no_dump", default=False, action="store_true")
    args = parser.parse_args()

    nx, ny = args.shape
    nt = args.nt
    nu = 0.5
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    sigma = 0.25
    dt = sigma * dx * dy / nu

    so = args.space_order
    to = args.time_order

    init_value = 10

    grid, u, eq0 = equation_and_time_function(nx, ny, so, to, init_value)

    u.data[:, :, :] = 0
    u.data[:, int(nx / 2), int(ny / 2)] = init_value
    u.data[:, int(nx / 2), -int(ny / 2)] = -init_value

    print("xdsl:", flush=True)

    xop = XDSLOperator([eq0])
    xop.apply(time_M=nt, a=0.1, dt=dt)

    x_data = u.data.copy()

    u.data[:, :, :] = 0
    u.data[:, int(nx / 2), int(ny / 2)] = init_value
    u.data[:, int(nx / 2), -int(ny / 2)] = -init_value

    print("devito:", flush=True)

    op = Operator([eq0])
    op.apply(time_M=nt, a=0.1, dt=dt)

    d_data = u.data

    #print("mean squared error:", ((d_data - x_data)**2).mean())
    #print("max abs error", np.abs((d_data - x_data)).max())
    #print("max abs val", max(np.abs(d_data).max(), np.abs(x_data).max()))

    if np.abs((d_data - x_data)).max() > 1e-5:
        print(f"Failure, max abs error too high: {np.abs((d_data - x_data)).max()}")


if __name__ == "__main__":
    bench_name = __file__.split(".")[0]
    main(bench_name)

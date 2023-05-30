import argparse
from functools import reduce
from math import prod
import os
import pathlib
from subprocess import PIPE, Popen
from typing import Any

import numpy as np

from devito import Constant, Eq, Grid, Operator, TimeFunction, XDSLOperator, solve
from devito.ir.ietxdsl.cluster_to_ssa import generate_launcher_base
from devito.logger import info
from devito.operator.profiling import PerfEntry, PerfKey, PerformanceSummary
import sys


CFLAGS = '-O3 -march=native -mtune=native'

CPU_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,finalize-memref-to-llvm,convert-scf-to-cf,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"'
OPENMP_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"'
GPU_PIPELINE = '"builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},${MEMREF_TO_LLVM_PASS}{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)"'

XDSL_CPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir"
XDSL_GPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-gpu"
XDSL_MPI_PIPELINE = "dmp-decompose-2d{slices=4},stencil-shape-inference,convert-stencil-to-ll-mlir,dmp-to-mpi{slices=4},lower-mpi"

MAIN_MLIR_FILE_PIPELINE = '"builtin.module(canonicalize, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64},finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"'

def get_equation(name:str, shape:tuple[int, ...], so: int, to: int, init_value: int):
    d = (2.0 / (n - 1) for n in shape)
    nu = .5
    sigma = .25
    dt = sigma * reduce(lambda a,b: a*b, d) / nu
    match name:
        case '2d5pt':
            nx, ny = shape
            # Field initialization
            grid = Grid(shape=shape)
            u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
            u.data[...] = 0
            u.data[..., int(nx / 2), int(ny / 2)] = init_value
            u.data[..., int(nx / 2), -int(ny / 2)] = -init_value

            # Create an equation with second-order derivatives
            a = Constant(name="a")
            eq = Eq(u.dt, a * u.laplace + 0.01)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case '3d_diff':
            nx, _, _ = shape
            grid = Grid(shape=shape, extent=(2., 2., 2.))
            u = TimeFunction(name='u', grid=grid, space_order=so)
            # init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
            u.data[...] = 0
            u.data[:, int(nx/2), ...] = 1

            a = Constant(name='a')
            # Create an equation with second-order derivatives
            eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case _:
            raise Exception("Unknown benchamark!")

    return (grid, u, eq0, dt)


def dump_input(input: TimeFunction, bench_name: str):
    input.data_with_halo[0,...].tofile(f'{bench_name}.input.data')


def dump_main_mlir(bench_name: str, grid: Grid, u: TimeFunction, xop: XDSLOperator, dt: float, nt: int):
    info("Main function " + bench_name + ".main.mlir")
    with open(bench_name + ".main.mlir", "w") as f:
        f.write(
            generate_launcher_base(
                xop._module,
                {
                    "time_m": 0,
                    "time_M": nt,
                    **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
                    "a": 0.1,
                    "dt": dt,
                },
                u.shape_allocated[1:],
            )
        )

def compile_main(bench_name: str, grid: Grid, u: TimeFunction, xop: XDSLOperator, dt: float, nt: int):
    main = generate_launcher_base(
        xop._module,
        {
            "time_m": 0,
            "time_M": nt,
            **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
            "a": 0.1,
            "dt": dt,
        },
        u.shape_allocated[1:],
    )
    cmd = f'tee main.mlir | mlir-opt --pass-pipeline={MAIN_MLIR_FILE_PIPELINE} | mlir-translate --mlir-to-llvmir | clang -x ir -c -o {bench_name}.main.o - {CFLAGS} 2>&1'
    out:str
    try:
        print(f"Trying to compile {bench_name}.main.o with:")
        print(cmd)
        mlir_opt = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, env=os.environ)
        t = mlir_opt.communicate(main.encode())
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e

def compile_interop(bench_name: str, nodump:bool):
    cmd = f'clang -O3 -c interop.c -o {bench_name}.interop.o {CFLAGS} {"-DNODUMP" if nodump else ""} -DOUTFILE_NAME="\\"{pathlib.Path(__file__).parent.resolve()}/{bench_name}.stencil.data\\"" -DINFILE_NAME="\\"{pathlib.Path(__file__).parent.resolve()}/{bench_name}.input.data\\""'
    out:str
    try:
        print(f"Trying to compile {bench_name}.interop.o with:")
        print(cmd)
        mlir_opt = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, env=os.environ)
        t = mlir_opt.communicate()
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e
    
def compile_kernel(bench_name: str, mlir_code:str, xdsl_pipe:str, mlir_pipe:str):
    cmd = f'xdsl-opt -t mlir -p {xdsl_pipe} | mlir-opt --pass-pipeline={mlir_pipe} | mlir-translate --mlir-to-llvmir | clang -x ir -c -o {bench_name}.kernel.o - {CFLAGS}'
    out:str
    try:
        print(f"Trying to compile {bench_name}.kernel.o with:")
        print(cmd)
        mlir_opt = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, env=os.environ)
        t = mlir_opt.communicate(mlir_code.encode())
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e

def link_kernel(bench_name:str):
    cmd = f'clang {bench_name}.main.o {bench_name}.kernel.o {bench_name}.interop.o -o {bench_name}.out {CFLAGS}'
    out:str
    try:
        print(f"Trying to compile {bench_name}.out with:")
        print(cmd)
        mlir_opt = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, env=os.environ)
        t = mlir_opt.communicate()
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e

def run_kernel(bench_name:str, env:dict[str, Any] = {}) -> float:

    env_str = " ".join(k+'='+str(v) for k,v in env.items())
    cmd = f'{env_str} ./{bench_name}.out'
    out:str
    try:
        print(cmd)
        mlir_opt = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, env=os.environ)
        t = mlir_opt.communicate()
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
        xdsl_line = next(line for line in out.split("\n") if line.startswith("Elapsed time is: "))
        return float(xdsl_line.split(" ")[-2])

    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e


def run_operator(op: Operator, nt: int, dt: float) -> float:
    res = op.apply(time_M=nt, a=0.1, dt=dt)
    assert isinstance(res, PerformanceSummary)
    o = res[PerfKey("section0", None)]
    assert isinstance(o, PerfEntry)
    return o.time


def main(bench_name: str, nt:int, dump_main:bool, dump_mlir:bool):
    global CFLAGS
    grid, u, eq0, dt = get_equation(bench_name, args.shape, so, to, init_value)

    if args.openmp:
        CFLAGS += ' -fopenmp'
    if args.mpi:
        CFLAGS += ' -lmpi'

    if args.xdsl:
        dump_input(u, bench_name)
        xop = XDSLOperator([eq0])
        if dump_main:
            dump_main_mlir(bench_name, grid, u, xop, dt, nt)
        compile_main(bench_name, grid, u, xop, dt, nt)
        compile_interop(bench_name, args.no_output_dump)
        mlir_code = xop.mlircode
        if dump_mlir:
            info("Dump mlir code in  in " + bench_name + ".mlir")
            with open(bench_name + ".mlir", "w") as f:
                f.write(mlir_code)
        xdsl_pipeline = XDSL_MPI_PIPELINE if args.mpi else XDSL_CPU_PIPELINE
        compile_kernel(bench_name, mlir_code, xdsl_pipeline, CPU_PIPELINE)
        link_kernel(bench_name)
        rt = run_kernel(bench_name)
        
    else:
        op = Operator([eq0])
        rt = run_operator(op, nt, dt)
        print(f"Devito finer runtime: {rt} s")
        if args.no_output_dump:
            info("Skipping result data saving.")
        else:
            # get final data step
            # this is cursed math, but we assume that:
            #  1. Every kernel always writes to t1
            #  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
            #  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
            #  4. time_M is always nt in this example
            t1 = (nt + u._time_size - 1) % (2)

            res_data: np.array = u.data[t1,...]
            info("Save result data to " + bench_name + ".devito.data")
            res_data.tofile(bench_name + ".devito.data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument('benchmark_name', choices=["2d5pt", "3d_diff"])
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
    parser.add_argument("-nod", "--no_output_dump", default=False, action="store_true")
    parser.add_argument('--dump_mlir', default=False, action='store_true')
    parser.add_argument('--dump_main', default=False, action='store_true')
    parser.add_argument('--mpi', default=False, action='store_true')
    parser.add_argument('--openmp', default=False, action='store_true')

    
    args = parser.parse_args()

    benchmark_dim: int
    match args.benchmark_name:
        case '2d5pt':
            benchmark_dim = 2
        case '3d_diff':
            benchmark_dim = 3
        case _:
            raise Exception("Unhandled benchmark?")
    
    if len(args.shape) != benchmark_dim:
        print(f"Expected {benchmark_dim}d shape for this benchmark, got {args.shape}")
        sys.exit(1)
    

    so = args.space_order
    to = args.time_order

    init_value = 10
    main(args.benchmark_name,
         args.nt,
         args.dump_main,
         args.dump_mlir)

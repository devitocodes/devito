# Benchmarking / Correctness checking directory

This folder is where benchmarking and correctness checks will happen in.

## Structure

Each benchmark has a `name`, and it's devito definition is stored in `name.py`.

All intermediate files discussed in this section are valid `make` targets, to enable partial compilation and result inspection.

Running this file with `-xdsl` will generate all the stencil source files and input data.
The initial data will be stored in `<name>.input.data`, the stencil source in `<name>.main.mlir` and the "driver" file containing the main and timers in `<name>.mlir`.

The kernel file is then compiled to either `<name>.cpu.o` or `<name>.gpu.o`. This is controllable using the `MODE` variabe: `make 2d5pt.out MODE=gpu`

The main file is compiled to `main.o`, and the `interop.c` file is compiled to `<name>.interop.o`, to read the `<name>.input.data` file and write to the correct output.

The `<name>.out` file, the actual executable, is then compiled from these three object files.

Running devito and saving the output in `<name>.devito.data` is also a valid makefile target.

Finally, the `<name>.bench` will run both devito and the stencil version and compare the results.

## Usage

Passing options to the benchmark files is done with the `BENCH_OPTS="..."` variable for make.

`make -B 2d5pt.bench BENCH_OPTS="-d 100 100 -nt 100" MODE=CPU`
`make -B 3d_diff.bench BENCH_OPTS="-d 100 100 100 -nt 100" MODE=CPU`


To conclude, running the `2d5pt` example on `gpu` and compare the results, use:

`make 2d5pt.bench BENCH_OPTS="-d 1000 1000 -nt 1000" MODE=gpu`

Current modes are: `cpu`(default), `openmp`, `gpu` and `mpi`

## ToDos:

- Controlling devito omp flags / gpu usage is currently not done in the `Makefile`

## Passing environment variables to devito/omp

Prefixing the `make` command with `NAME=val` will make the variable `NAME` available to all stages in the make file.

Example:

```bash
DEVITO_ARCH=gcc DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG python 3d_diff.py -d 300 300 300 -nt 300 -xdsl
DEVITO_ARCH=gcc DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG python 3d_diff.py -d 300 300 300 -nt 300
make 3d_diff.bench BENCH_OPTS="-d 300 300 300 -nt 300" MODE=cpu
```

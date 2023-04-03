# Benchmarking / Correctness checking directory

This folder is where benchmarking and correctness checks will happen in.

## Structure

Each benchmark has a `name`, and it's devito definition is stored in `name.py`.

All intermediate files discussed in this section are valid `make` targets, to enable partial compialtion and result inspection.

Running this file with `-xdsl` will generate all the stencil source files and input data.
The initial data will be stored in `<name>.input.data`, the stencil source in `<name>.main.mlir` and the "driver" file containing the main and timers in `<name>.mlir`.

The kernel file is then compiled to either `<name>.cpu.o` or `<name>.gpu.o`. This is controllable using the `MODE` variabe: `make 2d5pt.out MODE=gpu`

The main file is compiled to `main.o`, and the `interop.c` file is compiled to `<name>.interop.o`, to read the `<name>.input.data` file and write to the correct output.

The `<name>.out` file, the actual executable, is then compiled from these three object files.

Running devito and saving the output in `<name>.devito.data` is also a valid makefile target.

Finally, the `<name>.bench` will run both devito and the stencil version and compare the results.

## Usage

passing options to the benchmark files is done with the `BENCH_OPTS="..."` variable for make.

To conclude, running the `2d5pt` example and compare the results, use:

`make 2d5pt.bench BENCH_OPTS="-d 1000 -d 1000 -nt 1000 MODE=gpu`

## ToDos:

 - Controlling devito is currently not done in the `Makefile`


Example:
```bash
DEVITO_ARCH=gcc DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG python fast/3d_diff.py -xdsl
```

Running it without `-xdsl` will use devito to solve the problem and save the result to `<name>.devito.data`

Example:
```bash
DEVITO_ARCH=gcc DEVITO_LANGUAGE=openmp DEVITO_LOGGING=DEBUG python fast/3d_diff.py
```

The xDSL/MLIR compiled programs will produce a `<name>.stencil.data` file with their results.

## Correctness

Setup:

 1. Make sure to write the input data (with halo!) out to `input.data` using `u.data_with_halo[0,:,:,:].tofile('input.data')`
    This is already done in the `diffusion_3D_wBCs.py` file, so you can check there for reference
 2. Also write the results devito computes to file (see example file as well)
 3. Dump the xDSL code for the example with the same parameters (using the `-xdsl` flag on the diffusion example)
 4. Compile and run the xdsl code using `compile-devito.py`
 5. Change the shape in the `compare.py` script to match
 6. Run `compare.py` to compare results

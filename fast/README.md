# Benchmarking / Correctness checking directory

This folder is where benchmarking and correctness checks will happen in.

## Structure

Each benchmark has a `name`, and it's devito definition is stored in `name.py`.

Running this file with `-xdsl` will generate all the stencil source files and input data.

Running it without `-xdsl` will use devito to solve the problem and save the result to `<name>.devito.data`

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

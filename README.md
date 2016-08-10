# Devito: Fast Finite Difference Computation

Devito is a new tool for performing optimised Finite Difference (FD)
computation from high-level symbolic problem definitions. Devito
performs automated code generation and Just-In-time (JIT) compilation
based on symbolic equations defined in
[SymPy](http://www.sympy.org/en/index.html) to create and execute
highly optimised Finite Difference kernels on multiple computer
platforms.

Devito is also intended to provide the driving operator kernels for a
prototype Full Waveform Inversion (FWI) code that can be found
[here](https://github.com/opesci/inversion).

## Quickstart

Devito should be installed directly from github via:
```
git clone https://github.com/opesci/devito.git
cd devito && pip install --user -r requirements.txt
```
Please make sure you also add Devito to your `PYTHONPATH`.

## Examples

At the core of the Devito API are so-called `Operator` objects that
allow users to create efficient FD kernels from SymPy expressions.
Examples of how to configure operators are provided:

* A simple example of how to solve the 2D diffusion equation can be
  found in `tests/test_diffusion.py`. This example also demonstrates
  how the equation can be solved via pure Python and optimised
  `numpy`, as well as Devito.
* A more practical example of Forward, Adjoint, Gradient and Born
  operators for use in FWI can be found in
  `examples/acoustic_example.py` and `examples/fwi_operators.py`.

## Compilation

Devito's JIT compiler engine supports multiple backends, with provided
presets for the most common compiler toolchains. By default Devito
will use the default GNU compiler `g++`, but other toolchains may be
selected by setting the `DEVITO_ARCH` environment variable to one of
the following values:
 * `gcc` or `gnu` - Standard GNU compiler toolchain
 * `clang` or `osx` - Mac OSX compiler toolchain via `clang`
 * `intel` or `icpc` - Intel compiler toolchain via `icpc`
 * `intel-mic` or `mic` - Intel Xeon Phi using offload mode via the
   `pymic` package

Please note that the toolchain can also be set from within Python
by setting the `compiler` argument on `Operator` objects:
```
op = Operator(..., compiler=IntelCompiler)
```

Thread parallel execution via OpenMP can also be enabled by setting
`DEVITO_OPENMP=1`.

## Cache Blocking

Devito supports loop cache blocking, which increases the effectiveness
of memory by reusing the data in the cache. To enable this feature
in `Operator` set `cache_blocking` to the size of the block you want to use.
`cache_blocking` can be a single `int` value which will block all dimensions
except outer most or a `list` of same size as spacial domain explicitly
stating which dims to block (x,y,z). If you do not want to block some
dimension, set `cache_blocking` to `None` respectively.

Example usage:
```
op = Operator(..., cache_blocking=[5, 10, None])
```
 
## Auto tuning block sizes

Devito supports automatic auto tuning of block sizes when cache blocking.
To enable auto tuning create `AutoTuner` object while passing `Operator`
and `auto_tuning_report_dir_path` as its arguments.
`AutoTuner` will run the compiled file multiple times with different block sizes,
trying to find most effective option, which will be written into report file.

If auto tuning has completed and you want to use best block sizes, initialise 
`Operator`  with `cache_blocking` set to as before and `at_report` arg 
pointing to your auto tuning report directory. Devito will attempt to 
read the auto tuning report and will select best `block_size` based on it.
If corresponding value is not found, sub optimal `block_size` will be chosen 
based on architecture.

Note: 
 This feature has to be used in conjunction with `cache_blocking` parameter.
 This feature needs to run only once for each model. 
 You can specify tuning range when calling `auto_tune_blocks(min, max)`
 function.

Example usage:
```
op = Operator(..., cache_blocking=[5, 10, None])
at = AutoTuner(op, <at_report_directory_path>)
at.auto_tune_blocks(min_block_size, max_block_size)
```

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

Devito can be installed from github via pip:
```
pip install --user git+https://github.com/opesci/devito.git
```

Alternatively Devito can be be installed manually from github via:
```
git clone https://github.com/opesci/devito.git
cd devito && pip install --user -r requirements.txt
```
When manually installing Devito please make sure you also add Devito
to your `PYTHONPATH`.

## Examples

At the core of the Devito API are so-called `Operator` objects that
allow users to create efficient FD kernels from SymPy expressions.
Examples of how to configure operators are provided:

* A simple example of how to solve the 2D diffusion equation can be
  found in `examples/diffusion/example_diffusion.py`. This example
  also demonstrates how the equation can be solved via pure Python and
  optimised `numpy`, as well as Devito.
* A more practical example of acoustic Forward, Adjoint, Gradient and Born
  operators for use in FWI can be found in
  `examples/acoustic/acoustic_example.py` and `examples/acoustic/fwi_operators.py`.
* A more practical example of TTI Forward
  operator for use in FWI can be found in
  `examples/tti/tti_example.py` and `examples/tti/tti_operators.py`.
* A benchmark example for the acoustic and TTI Forward can be found in
  `examples/benchmark.py`

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
except inner most or a `list` of same size as spacial domain explicitly
stating which dims to block (x,y,z). If you do not want to block some
dimension, set `cache_blocking` to `None` respectively. Furthermore, if
you want Devito to guess sub-optimal block size set `cache_blocking` or
a respective dimension to `0`.

Example usage:
```
op = Operator(..., cache_blocking=[5, 0, None])
```
 
## Auto tuning block sizes

Devito supports automatic auto tuning of block sizes when cache blocking.
To enable auto tuning create `AutoTuner` object while passing `Operator` as its 
argument. It also takes `blocked_dims` and `auto_tune_report_path` as optional args.
`AutoTuner` will run the compiled file multiple times with different block sizes,
trying to find most effective option, which will be written into report file.

If auto tuning has completed and you want to use best block size, pass
`cache_blocking` arg to `Operator` as `AutoTuner.block_size`
or explicitly set `operator.propagator.cache_blocking = AutoTuner.block_size`
Devito will attempt to read the auto tuning report and will select best 
`block_size` based on it. If corresponding value is not found or report does
not exist exception will be thrown

Note: 
 This feature needs to run only once for each model. Thus, you can pass 
 `AutoTuner.block_size` as `cache_blocking` arg without running auto tuning
  again as long as `at_report_dir` is provided correctly.
 `AutoTuner` has to run before `operator.apply()` is called.
 You can specify tuning range when calling `auto_tune_blocks(min, max)`
 function.

Example usage:
```
op = Operator(...)
at = AutoTuner(op, [True, True, False], <at_report_directory_path>)
at.auto_tune_blocks(min_block_size, max_block_size)

#using auto tuned block size
new_op = Operator(..., cache_blocking=at.block_size)
```

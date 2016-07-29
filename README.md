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
set `cache_blocking` flag to `True` in `Operator`. Furthermore you can
specify the block sizes using `block_size` parameter. It can be a single
number which will be used for all dimensions or a list explicitly stating
block sizes for each dim(x,y,z). If you do not want to block some dimensions, 
set `block_size` to `None` respectively.

Note
 If `block_size` is set to `None` or list of `None`'s
 cache blocking will be turned off.
 
Example usage:
```
op = Operator(..., cache_blocking=True, block_size=[5, 10, None])
```
 

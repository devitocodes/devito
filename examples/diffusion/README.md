# Devito diffusion example

A simple example demonstrating how to solve the 2D diffusion equation
using Devito's `Operator` API can be found in
`example_diffusion.py`. The example also includes demo implementations
using more traditional methods, including pure Python, vectorized
NumPy and a vectorized symbolic implementation using SymPy's
`lambdify()` functionality. For a visual demonstration and comparison
between pure Python and NumPy, for example, run:
```
python example_diffusion.py run -m python numpy --show
```
The above runs highlight the performance gains from using NumPy's
array vectorization, but the baseline test is still fairly small. To
demonstrate Devito's performance in relation to NumPy please run:
```
python example_diffusion.py run -m numpy devito --spacing 0.001 --timesteps 500
```

### Benchmarking and plotting

The diffusion example also provides a benchmarking mode that depends
on the [opescibench](https://github.com/opesci/opescibench) utility
package. As an example, on the benchmarking architecture you can run
the following command to generate performance data:
```
python example_diffusion.py bench -m numpy lambdify devito --spacing 0.001 -t 500 --resultsdir='results'
```
This will store the generated performance data in a sub-directory `results`, which can then be used to create a comparison plot with:
```
python example_diffusion.py plot -m numpy lambdify devito --spacing 0.001 -t 500 --resultsdir='results' --plotdir='plots'
```
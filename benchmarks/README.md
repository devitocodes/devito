# Benchmarking Devito

There are two subdirectories:

* `benchmarks/user`, which provides a Python script, `benchmark.py`, to
  evaluate the performance of some relevant Operators defined in `/examples`.
  `python benchmark.py --help` explains how to configure a benchmark run.
  Users interested in benchmarking Devito may want to explore this approach.
* `benchmarks/regression`, which provides a series of performance regression
  tests. This is used by continuous integration, to ensure that new features
  to be merged into trunk do not cause slowdowns. The performance regression
  framework is based on [airspeed velocity](https://asv.readthedocs.io/en/stable/).

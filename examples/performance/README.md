### Summary of optimization parameters

* Levels
  * `noop`: disable optimizations
  * `advanced`: all optimizations
  * `advanced-fsg`: alternative optimization pipeline

* Options (type, default)
  * Parallelism:
    * `openmp` (boolean, False): disable/enable OpenMP parallelism
  * Blocking:
    * `blockinner` (boolean, False): disable/enable loop blocking along innermost loop
    * `blocklevels` (int, 1): 1 => classic loop blocking; 2 for two-level hierarchical blocking; etc.
  * CIRE:
    * `min-storage` (boolean, False): disable/enable dimension contraction for working set size reduction
    * `cire-repeats-sops` (int, 5): control detection of sum-of-products
    * `cire-mincost-sops` (int, 10): minimum cost of a sum-of-product candidate
    * `cire-repeats-inv` (int, 1): control detection of dimension-invariants
    * `cire-mincost-inv` (int, 50): minimum cost of a dimension-invariant candidate

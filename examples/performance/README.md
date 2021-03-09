### Summary of optimization parameters

* Levels
  * `noop`: disable optimizations
  * `advanced`: all optimizations
  * `advanced-fsg`: alternative optimization pipeline

* Options (type, default)
  * Parallelism:
    * `openmp` (boolean, False): enable/disable OpenMP parallelism
    * `par-collapse-ncores` (int, 4): control loop collapsing
    * `par-collapse-work` (int, 100): control loop collapsing
    * `par-chunk-nonaffine` (int, 3): control chunk size in nonaffine loops
    * `par-dynamic-work` (int, 10): switch between dynamic and static scheduling
    * `par-nested` (int, 2): control nested parallelism
  * Blocking:
    * `blockinner` (boolean, False): enable/disable loop blocking along innermost loop
    * `blocklevels` (int, 1): 1 => classic loop blocking; 2 for two-level hierarchical blocking; etc.
  * CIRE:
    * `min-storage` (boolean, False): smaller working set size, less loop fusion
    * `cire-rotate` (boolean, False): smaller working set size, fewer parallel dimensions
    * `cire-maxpar` (boolean, False): bigger working set size, more parallelism
    * `cire-maxalias` (boolean, False): bigger working set size, better flop count
    * `cire-repeats-sops` (int, 5): control detection of sum-of-products
    * `cire-mincost-sops` (int, 10): minimum cost of a sum-of-product candidate
    * `cire-repeats-inv` (int, 1): control detection of dimension-invariants
    * `cire-mincost-inv` (int, 50): minimum cost of a dimension-invariant candidate
  * Device-specific:
    * `gpu-fit` (boolean, False): list of saved TimeFunctions that fit in the device memory
    * `gpu-direct` (boolean, False): generate code for optimized GPU-aware MPI
    * `par-disabled` (boolean, True): enable/disable parallelism on the host


### Optimization parameters by platform

* Parallelism

|                     |        CPU          |         GPU        |
|---------------------|---------------------|--------------------|
| openmp              | :heavy_check_mark:  | :heavy_check_mark: |
| par-collapse-ncores | :heavy_check_mark:  |         :x:        |
| par-collapse-work   | :heavy_check_mark:  |         :x:        |
| par-chunk-nonaffine | :heavy_check_mark:  | :heavy_check_mark: |
| par-dynamic-work    | :heavy_check_mark:  |         :x:        |
| par-nested          | :heavy_check_mark:  |         :x:        |

* Blocking

|                     |        CPU          |         GPU        |
|---------------------|---------------------|--------------------|
| blockinner          | :heavy_check_mark:  |         :x:        |
| blocklevels         | :heavy_check_mark:  |         :x:        |

* CIRE

|                     |        CPU          |         GPU        |
|---------------------|---------------------|--------------------|
| minstorage          | :heavy_check_mark:  |         :x:        |
| cire-rotate         | :heavy_check_mark:  |         :x:        |
| cire-maxpar         | :heavy_check_mark:  | :heavy_check_mark: |
| cire-maxalias       | :heavy_check_mark:  | :heavy_check_mark: |
| cire-repeats-sops   | :heavy_check_mark:  | :heavy_check_mark: |
| cire-mincost-sops   | :heavy_check_mark:  | :heavy_check_mark: |
| cire-repeats-inv    | :heavy_check_mark:  | :heavy_check_mark: |
| cire-mincost-inv    | :heavy_check_mark:  | :heavy_check_mark: |

* Device-specific

|                     |        CPU          |         GPU        |
|---------------------|---------------------|--------------------|
| gpu-fit             |        :x:          | :heavy_check_mark: |
| gpu-direct          |        :x:          | :heavy_check_mark: |
| par-disabled        |        :x:          | :heavy_check_mark: |

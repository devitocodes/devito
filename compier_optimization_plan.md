# Faster-Python Iteration 1: Temporary OSS Compilation Optimization Plan

## Status after current landed caching/reuse, fusion, and derivative-topofuse work

Current measured compile times on April 10, 2026 with PRO `faster-python-1`,
OSS `faster-python-1`, `devitopro-cuda:latest`, `--taskset 0-15`, and
`--deviceid 0`:

- `test_profile_etti_stress_like_schedule_infos`: `13.63 s`
- `test_profile_etti_velocity_then_stress_like_schedule_infos`: `32.00 s`

Compared with the March 30, 2026 no-cache baseline on the same probe family:

- stress-like: `29.99 s -> 13.63 s` (`-16.36 s`, about `54.5%`)
- velocity+stress: `98.49 s -> 32.00 s` (`-66.49 s`, about `67.5%`)

Compared with the later retrieve-accesses-only replay on the same branch family:

- stress-like: `29.32 s -> 13.63 s` (`-15.69 s`)
- velocity+stress: `93.16 s -> 32.00 s` (`-61.16 s`)

Compared with the pre-`TimedAccess` landed branch state:

- stress-like: `24.65 s -> 13.63 s` (`-11.02 s`)
- velocity+stress: `79.68 s -> 32.00 s` (`-47.68 s`)

Compared with the pre-space/rebuild branch (`3f3017e46`,
`compiler: Augment caching and memoization`):

- stress-like: `27.45 s -> 13.63 s` (`-13.82 s`)
- velocity+stress: `84.56 s -> 32.00 s` (`-52.56 s`)

The current landed branch now covers:

- narrow helper memoization and finite-difference evaluation caching
  (`3f3017e46`)
- `Scope`/access-inventory caching and lazy function-view reuse (`3f3017e46`)
- conservative space-object caching and no-op `Cluster.rebuild()` reuse
  (`e16f222e1`)
- cached `TimedAccess` construction and reuse of its per-instance distance cache
  across repeated `Scope` builds (`fd850927a`)
- synthetic `Scope.from_scopes(...)` construction from cached access summaries,
  plus fusion-hazard analysis over those synthetic scopes rather than fresh
  `Scope(exprs0 + exprs1)` rescans
- bounded derivative-driven topofusion in `lower_index_derivatives`, using the
  maximum nested `IndexDerivative.depth` as an upper bound on the number of
  productive `toposort='nofuse'` rounds before the final plain `fuse(False)`

Current profiled bottlenecks on the landed branch:

- stress-like:
  `lowering.Clusters ~= 8.45 s`, `lowering.IET ~= 3.64 s`,
  `optimize_kernels ~= 6.12 s`, `fuse ~= 0.53 s`
- velocity+stress:
  `lowering.Clusters ~= 23.82 s`, `lowering.IET ~= 5.64 s`,
  `specializing.Clusters ~= 19.24 s`, `fuse ~= 2.71 s`
- hottest Cluster-side buckets on velocity+stress:
  `optimize_kernels ~= 18.17 s`, with the remaining clean OSS cluster-side
  work still dominated by fusion/topofusion rather than the derivative
  lowering wrapper itself
- hottest IET-side buckets on velocity+stress:
  `make_parallel ~= 1.59 s`, `place_definitions ~= 1.33 s`,
  `_place_transfers ~= 0.86 s`, `linearization ~= 0.37 s`,
  `_generate_macros ~= 0.24 s`, `minimize_symbols ~= 0.25 s`,
  `optimize_halospots ~= 0.22 s`

Validation status of the latest derivative-topofuse heuristic:

- targeted OSS sensitivity checks around `test_unexpansion.py::{test_v3,test_v4,
  test_v5}` passed
- the previously failing PRO CUDA regression in compressed layered MPI
  serialization turned out to be unrelated to compilation changes; it was a
  `NVIDIA_VISIBLE_DEVICES`/implicit `deviceid` correctness bug and is now fixed
- current PRO `tests/test_gpu_lang.py::TestKernelOptDefault::
  test_flip_for_canonical_ordering` is failing on the `faster-python-1`
  PRO/OSS pair, but the failure hits the baseline `op0.apply(...)` path with an
  undefined `npthreads0` symbol in generated CUDA, so it currently looks like
  an OSS-side issue unrelated to the derivative-topofuse / `dsequences()`
  changes
- a full fresh OSS + PRO sweep has not yet been rerun after the current
  derivative-topofuse heuristic

This temporary note captures the main OSS-side compilation optimizations explored in
iteration 0. The list below is intentionally in ascending order of complexity:
smaller and safer caching/micro-optimization ideas come first, while broader
algorithmic and threading changes come later.

1. ~~Cache tiny pure helper results and other stable scalar metadata first.~~

   Completed in a narrow form in `3f3017e46`
   (`compiler: Augment caching and memoization`) via cached
   `IndexDerivative.pivot`, memoized `Derivative._eval_fd`, and shared
   numeric-weight reuse.

   Performance:
   this helper bucket was not isolated cleanly from point 5 in the squashed
   branch, but it is part of the landed `29.32 s -> 27.45 s` and
   `93.16 s -> 84.56 s` move.

   Rationale:
   these changes are local, easy to reason about, and usually do not alter the
   structure of the compiler pipeline.

2. ~~Preserve identity on no-op symbolic and visitor rewrites.~~

   Completed in a narrow compiler-local form in `e16f222e1`
   (`compiler: Augment caching and tweak memoization heuristics`) via
   `Cluster.rebuild()` returning `self` when all effective rebuild inputs are
   already identical objects.

   Performance:
   not isolated cleanly from point 9 below; the combined landed diff moved the
   probes from `27.45 s -> 24.65 s` and `84.56 s -> 79.68 s`.

   Rationale:
   this is still fairly contained work, but it starts touching generic traversal
   machinery that is used in many places.

3. Specialize traversal-heavy symbol discovery before changing higher-level
   algorithms.

   Relevant iteration-0 commits:
   `e450f0546` (`ir: trim findsymbols stack overhead`),
   `ceeb42689` (`ir: specialize findsymbols traversal`),
   `ff8d9efcc` (`symbolics: trim IET traversal overhead`).

   Rationale:
   these changes stay within existing traversal semantics, but they attack some
   of the hottest generic walks in lowering.

4. Reuse already computed inventories in IET cleanup and callable deduplication.

   Relevant iteration-0 commits:
   `f91d9256f` (`CODEX: ITER 6`, better caller tracking and cheaper param drops),
   `b05dd2084` (`iet: reuse symbol inventory in parameter updates`),
   `c26dbc3e6` (`WIP`, shared DataManager inventory collection and `reuse_efuncs`
   caches),
   `96ff77a94` (`iet: Prune reuse_efuncs by name family`).

   Current replay status:
   still WIP and intentionally not landed.

   April 7, 2026 replay findings on the current iteration-1 branch:
   rebuilding the non-WIP subset (`b05dd2084` + `f91d9256f` + `96ff77a94`)
   was correct on targeted `test_iet.py` / DSE checks, but the payoff was small
   relative to the extra engine/utils complexity.

   Performance:
   `b05dd2084` alone was flat-to-worse on the probes (`23.16 s -> 23.16 s` and
   `72.34 s -> 73.66 s`).
   Adding the two non-WIP `engine.py` follow-ups improved that to
   `23.16 s -> 22.01 s` and `72.34 s -> 72.03 s`.
   The subset was therefore dropped rather than landed: the light probe moved
   nicely, but the heavy probe improved by only about `0.31 s`.

   Rationale:
   this is the first bucket that spans multiple IET passes and shared helper
   caches, so it is more invasive than the previous purely local fast paths.

5. ~~Cheapen `Scope` construction and pairwise dependence pre-checks used by
   fusion/topofusion.~~

   Completed in `3f3017e46` via memoized `retrieve_accesses`, lazy cached
   `IREq` read/write inventories, and reuse of cached function views in
   `Scope`, `Cluster.traffic`, `Expression`, and `Operator`.

   Performance:
   the landed cache/memoization batch moved the probes from
   `29.32 s -> 27.45 s` and `93.16 s -> 84.56 s`. A narrower mid-iteration
   replay of the `Scope`/access portion alone had already reached roughly
   `27.65 s` and `86.21 s`. A later `TimedAccess` follow-up in `fd850927a`
   moved the landed branch further from `24.65 s -> 23.16 s` and
   `79.68 s -> 72.34 s`, for a total point-5-aligned move of roughly
   `29.32 s -> 23.16 s` and `93.16 s -> 72.34 s`.

   Rationale:
   these changes keep the same broad fusion algorithm, but they start replacing
   repeated rescans with cached summaries and synthetic scopes.

6. ~~Replace repeated generic fusion-hazard walks with focused hazard summaries,
   and tighten derivative-driven rescans.~~

   Relevant iteration-0 commits:
   `8c2e76a99` (`CODEX: ITER 5`, `fusion_hazards` summary),
   `024de93a2` (`clusters: Cheapen derivative topofusion hazards`),
   `0abbe2cb9` (`clusters: Restrict derivative nofuse rescans`).

   Completed on the current branch in a simpler form than the original iteration-0
   patches: fusion hazard analysis now reuses the already-cached per-ClusterGroup
   `Scope` inventories and synthesizes cross-scope dependences via
   `Scope.from_scopes(...)`, instead of repeatedly constructing fresh
   `Scope(exprs0 + exprs1)` objects from raw expressions. The derivative side is
   also now bounded: `lower_index_derivatives` runs at most `max_depth`
   `toposort='nofuse'` rounds, where `max_depth` is the maximum nested
   `IndexDerivative.depth` across the input clusters, and then finishes with the
   usual plain `fuse(False)`.

   Performance:
   compared with the pre-fusion landed state, the probes moved from
   `23.16 s -> 13.63 s` and `72.34 s -> 32.00 s`.

   Rationale:
   this turned out to be the dominant remaining algorithmic win after the earlier
   caching groundwork was in place. The essential gain is sparing repeated
   expression rescans during fusion/topofusion legality checks.

   Deferred April 17, 2026 follow-up:
   while profiling the PRO heavy `velocity_then_stress` compile on the paired
   OSS/PRO `faster-python-1` worktrees, `minimize_barrier_likelihood`
   consistently spent about `2.5-2.6 s` inside `fuse(toposort=True)`, with
   `_build_dag` and `_fusion_hazards` dominating that cost. A trial OSS patch
   in `Fusion._build_dag` skipped `_fusion_hazards` for unfenced ClusterGroup
   pairs whose scopes cannot possibly interact
   (`cg0.scope.writes.keys().isdisjoint(cg1.scope.functions)` and vice versa).

   Measured effect:
   `_fusion_hazards` calls dropped from about `47k` to about `5.7k`, and the
   barrier-minimization slice improved by about `0.12-0.17 s`, but the end-to-end
   heavy compile-time win was noisy and marginal. Focused OSS topofusion/barrier
   tests passed, but the change was still deferred rather than landed.

   Why deferred:
   this is exactly the kind of fast path that is easy to justify locally but
   hard to value globally. The measured win is real but small, and fusion/toposort
   is regression-prone enough that carrying extra control-flow in this area
   should require a clearer compile-time payoff.

   If revisited later:
   keep the prefilter in `_build_dag`, not inside `_fusion_hazards`.
   Moving it into `_fusion_hazards` would still pay the function-call and
   memoization overhead that the experiment was specifically avoiding, while
   `fenced` is a `_build_dag` scheduling concern rather than a property of the
   pairwise hazard relation itself.

7. Add concurrency inside expression lowering only after the single-threaded
   fast paths are understood.

   Relevant iteration-0 commits:
   `cd8bbec49` (`equations: Thread per-expression lowering`),
   `0f8d775c3` (`operator: Thread expression evaluation`).

   Rationale:
   threading can move the needle, but it also introduces option plumbing,
   scheduling questions, and failure modes that are harder to debug than the
   earlier single-threaded wins.

8. Add concurrency inside fusion/toposort last.

   Relevant iteration-0 commits:
   `e94ee8b52` (`CODEX: ITER 7`, `fuse-workers` and threaded DAG row building).

   Rationale:
   this depends on the earlier `Scope` and hazard-summary work, and it sits in a
   particularly regression-prone area of the compiler.

9. ~~Treat aggressive object/space caching as a late experiment, not an
   initial iteration-1 target.~~

   Completed in a conservative form in `e16f222e1` via cached
   `Interval`/`IterationSpace`-family objects, immutable/hashable `Properties`,
   `Prefix._preprocess_args`, and the no-op `Cluster.rebuild()` fast path
   above.

   Performance:
   compared with the pre-space/rebuild branch, the landed diff moved the probes
   from `27.45 s -> 24.65 s` and `84.56 s -> 79.68 s`.

   Rationale:
   iteration 0 showed that this class of optimization can improve compile-time
   behavior, but it also showed that the semantic risk is high enough that it
   should not be part of the first iteration-1 subset.

Regression-fix commits such as `cc6ee524a`, `6bc7ea1fd`, `9014e0ad0`, and
`d8981b0de` are intentionally not part of the ordered list above. They matter
for keeping iteration 0 green, but they are correctness follow-ups rather than
the primary optimization ideas to replay in iteration 1.

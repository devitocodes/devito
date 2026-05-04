# Faster-Python Iteration 1: Temporary OSS Compilation Optimization Plan

Benchmark nomenclature used below:

- `light`: `test_profile_eiso_stress_like_schedule_infos`
- `heavy`: `test_profile_etti_velocity_then_stress_like_schedule_infos`
- `heavy_IO`:
  `test_profile_etti_velocity_then_stress_like_bitcomp_serial_schedule_infos`
  (the `heavy` operator plus bitcomp compression and serialization)

## April 28, 2026 update: impact of the latest two OSS commits

The latest two OSS commits moved the paired PRO/OSS branch beyond the previous
April 21/22 `~26 s` heavy-compile checkpoint:

- `fc755479d` (`compiler: Split into EqBlock and Cluster`)
  - split the structural Cluster payload into cached `EqBlock` objects while
    preserving Cluster identity semantics
  - made `IREq` hashing/equality include IR metadata via `_hashable_content`,
    with the reusable `as_hashable` helper, so EqBlock caching does not merge
    equations that only differ in `ispace`, `conditionals`, `implicit_dims`, or
    `operation`
  - measured impact after correctness fixes:
    - stress-only: about `11.0 s`
    - heavy `velocity+stress`: about `24.4 s`
    - heavy `optimize_kernels`: about `11.3-11.4 s`
  - validation at this point included the targeted EqBlock repros, nearby
    equation/visitor/CSE tests, and a full OSS suite:
    `3549 passed, 5 skipped, 4 xfailed, 1 xpassed`

- `771f807ab` (`compiler: Stash hash were essential for compilation performance`)
  - added `cached_hash`, which stashes immutable-object `__hash__` results in
    `_mhash`
  - applied it to the hottest hash sites from the profiling investigation:
    support-space objects (`Interval`, `IntervalGroup`, `IterationInterval`,
    `IterationSpace`, `DataSpace`, and `IterationDirection`) plus Cluster queue
    keys (`Prefix`) and `ClusterGroup`
  - removed the generic `Space.__hash__` path and made subclasses hash their
    concrete payloads directly, avoiding shared base/subclass hash-cache
    ambiguity
  - measured impact after this commit:
    - stress-only: `10.45-10.57 s`, with `optimize_kernels 3.30-3.43 s`
    - heavy `velocity+stress`: `22.82-22.89 s`, with
      `optimize_kernels 10.43-10.50 s`
  - validation at this point included `test_lower_clusters.py + test_ir.py`,
    the four EqBlock/equation repros, and the two benchmark probes above; a
    full OSS suite has not yet been rerun after this second commit

Current practical checkpoint for the heavy benchmark is therefore now about
`22.8-22.9 s`, not the older `25.8-26.0 s` plateau. Relative to the previous
April 22 plateau, the latest two commits are worth roughly `3 s` on the heavy
compile, with the larger second-step gain coming from cached support-space and
Cluster queue hashing.

## Historical April 21 status after caching/reuse, fusion, and derivative-topofuse work

More recent paired PRO/OSS reruns on April 21, 2026 with PRO
`faster-python-1` at `b770aaee`, OSS `/home/fl1612/devito-faster-python-1` at
`30715c026`, `devitopro-cuda:latest`, `--taskset 0-15`, `--deviceid 3`, and
the two `schedule_infos` probes in
`devitopro/tests/test_kernelopt_nogil_tmp.py` reproduced the current practical
checkpoint as:

- `test_profile_eiso_stress_like_schedule_infos`: `5.81-5.83 s`
- `test_profile_etti_velocity_then_stress_like_schedule_infos`: `25.81-26.21 s`

At that point, the paired `velocity+stress` checkpoint for this branch family
was still about `26 s`; the older `32.00 s` value below is only a historical
April 10 milestone, and the April 28 section above supersedes both numbers for
the current branch.

## April 22, 2026 plateau note on the paired PRO/OSS branches

Fresh clean-`HEAD` reruns on April 22, 2026 with PRO `faster-python-1` at
`b770aaee`, OSS `/home/fl1612/devito-faster-python-1` at `c44c30339`,
`devitopro-cuda:latest`, `--taskset 0-15`, `--deviceid 3`, and the same
temporary PRO harness confirm that the practical checkpoint for the current
pair is still:

- light probe: still about `5.8 s`
- heavy `test_profile_etti_velocity_then_stress_like_schedule_infos`:
  still about `25.8-26.0 s` (`25.80 s`, `25.99 s` in fresh reruns)

Latest findings from the PRO-side deep profiling on this same pair:

- isolated `search_rotation` and `search_shm` postponement tweaks were both
  flat/noisy when rerun independently and were reverted
- more invasive scheduler and node-caching refactors were also dropped after
  either code-complexity growth or measurable regressions
- explicit per-kernel accounting on the heavy benchmark shows the dominant
  remaining cost is the initial stress kernel lineage, not the velocity kernel:
  - helper lineage: `0.402522 s`
  - velocity lineage: `0.585283 s`
  - stress lineage: `11.393894 s`
- splitting the first `optimize()` pass by initial kernel yields:
  - helper (`r0..r5`): `0.315156 s`
  - velocity (`v_x,v_y,v_z`): `0.426755 s`
  - stress (`tau_*`): `10.524947 s`
- the stress-kernel `optimize()` time is currently dominated by:
  - `action_apply`: `5.570871 s`
  - `minimize_barrier_likelihood`: `2.337799 s`
  - `reschedule`: `2.169853 s`
  - with `68` action steps, `69` schedule calls, and an average per-schedule
    state of about `229.7` IDG nodes and `30.4` queued actions
- current IET-side coarse buckets on the heavy probe are still:
  - `make_parallel ~= 1.99 s`
  - `place_definitions ~= 0.87 s`
  - `_place_transfers ~= 0.86 s`
  - `linearization ~= 0.37 s`

Current interpretation:
the current pair looks close to a local plateau on this benchmark family. The
next step should therefore be to move up in benchmark complexity rather than
keep forcing increasingly marginal scheduler micro-optimizations on the current
`velocity+stress` case.

First result after moving up in benchmark complexity on the PRO scratch harness:

- new `heavy_IO` benchmark:
  `test_profile_etti_velocity_then_stress_like_bitcomp_serial_schedule_infos`
- construction:
  extend the current `heavy` `velocity+stress` operator with one compressed,
  serialized saved `TimeFunction` per `tau_*` component and an extra
  `Eq(tau_*save, tau_*.forward)` per component
- first pinned compile result:
  - total compile: `30.44 s`
  - `lowering.Clusters`: `18.74 s`
  - `optimize_kernels`: `12.55 s`
  - `lowering.IET`: `9.39 s`
  - `specializing.IET`: `8.49 s`
  - main new IET-side buckets:
    `lower_async_objs ~= 1.29 s`,
    `place_definitions ~= 1.27 s`,
    `linearization ~= 1.12 s`,
    `_place_transfers ~= 0.92 s`

Interpretation:
the first heavier benchmark does become meaningfully slower, but the extra cost
lands primarily in IET / serialization-lowering rather than in the already
stress-dominated `kernelopt` slice.

Older measured compile times on April 10, 2026 with PRO `faster-python-1`,
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

   April 28, 2026 landed follow-up:
   the current branch now extends this bucket with the EqBlock/Cluster split in
   `fc755479d` and cached immutable-object hashes in `771f807ab`. This is the
   first object-caching follow-up in a while that clearly moved the main heavy
   benchmark rather than only shaving noise: the heavy `velocity+stress` probe
   moved from the previous `25.8-26.0 s` plateau through about `24.4 s` after
   EqBlock caching, then to `22.8-22.9 s` after cached support-space and
   Cluster queue hashes. The `cached_hash` result also confirms that repeated
   hashing was a real compile-time cost, not just profiling noise.

Regression-fix commits such as `cc6ee524a`, `6bc7ea1fd`, `9014e0ad0`, and
`d8981b0de` are intentionally not part of the ordered list above. They matter
for keeping iteration 0 green, but they are correctness follow-ups rather than
the primary optimization ideas to replay in iteration 1.

April 22, 2026 IET / bitcomp+serialization (`heavy_IO`) follow-up:

- New `heavy_IO` PRO scratch benchmark:
  start from the current `heavy` `velocity_then_stress` case and add one
  bitcomp+serialized saved `TimeFunction` per `tau_*` component.

- Paired clean baseline:
  about `30.3-31.2 s` total compile, with `optimize_kernels` still around
  `12.5-12.7 s` and the extra cost landing primarily in `lowering.IET`
  (`~9.4-9.7 s`).

- Profiling conclusions:
  `lower_async_objs` scanning is not the dominant new cost; the more relevant
  IET-side work is in `update_args` and in the second `place_definitions`
  pass triggered after `pthreadify`.

- Reverted experiment 1:
  simplify `engine.py:update_args` by collapsing the separate
  `FindSymbols('basics')` / `FindSymbols('symbolics')` scans and computing
  `drop_params` directly by index.

  Result:
  the compile-time probe looked mildly positive/noisy, but narrow
  compressed-layer runtime tests failed with the same
  `nbytes_avail_mapper` / `deviceid=-1` breakage, so this is not safe as-is.

- Reverted experiment 2:
  after `pthreadify`, rerun `place_definitions` only on callables touched by
  async lowering rather than across the whole graph.

  Result:
  this was the strongest local compile-time signal in the new `heavy_IO`
  benchmark:
  the second-epoch `place_definitions` visits dropped from `31` to `5`, and
  the heavy compile moved into roughly the `30.2-30.4 s` band.
  However, the same compressed-layer runtime tests failed, so the idea was
  reverted as well.

- Current recommendation:
  treat the async/definitions area as the right place to look for the new
  heavier benchmark, but do not carry either of the above optimizations
  without a stronger correctness story. The paired OSS worktree should stay at
  clean `HEAD`.

- April 22 late follow-up:
  a narrower post-`pthreadify` rerun of `place_definitions` does look viable
  after all. Instead of revisiting the whole graph, the current worktree now
  reruns the pass only on async-owned callables: the transformed
  `ThreadCallable`s, the helper callables (`activate*`, `init_sdata*`,
  `shutdown*`), and callers that reference those helpers. This is implemented
  by allowing `Graph.apply(..., targets=...)` and passing the selected names
  from `pthreadify`.

  Validation:
  the CPU layered async cases
  `tests/test_layered_funcs.py::TestSerialization::test_diskhost[...]` with
  `buf-async-degree=1` still pass on the paired worktrees.
  The higher-degree `buf-async-degree=4` variant remains baseline-red because
  of the pre-existing `npthreads0` codegen issue, so it is not a useful gate.

  Performance:
  on the `heavy_IO` bitcomp+serialization benchmark, the second-epoch
  `place_definitions` visits shrink from `31` down to `5`, and the local IET
  bucket improves from about `1.80 s` to about `1.57-1.66 s`.
  End-to-end compile time is a small but repeatable win on the latest paired
  reruns, moving from roughly `30.54 s` to about `30.24-30.49 s`.

April 30, 2026 IET memoization / no-op rebuild follow-up:

- Baseline before this IET-focused patch series:
  - `heavy`: `21.99 s`, `21.86 s`, `22.26 s`; average `22.04 s`.
    `lowering.IET`: `5.55 s`, `5.49 s`, `5.89 s`; average `5.64 s`.
  - `heavy_IO`: `26.38 s`, `26.42 s`, `26.51 s`; average `26.44 s`.
    `lowering.IET`: `8.98 s`, `8.96 s`, `9.54 s`; average `9.16 s`.

- Current simplified patch:
  - memoize public `create_call_graph`, with callers passing
    `as_hashable(self.efuncs)` / `as_hashable(efuncs)` rather than using a
    private cached helper;
  - memoize public `abstract_efunc`;
  - memoize public `abstract_objects` directly. `rg` across OSS and PRO shows
    no caller passes an explicit `sregistry`, so the old optional parameter was
    removed and the function now always uses its local `SymbolRegistry`;
  - simplify IET `reuse_if_unchanged` by using `Node._same_arg` instead of a
    duplicate local kwarg comparison helper.

- Dropped follow-up:
  a generic `memoized_func` key-path optimization was tested but left out of the
  patch. It appeared mildly positive in one set of runs, but was not necessary
  for the main IET win and is too broad for this focused change.

- Current measured performance with the simplified patch and unchanged
  `memoized_func`:
  - `heavy`: `21.63 s`, `21.53 s`, `21.57 s`; average `21.58 s`.
  - `heavy_IO`: `25.32 s`, `25.40 s`, `25.35 s`; average `25.36 s`.
  - net improvement versus the pre-patch reference is about `0.46 s` on
    `heavy` and about `1.08 s` on `heavy_IO`.

- Validation:
  targeted OSS IET/tool tests passed:
  `/app/devitopro/submodules/devito/tests/test_iet.py`,
  `/app/devitopro/submodules/devito/tests/test_visitors.py`,
  `/app/devitopro/submodules/devito/tests/test_tools.py` (`72 passed`).

- Interpretation:
  the durable win is in the IET callable-deduplication/reuse path, especially
  repeated call-graph creation and repeated abstraction of structurally stable
  callables. Dropping `abstract_objects` caching regressed `heavy_IO` back to
  roughly `25.5 s`, so that cache is worth keeping now that the unused
  `sregistry` parameter has been removed.

May 4, 2026 benchmark refresh after the no-op IET transform and visitor-cache
follow-ups:

- Setup:
  PRO `faster-python-1` worktree with paired OSS `faster-python-1`, CUDA docker
  image `devitopro-cuda:latest`, GPU device `3`, launcher pinned with
  `taskset 0-15`. The three schedule-info probes were run in one pytest-docker
  invocation.

- `stress-only` (`test_profile_etti_stress_like_schedule_infos`):
  - total compile: `10.06 s`;
  - `lowering.Clusters`: `5.52 s`;
  - `specializing.Clusters`: `4.18 s`;
  - `optimize_kernels`: `3.39 s`;
  - `lowering.IET`: `3.23 s`;
  - `specializing.IET`: `3.00 s`;
  - IET notable buckets: `make_parallel 1.59 s`,
    `_place_transfers 0.70 s`, `place_definitions 0.29 s`;
  - kernelopt `fuse`: `0.60 s`.

- `heavy` velocity+stress
  (`test_profile_etti_velocity_then_stress_like_schedule_infos`):
  - total compile: `21.63 s`;
  - `lowering.Clusters`: `14.69 s`;
  - `specializing.Clusters`: `11.17 s`;
  - `optimize_kernels`: `10.08 s`;
  - `lowering.IET`: `5.02 s`;
  - `specializing.IET`: `4.43 s`;
  - IET notable buckets: `make_parallel 1.47 s`,
    `_place_transfers 1.37 s`, `place_definitions 0.65 s`,
    `linearization 0.28 s`;
  - kernelopt `fuse`: `1.82 s`.

- `heavy_IO` velocity+stress plus bitcomp+serialization
  (`test_profile_etti_velocity_then_stress_like_bitcomp_serial_schedule_infos`):
  - total compile: `25.26 s`;
  - `lowering.Clusters`: `15.58 s`;
  - `specializing.Clusters`: `11.63 s`;
  - `optimize_kernels`: `10.53 s`;
  - `lowering.IET`: `7.59 s`;
  - `specializing.IET`: `6.85 s`;
  - IET notable buckets: `make_parallel 1.59 s`,
    `place_definitions 1.52 s`, `lower_async_objs 1.16 s`,
    `process 0.73 s`, `_place_transfers 0.54 s`,
    `linearization 0.47 s`;
  - kernelopt `fuse`: `1.95 s`.

- Interpretation:
  the three current probes are still in the expected post-IET-cache band:
  about `10.1 s` for stress-only, `21.5-21.7 s` for `heavy`, and
  `25.0-25.3 s` for `heavy_IO`. The `FindNodes` visitor cache reduced direct
  repeated visitor cost in profiling, but it remains a small/noise-level
  end-to-end compile-time effect. The dominant open costs are still
  `optimize_kernels`/cluster specialization and, for `heavy_IO`, the IET
  async/definitions path.

May 4, 2026 IET `reuse_efuncs` drill-down:

- The expensive IET buckets in `heavy_IO` (`make_parallel`,
  `place_definitions`, `_place_transfers`, `lower_async_objs`, and `process`)
  are mostly paying common `Graph.apply` post-processing cost rather than pass
  body cost. A temporary graph-phase profile showed:
  - `Graph.apply` total: about `7.33 s` across `25` calls;
  - `reuse_efuncs`: about `3.93 s` across `5` calls;
  - pass bodies: about `2.17 s`;
  - `update_args`: about `0.85 s`.

- Inside `reuse_efuncs`, the hot path is abstraction/signature generation:
  - before the new signature cache: `reuse_efuncs ~3.93 s`,
    `abstract_efunc ~1.91 s`, `_signature ~1.75 s`;
  - with IET `Node._signature()` memoized per node: `reuse_efuncs` drops to
    about `3.62-3.69 s`, and `_signature` drops to about `1.41-1.44 s`.

- The tested signature-cache patch was deliberately narrow:
  IET `Node` overrode `_signature()` with `@memoized_meth` and delegated to
  `Signer._signature()`, caching the SHA1 signature on the immutable-ish IET
  node instance without caching the full CIR string.

- Direct multiplicity check on `heavy_IO` showed why the patch is not a
  meaningful end-to-end win:
  - `_signature()` calls: `180`;
  - unique IET nodes: `150`;
  - repeated calls on the same node: only `30`;
  - call histogram: `121` nodes called once, `28` nodes called twice, `1` node
    called three times.

- The remaining `abstract_efunc` body cost is still substantial. A temporary
  body-level profile of `heavy_IO` showed about `150` misses and `30` hits
  across the five `reuse_efuncs` calls. Miss cost split roughly as:
  - `Uxreplace`: `0.63 s`;
  - `abstract_objects`: `0.63 s`;
  - `FindSymbols('basics|symbolics|dimensions')`: `0.23 s`.

- Dropped variants:
  - IET `Node._signature()` memoization was dropped after the multiplicity
    check. There are not enough repeated calls on the same node to justify even
    this small cache as a production change;
  - filtering identity mappings out of `abstract_objects` was slower in
    practice; `abstract_objects` increased from about `0.63 s` to about
    `1.62 s` in the instrumented run, because rebuilding the mapper dominated;
  - returning raw CIR from IET `Node._signature()` instead of the SHA1 digest
    was also rejected. It retains large strings and made the instrumented
    profile noisier/worse, without a clear wall-time win.

- Validation and benchmark signal from the rejected signature-cache patch:
  - targeted OSS IET/visitor tests still pass:
    `/app/devitopro/submodules/devito/tests/test_iet.py` and
    `/app/devitopro/submodules/devito/tests/test_visitors.py`
    (`42 passed`);
  - the earlier `heavy 22.25 s` combined-run sample was confirmed noisy and
    should be ignored.

- May 4 rerun, three combined invocations before and after the signature-cache
  patch, same setup (`devitopro-cuda:latest`, GPU `3`, `taskset 0-15`):
  - without signature cache:
    `stress-only 10.02/10.03/10.00 s` (avg `10.02 s`),
    `heavy 21.29/21.27/21.27 s` (avg `21.28 s`),
    `heavy_IO 24.80/24.66/24.63 s` (avg `24.70 s`);
  - with signature cache:
    `stress-only 10.02/10.03/9.98 s` (avg `10.01 s`),
    `heavy 21.36/21.40/21.29 s` (avg `21.35 s`),
    `heavy_IO 24.55/24.49/24.37 s` (avg `24.47 s`).

- Interpretation:
  memoizing IET node signatures is not worth keeping. The end-to-end signal is
  neutral for `stress-only`, neutral/slightly negative for `heavy`, and only
  mildly positive for `heavy_IO` (`~0.23 s`). The direct multiplicity check
  shows the cache surface is tiny: only `30/180` calls are repeated on the same
  node. The next meaningful IET win is unlikely to come from the individual
  pass bodies. It would need to reduce repeated `abstract_efunc` misses, likely
  by making `reuse_efuncs` more incremental/cache-aware across successive
  `Graph.apply` calls.

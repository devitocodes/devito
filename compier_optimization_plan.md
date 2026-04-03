# Faster-Python Iteration 1: Temporary OSS Compilation Optimization Plan

## Status after current landed caching/reuse work

Current measured compile times with PRO `faster-python-1`, OSS `faster-python-1`,
`devitopro-cuda:latest`, `--taskset 0-15`, and `--deviceid 0`:

- `test_profile_etti_stress_like_schedule_infos`: `23.16 s`
- `test_profile_etti_velocity_then_stress_like_schedule_infos`: `72.34 s`

Compared with the March 30, 2026 no-cache baseline on the same probe family:

- stress-like: `29.99 s -> 23.16 s` (`-6.83 s`, about `22.8%`)
- velocity+stress: `98.49 s -> 72.34 s` (`-26.15 s`, about `26.6%`)

Compared with the later retrieve-accesses-only replay on the same branch family:

- stress-like: `29.32 s -> 23.16 s` (`-6.16 s`)
- velocity+stress: `93.16 s -> 72.34 s` (`-20.82 s`)

Compared with the pre-`TimedAccess` landed branch state:

- stress-like: `24.65 s -> 23.16 s` (`-1.49 s`)
- velocity+stress: `79.68 s -> 72.34 s` (`-7.34 s`)

Compared with the pre-space/rebuild branch (`3f3017e46`,
`compiler: Augment caching and memoization`):

- stress-like: `27.45 s -> 23.16 s` (`-4.29 s`)
- velocity+stress: `84.56 s -> 72.34 s` (`-12.22 s`)

The current landed branch now covers:

- narrow helper memoization and finite-difference evaluation caching
  (`3f3017e46`)
- `Scope`/access-inventory caching and lazy function-view reuse (`3f3017e46`)
- conservative space-object caching and no-op `Cluster.rebuild()` reuse
  (`e16f222e1`)
- cached `TimedAccess` construction and reuse of its per-instance distance cache
  across repeated `Scope` builds (`fd850927a`)

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

6. Replace repeated generic fusion-hazard walks with focused hazard summaries,
   and tighten derivative-driven rescans.

   Relevant iteration-0 commits:
   `8c2e76a99` (`CODEX: ITER 5`, `fusion_hazards` summary),
   `024de93a2` (`clusters: Cheapen derivative topofusion hazards`),
   `0abbe2cb9` (`clusters: Restrict derivative nofuse rescans`).

   Rationale:
   this bucket produced meaningful wins, but it was also one of the first places
   where correctness regressions appeared. It should be revisited only after the
   simpler cache-based groundwork is back in place.

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

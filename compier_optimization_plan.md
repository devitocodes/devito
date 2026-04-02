# Faster-Python Iteration 1: Temporary OSS Compilation Optimization Plan

## Status after `compiler: Augment caching and memoization`

Current measured compile times with PRO `faster-python-1`, OSS `faster-python-1`,
`devitopro-cuda:latest`, `--taskset 0-15`, and `--deviceid 0`:

- `test_profile_etti_stress_like_schedule_infos`: `27.45 s`
- `test_profile_etti_velocity_then_stress_like_schedule_infos`: `84.56 s`

Compared with the March 30, 2026 no-cache baseline on the same probe family:

- stress-like: `29.99 s -> 27.45 s` (`-2.54 s`, about `8.5%`)
- velocity+stress: `98.49 s -> 84.56 s` (`-13.93 s`, about `14.1%`)

Compared with the later retrieve-accesses-only replay on the same branch family:

- stress-like: `29.32 s -> 27.45 s` (`-1.87 s`)
- velocity+stress: `93.16 s -> 84.56 s` (`-8.60 s`)

This iteration completed the `Scope`/access-inventory caching replay and a
narrow subset of the finite-difference helper memoization work. It also fixed
the build-lifetime leak introduced by compiler-scoped memoization by clearing
build-scoped caches after `Operator` construction and in `clear_cache()`.

This temporary note captures the main OSS-side compilation optimizations explored in
iteration 0. The list below is intentionally in ascending order of complexity:
smaller and safer caching/micro-optimization ideas come first, while broader
algorithmic and threading changes come later.

1. ~~Cache tiny pure helper results and other stable scalar metadata first.~~

   Completed in a narrow form in the current branch via cached
   `IndexDerivative.pivot`, memoized `Derivative._eval_fd`, and shared
   numeric-weight reuse.

   Relevant iteration-0 commits:
   `981ab5b89` (`compiler: Memoize distance`),
   `f49400743` (`ir: Cache TimedAccess instances`),
   `4817f16fb` (`finite-differences: Cache IndexDerivative pivot`),
   `de20783e6` (`finite-differences: Cache IndexDerivative pivot functions`),
   `4b5572663` (`finite-differences: Memoize derivative evaluation`),
   `eb9cfc161` (`finite-differences: Relax numeric weight cache keys`).

   Rationale:
   these changes are local, easy to reason about, and usually do not alter the
   structure of the compiler pipeline.

2. Preserve identity on no-op symbolic and visitor rewrites.

   Relevant iteration-0 commits:
   `e01862ac3` (`tools: cache generic visitor handlers`),
   `11220b982` (`symbolics: fast-path untouched uxreplace leaves`),
   `ecc90f11e` (`symbolics: avoid rebuilding untouched uxreplace containers`),
   `a0f779421` (`iet: Preserve identity on no-op visitor rewrites`),
   `f31d2e0c7` (`CODEX: ITER 8`, no-op `Transformer` / `Uxreplace` fast paths).

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

   Completed in the current branch via memoized `retrieve_accesses`, lazy
   cached `IREq` read/write inventories, and reuse of cached function views in
   `Scope`, `Cluster.traffic`, `Expression`, and `Operator`.

   Relevant iteration-0 commits:
   `0f4355875` (`CODEX: ITER 1`, cached read/write target sets and `may_interact`),
   `dc9408808` (`CODEX: ITER 2`, reuse `getreads` and avoid repeated generators),
   `f56e22998` (`CODEX: ITER 3`, synthetic pair scopes built from cached accesses),
   `ca76e0472` (`CODEX: ITER 4`, memoized access extraction helpers).

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

9. Treat aggressive object/space caching as a late experiment, not an initial
   iteration-1 target.

   Relevant iteration-0 commits:
   `d67aab676` (`ir: cache space objects conservatively`),
   `b52716dfa` (revert of the above).

   Rationale:
   iteration 0 showed that this class of optimization can improve compile-time
   behavior, but it also showed that the semantic risk is high enough that it
   should not be part of the first iteration-1 subset.

Regression-fix commits such as `cc6ee524a`, `6bc7ea1fd`, `9014e0ad0`, and
`d8981b0de` are intentionally not part of the ordered list above. They matter
for keeping iteration 0 green, but they are correctness follow-ups rather than
the primary optimization ideas to replay in iteration 1.

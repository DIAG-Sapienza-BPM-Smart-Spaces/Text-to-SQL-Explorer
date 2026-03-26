## Plan: Demo Paper Pipeline Rework

Rework the pipeline around new inputs/outputs while preserving the core ideas: use real BIRD Development metrics for the two visualizations, generate realistic fake data for other datasets, switch selector evaluation to persisted LLM-judge pairwise comparisons, and split embedding flow into precomputing embeddings and (in a separate file) the similirity matrices with aggregate statistic. Then, in the first visualizion, perform real-time clustering from precomputed matrices for Embedder selector and ranking winner for LLM-as-selector from precomputed pairwise comparisons.

**Steps**

1. Phase 1: Baseline contracts and schemas.
2. Define a single canonical metric contract used everywhere: execution_accuracy, exact_match, sql_f1_score, response_schema_f1_score, cell_f1_score. _blocks all later phases_
3. Define canonical model ID inventory by scanning metrics and sql candidate sources, including all currently available models (deepseek-chat, cogito_70b, qwen2.5-coder_32b, qwen3-coder_30b, codellama_70b, codestral_22b, sqlcoder_15b, plus ground_truth only when needed as reference). _blocks selector + fake data + visualization wiring_
4. Define new output artifact schemas and paths (non-breaking new files/folders): pairwise judge results, fake pairwise results, precomputed embeddings, and precomputed similarity matrices. _blocks Phases 2-5_
5. Phase 2: Visualization data-source refactor (real + fake unified).
6. Refactor model/metric loading in first visualization to be data-driven (no hardcoded 4-model list) and to prioritize real BIRD Development metrics from metrics_results. _depends on 1-4_
7. Map metrics_results JSON fields into canonical metric names and percentages consistently for charts/tables; remove legacy metric fields from visualization controls. _depends on 6_
8. Update second visualization (binary_visualization.py) metric fields and loaders to the same canonical 5 metrics so both visualizations stay aligned. _parallel with 7 after 6_
9. Keep fake data as fallback for non-BIRD datasets only; merge real and fake records through one normalized loader path for both visualizations. _depends on 7-8_
10. Phase 3: Fake data generators expansion.
11. Update existing fake visualization generator to produce records for all models and only canonical metrics across non-BIRD datasets. _depends on 1-4_
12. Add a new fake generator dedicated to selector-like pairwise judge outputs (query-level matchup decisions and metadata), so non-BIRD datasets can mimic real pairwise artifacts. _parallel with 11_
13. Add deterministic seeding/signature strategy and dataset/model coverage checks so fake outputs are stable across runs and complete for visualization ingestion. _depends on 11-12_
14. Phase 4: Selector rework to pairwise LLM judging.
15. Modify selector pipeline to create pairwise prompts per query over candidate-model pairs (candidate-only comparisons, excluding ground_truth from pairwise competition), using a judge model (default deepseek-chat now, configurable later). _depends on 1-4_
16. Persist all pairwise judgments per query (winner/loser/tie metadata, prompt/response traces as appropriate, participating model IDs), without collapsing to only final selected candidate. _depends on 15_
17. Define new function to derive per-query leaderboard from persisted pairwise outcomes (wins, losses, ties, win rate, optional tie-break policy) and expose chosen candidate as a derived view, not the sole stored output. _depends on 16_
18. Update first visualization to load pairwise outcomes and compute the metrics of the LLM-as-selector given the chosen candidates group and metrics. Ties between best-performing candidates are broken at random. _depends on 16-17 + Phase 2 loader changes_
19. Phase 5: Embedder pipeline split into precompute + interactive clustering.
20. Add a precompute stage that encodes SQL candidates for all models and saves reusable embeddings to npz + json metadata. _depends on 1-4_
21. Precompute similarity matrices for BIRD Development query/model candidate sets and persist them as reusable artifacts. _depends on 20_
22. Update embedding selector pipeline to load precomputed matrices rather than recomputing from scratch, with artifact validation/invalidation when source SQL inputs change. _depends on 20-21_
23. In first visualization, when user selects embedder selector mode, perform real-time clustering via compute_similarity_groups_pairwise using the precomputed similarity matrices for active query/model subset; keep threshold strategy based on average similarity. _depends on 21-22 + Phase 2 loader changes_
24. Phase 6: Integration and migration safeguards.
25. Remove any deprecated features or code paths. _depends on 2-5_
26. Add schema validators for new artifacts and clear error messaging in visualization when required files are missing/incompatible. _parallel with 25_
27. Update README/documentation with new pipeline commands, file contracts, and real-vs-fake data boundaries. _depends on 25-26_
28. Phase 7: Verification.
29. Run selector on a small BIRD subset and verify pairwise artifacts contain full matchup coverage for each query (n*(n-1)/2 comparisons for n candidate models used). *depends on 15-18\*
30. Run fake generators and verify non-BIRD datasets receive complete model x metric and model-pair coverage with deterministic reproducibility.
31. Validate first visualization: barplots computed from BIRD real metrics for canonical 5 metrics, selector-derived rows populated from pairwise outputs, embedder mode clustering recomputed live from precomputed matrices.
32. Validate second visualization (binary_visualization.py) field population and metric alignment with first visualization.
33. Perform spot checks comparing raw artifact values to displayed values (unit conversion, model labels, metric name mapping).

**Relevant files**

- c:/Users/adria/GitHub/Demo-Paper/first_visualization.py — dynamic model/metric loading, real+fake merge path, embedder interactive clustering hook.
- c:/Users/adria/GitHub/Demo-Paper/binary_visualization.py — second visualization metric field alignment to canonical five.
- c:/Users/adria/GitHub/Demo-Paper/generate_fake_visualization_data.py — expanded fake metric generation across all models with canonical metrics.
- c:/Users/adria/GitHub/Demo-Paper/selector.py — pairwise judge orchestration and persisted per-query matchup outputs.
- c:/Users/adria/GitHub/Demo-Paper/selector_pre_calculation.py — derived leaderboard/wins summaries from raw pairwise judgments. Must be renamed to reflect new role as real-time selector from precomputed data and given set of candidate models.
- c:/Users/adria/GitHub/Demo-Paper/common_utils.py — shared schema helpers, pairwise indexing, canonical metric utilities.
- c:/Users/adria/GitHub/Demo-Paper/embedding.py — reuse compute_similarity_groups_pairwise and add persistence helpers as needed.
- c:/Users/adria/GitHub/Demo-Paper/embedding_pipeline_selection.py — switch to precomputed artifacts and artifact integrity checks.
- c:/Users/adria/GitHub/Demo-Paper/metrics_results/ — authoritative real BIRD Development metrics input.
- c:/Users/adria/GitHub/Demo-Paper/sqls/ — candidate SQL source per model. Should not be used.
- c:/Users/adria/GitHub/Demo-Paper/README.md — pipeline and artifact contract documentation.
- c:/Users/adria/GitHub/Demo-Paper (new folders) — precomputed/, pairwise_results/, fake_data/selector_pairwise/.

**Verification**

1. Schema validation: each new artifact passes required key/type checks and expected cardinalities.
2. Coverage checks: for each dataset/query, all expected models exist in visualization metric tables; for each selector query, all expected model pairs are present.
3. Metric sanity checks: canonical metrics remain within [0,1] in storage and convert correctly to percentages in plots.
4. Consistency checks: first and second visualization expose the same metric set and model IDs.
5. Embedder checks: precompute artifacts are reused (no redundant encoding in normal run), and real-time clustering reacts correctly to query/model filter changes.
6. Smoke run commands: fake generation, selector pairwise generation, precompute embeddings/matrices, visualization startup.

**Decisions**

- Second visualization target is binary_visualization.py.
- Pairwise comparisons are among candidate models only (ground_truth excluded from pairwise competition).
- Pairwise winner is decided by an LLM judge prompt per matchup; default judge is deepseek-chat but configurable.
- Raw pairwise judgments are primary persisted output; final selected candidate is a derived view.
- Embedding persistence format is npz + json metadata.
- New non-breaking output locations are preferred over overwriting legacy artifacts.

**Further Considerations**

1. Judge prompt/versioning: persist prompt template version and judge model/version in each pairwise record for reproducibility.
2. Efficiency of offline processing: consider batch processing for pairwise judgments and embedding precomputations to speed up pipeline runs. During heavy offline computation phases (selector pairwise generation, embedding precompute), provide progress logging and estimated time to completion, as well as checkpointing intermediate results to allow resumption in case of interruptions.
3. Performance limits: bound real-time clustering latency in UI by caching active-subset matrix slices and avoiding repeated full-matrix scans.

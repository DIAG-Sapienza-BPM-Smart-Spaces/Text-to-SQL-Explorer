# Demo Paper - Code Documentation

## Overview

This repository evaluates and compares multiple text-to-SQL candidate systems, then analyzes selection strategies for choosing among candidate SQL outputs.

Primary execution paths:

- Binary LLM judging pipeline.
- Pairwise-selector LLM pipeline.
- Embedding-based selector pipeline.

Supporting assets:

- Streamlit visualizations for performance exploration.
- Unified fake-data generation for missing visualization scenarios.
- SQL normalization and schema retrieval utilities.

## Repository Structure

### Core Pipelines

- `llm_as_judge.py`: Binary ACCEPT/REJECT judging against candidate SQLs.
- `selector.py`: Pairwise selector pipeline with persisted raw model-vs-model judgments.
- `selector_realtime_ranking.py`: Derives runtime selector metrics from persisted pairwise judgments.
- `embedding.py`: SQL embedding and clustering utilities.
- `embedding_pipeline_selection.py`: End-to-end embedding selector evaluation.
- `precompute_embeddings.py`: Offline embedding precomputation.
- `precompute_similarity_matrices.py`: Offline similarity matrix precomputation.

### Data Utilities

- `retrieve_database_info.py`: Dataset/table JSON loading and schema formatting.
- `sql_cleaner.py`: Cleans model SQL outputs in `candidates/` and normalizes text artifacts.

### Visualization

- `first_visualization.py`: Main performance dashboard with query filtering, model metrics, and selector overlays.
- `binary_visualization.py`: Detailed binary-judge exploration view.

### Generated Data Artifacts

- `all_pairwise_comparisons.json`: Ground-truth pairwise metrics source.
- `binary_choices/`: True binary judge outputs.
- `selectors/`: True selector outputs and selector-vs-ground-truth metrics.
- `embedding_results/`: True embedding selection outputs.
- `fake_data/`: Unified fake outputs used to fill missing scenarios.
- `cache_results/`: On-disk cache for expensive visualization preprocessing.

## Fake Data System

### Unified Generator

- `generate_fake_visualization_data.py`

Responsibilities:

- Generate deterministic fake execution metrics for missing model/dataset combinations.
- Generate deterministic fake pairwise-selector judgments for all selector models.
- Generate deterministic fake embedding selector outputs.
- Generate deterministic fake binary judge outputs.

Main outputs:

- `fake_data/fake_generation_bundle.json`
- `fake_data/fake_execution_metrics.json`
- `fake_data/fake_selector_pairwise_results.json`
- `fake_data/fake_embedding_selection.json`
- `fake_data/fake_binary_choices.json`

## Visualization Data Flow

### Main Dashboard (`first_visualization.py`)

- Loads and normalizes query datasets from `datasets_files/`.
- Computes query-level table/attribute/length features from SQL and schema.
- Uses true metrics when available and merges fake fallback data when true data is absent.
- Supports selector overlays for:
  - Pairwise-selector judge models (winner derived from persisted pairwise outcomes).
  - Embedding selector.

### Binary Dashboard (`binary_visualization.py`)

- Reads true binary outputs from `binary_choices/`.
- Uses `fake_data/fake_binary_choices.json` as fallback coverage.
- Enriches rows with query metadata when available.

## Caching Strategy

### Streamlit Cache

- `@st.cache_data` is used for expensive load/transform stages.

### Disk Cache (`cache_results/`)

- `first_visualization.py` writes and reads preprocessing cache files.
- Cache invalidation uses file signatures based on path, size, and mtime.
- Cached artifacts currently include:
  - Per-split SQL stats cache files (`sql_stats_*.json`).
  - Full preprocessed query dataframe (`queries_cache.pkl` + `queries_cache_meta.json`).

## Operational Notes

- True and fake artifacts are intentionally coexisting so visualizations can cover all datasets and selector/model combinations.
- If source dataset files change, signature-based invalidation rebuilds cached query artifacts automatically.
- If behavior looks stale in Streamlit, clear cache and rerun the app.

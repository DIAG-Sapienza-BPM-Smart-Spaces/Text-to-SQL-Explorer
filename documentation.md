# Demo Paper - Code Documentation

## Overview
This repository evaluates and compares multiple text-to-SQL candidate systems, then explores two selection strategies:
- LLM-as-judge selection (`llm_as_judge.py` and `selector.py`)
- Embedding-based consensus selection (`embedding_pipeline_selection.py`)

It also contains:
- Data generation scripts for synthetic/demo datasets (`generate_results.py`, `generate_fake_visualization_data.py`)
- SQL cleanup utilities (`sql_cleaner.py`)
- Visualization apps built with Streamlit (`first_visualization.py`, `binary_visualization.py`)

## Main Pipeline Components

### 1. Data and Schema Access
- `retrieve_database_info.py`
- Responsibilities:
  - Load JSON files for queries/tables.
  - Retrieve schema by `db_id`.
  - Format schema into prompt-friendly structure used by judge/selector prompts.

### 2. SQL Cleaning
- `sql_cleaner.py`
- Responsibilities:
  - Remove markdown code fences from model outputs.
  - Decode escaped whitespace.
  - Remove control characters.
  - Normalize whitespace.
  - Add `clean_sql` fields to SQL result JSON files in `sqls/`.

### 3. LLM Binary Judging
- `llm_as_judge.py`
- Responsibilities:
  - Load candidate SQL outputs (`sqls/*_query_results.json`).
  - For each query, run an LLM prompt in binary mode (`ACCEPT`/`REJECT`).
  - Attach pairwise comparison metadata vs. ground truth.
  - Save per-candidate-model output in `binary_choices/`.

### 4. Single Selector (4-way Choice)
- `selector.py`
- Responsibilities:
  - Load 4 candidate SQLs per query.
  - Ask selector LLM to pick best candidate (`CANDIDATE 1..4`).
  - Map selected candidate to pairwise metrics against ground truth.
  - Save selector decisions and metadata.

### 5. Selector Performance Aggregation
- `selector_pre_calculation.py`
- Responsibilities:
  - Read selector outputs.
  - Resolve selected candidate vs. ground-truth metrics from pairwise results.
  - Produce aggregate metrics and per-query diagnostics.

### 6. Embedding-Based Selection
- `embedding.py`
- Responsibilities:
  - Load SQL embedder and cache it.
  - Parse and normalize SQL (`sqlglot`).
  - Build cosine-similarity matrix and cluster SQL candidates.
  - Select query closest to centroid of largest cluster.

- `embedding_pipeline_selection.py`
- Responsibilities:
  - Run embedding-based selection for all `general_id` entries.
  - Attach ground-truth pairwise metrics.
  - Compare ground-truth embedding against selected cluster.
  - Save per-query results and aggregate ground-truth cluster stats.

### 7. Visualization
- `first_visualization.py`
  - Multi-metric performance and selector analysis dashboard.
- `binary_visualization.py`
  - Interactive inspection of binary judge decisions and ground-truth metrics.

## Redundant Code Found and Refactor Applied

### Redundancy Identified
The following logic appeared repeatedly in different scripts:
- JSON file loading.
- Atomic JSON writes via temp-file replacement.
- Pairwise index construction by `(db_id, question_id)`.
- Ground-truth comparison extraction for a candidate model.

### Shared Reusable Module Added
- New file: `common_utils.py`
- Shared functions:
  - `resolve_path(...)`
  - `load_json(...)`
  - `atomic_dump_json(...)`
  - `build_pairwise_index(...)`
  - `extract_ground_truth_comparison(...)`

### Scripts Updated to Reuse Shared Code
- `selector.py`
  - Uses `atomic_dump_json`, `build_pairwise_index`, and `extract_ground_truth_comparison` from `common_utils.py`.
- `llm_as_judge.py`
  - Uses same shared helpers for consistency.
- `embedding_pipeline_selection.py`
  - Uses shared JSON read/write helpers.
- `selector_pre_calculation.py`
  - Uses shared JSON load helper.
- `retrieve_database_info.py`
  - Uses shared JSON load helper in `load_json_file`.

## Why This Improves Readability and Ease of Use
- Single source of truth for core file/pairwise utility behavior.
- Less duplicated code to maintain.
- Lower risk of diverging logic between judge and selector pipelines.
- Easier future extension (new scripts can import the same helpers).

## Typical Execution Flow
1. Clean SQL outputs (optional but recommended):
   - Run `sql_cleaner.py`.
2. Produce binary judge outputs:
   - Run `llm_as_judge.py`.
3. Produce selector outputs:
   - Run `selector.py`.
4. Aggregate selector-vs-ground-truth metrics:
   - Run `selector_pre_calculation.py`.
5. Run embedding selection analysis:
   - Run `embedding_pipeline_selection.py`.
6. Open visual dashboards:
   - `streamlit run first_visualization.py`
   - `streamlit run binary_visualization.py`

## Notes
- Some scripts currently include inline API keys in `__main__`; these should be moved to environment variables for safety.
- Existing outputs in `results/`, `selectors/`, `binary_choices/`, and `sqls/` are treated as data artifacts consumed by the pipelines and visualizations.

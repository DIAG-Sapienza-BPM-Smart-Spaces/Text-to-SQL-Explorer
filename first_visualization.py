import streamlit as st
import json
import os
import pandas as pd
import hashlib
import re

models = ["DeepSeek Chat", "Qwen2.5 Coder 32B", "Qwen3 Coder 30B", "Cogito 70b"]
datasets = ["BIRD Training", "BIRD Developer", "SPIDER"]

MODEL_TO_SYSTEM_ID = {
    "DeepSeek Chat": "deepseek-chat",
    "Qwen2.5 Coder 32B": "qwen2.5-coder_32b",
    "Qwen3 Coder 30B": "qwen3-coder_30b",
    "Cogito 70b": "cogito_70b",
}
SYSTEM_ID_TO_MODEL = {v: k for k, v in MODEL_TO_SYSTEM_ID.items()}

PAIRWISE_METRICS = [
    "schema_precision",
    "schema_recall",
    "cell_value_accuracy",
    "row_set_jaccard",
    "execution_accuracy",
    "f1_score",
]

SELECTOR_SOURCE_LABEL = "DeepSeek selector (GT)"
EMBEDDING_SELECTOR_SOURCE_LABEL = "Embedding selector"

DATASET_FLAGS = {
    "BIRD Training": False,
    "BIRD Developer": True,
    "SPIDER": False,
}

SQL_KEYWORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "full", "outer", "on", "and", "or",
    "not", "in", "is", "null", "as", "case", "when", "then", "else", "end", "distinct", "order",
    "by", "group", "having", "limit", "offset", "union", "all", "exists", "between", "like", "asc",
    "desc", "cast", "real", "integer", "count", "sum", "avg", "min", "max", "over", "partition",
    "rank", "row_number", "dense_rank", "with", "recursive", "cross", "using", "into"
}


def string_to_deterministic_int(s):
    """Convert a string to a deterministic integer using SHA256 hash.
    This allows faster equality comparisons than string comparisons.
    """
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)


def _clean_identifier(token):
    if token is None:
        return ""
    cleaned = str(token).strip().strip('`"[]').strip().lower()
    return cleaned


def _extract_sql_identifier_candidates(sql_text):
    """Extract potential identifier tokens (tables/columns) from SQL text."""
    if not sql_text:
        return set()

    candidates = set()

    # Quoted identifiers: `Column Name`, "Column Name", [Column Name]
    for token in re.findall(r'`([^`]+)`|"([^"]+)"|\[([^\]]+)\]', sql_text):
        value = next((part for part in token if part), "")
        cleaned = _clean_identifier(value)
        if cleaned and cleaned != "*":
            candidates.add(cleaned)

    # FROM/JOIN table references
    for table_name in re.findall(r'\b(?:from|join)\s+([a-zA-Z_][\w]*)\b', sql_text, flags=re.IGNORECASE):
        cleaned = _clean_identifier(table_name)
        if cleaned:
            candidates.add(cleaned)

    # Dotted references like T1.column_name
    for column_name in re.findall(r'\b[a-zA-Z_][\w]*\s*\.\s*([a-zA-Z_][\w]*)\b', sql_text):
        cleaned = _clean_identifier(column_name)
        if cleaned:
            candidates.add(cleaned)

    # Unqualified identifiers (later filtered through schema names and SQL keywords)
    for token in re.findall(r'\b[a-zA-Z_][\w]*\b', sql_text):
        cleaned = _clean_identifier(token)
        if cleaned and cleaned not in SQL_KEYWORDS:
            candidates.add(cleaned)

    return candidates


def _build_schema_lookup(dev_tables_payload):
    """Build DB-specific sets of table and column names."""
    lookup = {}
    if not isinstance(dev_tables_payload, list):
        return lookup

    for db_schema in dev_tables_payload:
        if not isinstance(db_schema, dict):
            continue

        db_id = db_schema.get('db_id')
        if not db_id:
            continue

        table_names = {
            _clean_identifier(name)
            for name in db_schema.get('table_names_original', [])
            if _clean_identifier(name)
        }

        column_names = set()
        for col in db_schema.get('column_names_original', []):
            if not isinstance(col, (list, tuple)) or len(col) < 2:
                continue
            col_name = _clean_identifier(col[1])
            if col_name and col_name != "*":
                column_names.add(col_name)

        lookup[db_id] = {
            'tables': table_names,
            'columns': column_names,
        }

    return lookup


def _compute_sql_length_bucket(values, num_buckets=4):
    """Convert raw SQL lengths to discrete buckets (0..num_buckets-1)."""
    if not values:
        return []

    if len(set(values)) <= 1:
        return [0] * len(values)

    # Rank first to avoid duplicate-edge issues in qcut, then bucket by quantiles.
    ranked = pd.Series(values).rank(method='first')
    effective_buckets = max(1, min(num_buckets, len(values)))
    buckets = pd.qcut(ranked, q=effective_buckets, labels=False)
    return buckets.astype(int).tolist()


@st.cache_data
def load_bird_dev_ground_truth_stats():
    """Precompute table/attribute/length stats from BIRD Developer ground-truth SQL once."""
    dev_path = 'dev.json'
    tables_path = 'dev_tables.json'
    if not os.path.exists(dev_path) or not os.path.exists(tables_path):
        return {}

    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_payload = json.load(f)
    with open(tables_path, 'r', encoding='utf-8') as f:
        dev_tables_payload = json.load(f)

    schema_lookup = _build_schema_lookup(dev_tables_payload)
    stats_by_gid = {}
    pending_rows = []
    raw_lengths = []

    for row in dev_payload:
        if not isinstance(row, dict):
            continue

        db_id = row.get('db_id')
        query_id = row.get('question_id')
        sql_text = row.get('ground_truth') or row.get('SQL') or ""

        schema_info = schema_lookup.get(db_id, {})
        table_set = schema_info.get('tables', set())
        column_set = schema_info.get('columns', set())

        tokens = _extract_sql_identifier_candidates(str(sql_text))
        table_count = len(tokens.intersection(table_set)) if table_set else 0
        attribute_count = len(tokens.intersection(column_set)) if column_set else 0
        sql_length = len(re.findall(r'\S+', str(sql_text)))

        gid = string_to_deterministic_int(f"BIRD Developer|{db_id}|{query_id}")
        pending_rows.append((gid, table_count, attribute_count, sql_length))
        raw_lengths.append(sql_length)

    discrete_lengths = _compute_sql_length_bucket(raw_lengths, num_buckets=4)
    for idx, (gid, table_count, attribute_count, _) in enumerate(pending_rows):
        stats_by_gid[gid] = {
            'tables': int(table_count),
            'attributes': int(attribute_count),
            'length': int(discrete_lengths[idx]) if idx < len(discrete_lengths) else 0,
        }

    return stats_by_gid

# Load query datasets
@st.cache_data
def load_queries():
    """Load query datasets from JSON files with BIRD Developer aligned to real dev.json IDs."""
    queries_list = []
    
    try:
        # Load BIRD Training queries
        with open('datasets/bird_training_queries.json', 'r', encoding='utf-8') as f:
            q_data = pd.DataFrame(json.load(f))
            q_data['dataset'] = 'BIRD Training'
            queries_list.append(q_data)
        
        # Load BIRD Developer queries from real benchmark source
        with open('dev.json', 'r', encoding='utf-8') as f:
            raw_dev = pd.DataFrame(json.load(f))
            difficulty_map = {
                'simple': 0,
                'moderate': 1,
                'challenging': 2,
            }
            q_data_developer = pd.DataFrame({
                'id': raw_dev['question_id'],
                'query': raw_dev['question'],
                'database': raw_dev['db_id'],
                'complexity': raw_dev['difficulty'].map(difficulty_map).fillna(0).astype(int),
                'length': 0,
                'tables': 0,
                'attributes': 0,
            })
            q_data_developer['dataset'] = 'BIRD Developer'

            # Precomputed once (cached): use reconstructed general id to map table/attribute counts.
            dev_stats = load_bird_dev_ground_truth_stats()
            q_data_developer['g_id'] = (
                q_data_developer['dataset'].astype(str) + '|' +
                q_data_developer['database'].astype(str) + '|' +
                q_data_developer['id'].astype(str)
            ).apply(string_to_deterministic_int)
            q_data_developer['tables'] = q_data_developer['g_id'].map(
                lambda gid: dev_stats.get(gid, {}).get('tables', 0)
            ).astype(int)
            q_data_developer['attributes'] = q_data_developer['g_id'].map(
                lambda gid: dev_stats.get(gid, {}).get('attributes', 0)
            ).astype(int)
            q_data_developer['length'] = q_data_developer['g_id'].map(
                lambda gid: dev_stats.get(gid, {}).get('length', 0)
            ).astype(int)

            queries_list.append(q_data_developer)
        
        # Load SPIDER queries
        with open('datasets/spider_queries.json', 'r', encoding='utf-8') as f:
            q_data_spider = pd.DataFrame(json.load(f))
            q_data_spider['dataset'] = 'SPIDER'
            queries_list.append(q_data_spider)
        
        # Concatenate all dataframes at once for efficiency
        queries_data = pd.concat(queries_list, ignore_index=True) if queries_list else pd.DataFrame()
        return queries_data
    except FileNotFoundError as e:
        st.error(f"Error loading query files: {e}")
        return pd.DataFrame()

# Load queries on app start
all_queries = load_queries()
databases = {
    dataset_name: sorted(
        all_queries[all_queries['dataset'] == dataset_name]['database'].dropna().unique().tolist()
    )
    for dataset_name in datasets
}

# Load model results
@st.cache_data
def load_model_results():
    """Load model-vs-ground-truth pairwise metrics for all candidate models."""
    all_rows = []
    seen = set()
    
    try:
        with open('all_pairwise_comparisons.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for query_data in json_data.values():
            query_id = query_data.get('question_id')
            database = query_data.get('db_id')

            for comparison in query_data.get('comparisons', {}).values():
                system1 = comparison.get('system1')
                system2 = comparison.get('system2')

                if system1 == 'ground_truth' and system2 in SYSTEM_ID_TO_MODEL:
                    model_name = SYSTEM_ID_TO_MODEL[system2]
                elif system2 == 'ground_truth' and system1 in SYSTEM_ID_TO_MODEL:
                    model_name = SYSTEM_ID_TO_MODEL[system1]
                else:
                    continue

                for metric_key in PAIRWISE_METRICS:
                    metric_value = comparison.get(metric_key)
                    if not isinstance(metric_value, (int, float)):
                        continue

                    # Pairwise metrics are in [0,1], normalize to percentage scale for chart consistency.
                    if metric_value <= 1.0:
                        metric_value *= 100.0

                    dedup_key = (query_id, database, model_name, metric_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    all_rows.append({
                        'dataset': 'BIRD Developer',
                        'database': database,
                        'id': query_id,
                        'model': model_name,
                        'metric': metric_key,
                        'value': metric_value,
                    })
        
        # Create DataFrame from all rows
        results_data = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        return results_data
    except FileNotFoundError as e:
        st.error(f"Error loading results files: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error parsing results files: {e}")
        return pd.DataFrame()


@st.cache_data
def load_selector_ground_truth_results():
    """Load precomputed DeepSeek selector performance metrics against ground truth."""
    file_path = 'selectors/deepseek-chat_selector_performance_vs_ground_truth.json'
    if not os.path.exists(file_path):
        return pd.DataFrame()

    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    for row in payload.get('per_query', []):
        query_id = row.get('question_id')
        database = row.get('db_id')
        for metric_key in PAIRWISE_METRICS:
            metric_value = row.get(metric_key)
            if not isinstance(metric_value, (int, float)):
                continue
            if metric_value <= 1.0:
                metric_value *= 100.0
            rows.append({
                'dataset': 'BIRD Developer',
                'database': database,
                'id': query_id,
                'model': SELECTOR_SOURCE_LABEL,
                'metric': metric_key,
                'value': metric_value,
            })

    return pd.DataFrame(rows)


@st.cache_data
def load_embedding_selector_results():
    """Load embedding selector performance metrics against ground truth."""
    file_path = 'embedding_pipeline_selection_results.json'
    if not os.path.exists(file_path):
        return pd.DataFrame()

    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        return pd.DataFrame()

    for row in payload:
        if not isinstance(row, dict):
            continue

        query_id = row.get('question_id')
        database = row.get('db_id')
        metrics = row.get('ground_truth_comparison_metrics') or {}

        for metric_key in PAIRWISE_METRICS:
            metric_value = metrics.get(metric_key)
            if not isinstance(metric_value, (int, float)):
                continue

            if metric_value <= 1.0:
                metric_value *= 100.0

            rows.append({
                'dataset': 'BIRD Developer',
                'database': database,
                'id': query_id,
                'model': EMBEDDING_SELECTOR_SOURCE_LABEL,
                'metric': metric_key,
                'value': metric_value,
            })

    return pd.DataFrame(rows)

# Load results on app start
all_results = load_model_results()
all_selector_results = load_selector_ground_truth_results()
all_embedding_selector_results = load_embedding_selector_results()

# Create a general_id column in both dataframes for quick lookup using vectorized operations
if not all_queries.empty:
    # Vectorized string concatenation and hash creation when g_id is not already present.
    if 'g_id' not in all_queries.columns:
        all_queries['g_id'] = (all_queries['dataset'].astype(str) + '|' + 
                               all_queries['database'].astype(str) + '|' + 
                               all_queries['id'].astype(str)).apply(string_to_deterministic_int)
    else:
        missing_gid_mask = all_queries['g_id'].isna()
        if missing_gid_mask.any():
            all_queries.loc[missing_gid_mask, 'g_id'] = (
                all_queries.loc[missing_gid_mask, 'dataset'].astype(str) + '|' +
                all_queries.loc[missing_gid_mask, 'database'].astype(str) + '|' +
                all_queries.loc[missing_gid_mask, 'id'].astype(str)
            ).apply(string_to_deterministic_int)
if not all_results.empty:    
    # Vectorized string concatenation and hash creation
    all_results['g_id'] = (all_results['dataset'].astype(str) + '|' + 
                           all_results['database'].astype(str) + '|' + 
                           all_results['id'].astype(str)).apply(string_to_deterministic_int)
if not all_selector_results.empty:
    all_selector_results['g_id'] = (
        all_selector_results['dataset'].astype(str) + '|' +
        all_selector_results['database'].astype(str) + '|' +
        all_selector_results['id'].astype(str)
    ).apply(string_to_deterministic_int)
if not all_embedding_selector_results.empty:
    all_embedding_selector_results['g_id'] = (
        all_embedding_selector_results['dataset'].astype(str) + '|' +
        all_embedding_selector_results['database'].astype(str) + '|' +
        all_embedding_selector_results['id'].astype(str)
    ).apply(string_to_deterministic_int)

def collect_active_results(active_queries, selected_models, selected_metrics_dict, selected_selectors_dict=None):
    """
    Collect all relevant results for active queries based on selections.
    
    Args:
        active_queries: DataFrame of active queries
        selected_models: List of selected candidate model names
        selected_metrics_dict: Dictionary of selected metrics {metric_key: bool}
        selected_selectors_dict: Dictionary of selected selectors {selector_key: bool}
    
    Returns:
        DataFrame with columns: query_id, dataset, database, model, metric, agent, value
    """
    if active_queries.empty:
        return pd.DataFrame()
    
    # Get active query IDs using pandas Series
    active_g_ids = active_queries['g_id'] if 'g_id' in active_queries.columns else pd.Series(dtype='int64')
    
    # Get selected metrics using pandas-friendly list comprehension
    selected_metric_keys = [m for m, selected in selected_metrics_dict.items() if selected]
    
    # Start with candidate-model rows when available.
    active_results = pd.DataFrame()
    if not all_results.empty:
        active_results = all_results[
            all_results['g_id'].isin(active_g_ids) &
            all_results['model'].isin(selected_models) &
            all_results['metric'].isin(selected_metric_keys)
        ].copy()

    # Build selector-derived metric rows if selector mode is active.
    if selected_selectors_dict is None:
        selected_selectors_dict = {}

    selector_rows = []
    selector_enabled = selected_selectors_dict.get('single_selector', False)
    embedding_selector_enabled = selected_selectors_dict.get('embedding_selector', False)
    all_four_selected = set(selected_models) == set(models)

    if selector_enabled and all_four_selected and selected_metric_keys and not all_selector_results.empty:
        selector_df = all_selector_results[
            all_selector_results['g_id'].isin(active_g_ids) &
            all_selector_results['metric'].isin(selected_metric_keys)
        ].copy()
        if not selector_df.empty:
            selector_rows = selector_df.to_dict(orient='records')

    if embedding_selector_enabled and all_four_selected and selected_metric_keys and not all_embedding_selector_results.empty:
        embedding_selector_df = all_embedding_selector_results[
            all_embedding_selector_results['g_id'].isin(active_g_ids) &
            all_embedding_selector_results['metric'].isin(selected_metric_keys)
        ].copy()
        if not embedding_selector_df.empty:
            selector_rows.extend(embedding_selector_df.to_dict(orient='records'))

    if selector_rows:
        selector_df = pd.DataFrame(selector_rows)
        if active_results.empty:
            active_results = selector_df
        else:
            active_results = pd.concat([active_results, selector_df], ignore_index=True)

    return active_results


metrics = [
    {"name": "Schema Precision", "key": "schema_precision", "default": True, "tooltip": "Precision on schema elements used by predicted SQL vs ground truth."},
    {"name": "Schema Recall", "key": "schema_recall", "default": True, "tooltip": "Recall on schema elements used by predicted SQL vs ground truth."},
    {"name": "Cell Value Accuracy", "key": "cell_value_accuracy", "default": True, "tooltip": "Correctness of result cell values against ground truth output."},
    {"name": "Row Set Jaccard", "key": "row_set_jaccard", "default": True, "tooltip": "Overlap between predicted and ground-truth result rows."},
    {"name": "Execution Accuracy", "key": "execution_accuracy", "default": True, "tooltip": "Execution-level correctness compared with ground truth."},
    {"name": "F1 Score", "key": "f1_score", "default": True, "tooltip": "Harmonic mean balancing precision and recall for result correctness."},
]

selectors = [
    {"name": "Single LLM Selector (DeepSeek)", "key": "single_selector", "default": False, "enabled": True, "tooltip": "Prototype mode: plots DeepSeek selector precomputed metrics against ground truth."},
    {"name": "Embedding Selector", "key": "embedding_selector", "default": False, "enabled": True, "tooltip": "Plots embedding pipeline selector metrics from embedding_pipeline_selection_results.json."},
    {"name": "Ensemble of LLMs", "key": "llms_ensemble", "default": False, "enabled": False, "tooltip": "Disabled in prototype: no real datapoints available yet."}
]

# Color definitions - organized by color families
model_colors = {
    "DeepSeek Chat": "#246BCE",
    "Qwen2.5 Coder 32B": "#1C9A79",
    "Qwen3 Coder 30B": "#F39C12",
    "Cogito 70b": "#C0392B"
}

dataset_colors = {
    "BIRD Training": "#E85D04",  # Dark orange
    "BIRD Developer": "#FF9500",  # Bright orange
    "SPIDER": "#FFBB66"  # Light orange
}

metric_colors = {
    "schema_precision": "#0F4C81",
    "schema_recall": "#2E86AB",
    "cell_value_accuracy": "#00A878",
    "row_set_jaccard": "#F18F01",
    "execution_accuracy": "#D1495B",
    "f1_score": "#6C5CE7",
}

selectors_colors = {
    "single_selector": "#6A1B9A",  # Purple
    "embedding_selector": "#8E44AD",  # Violet
    "llms_ensemble": "#AB47BC"  # Light purple
}

st.set_page_config(layout="wide")

# Define CSS styles for colors
st.markdown("""
<style>
/* Model colors - Blue family */
.color-deepseek { color: #246BCE !important; }
.color-qwen25 { color: #1C9A79 !important; }
.color-qwen3 { color: #F39C12 !important; }
.color-cogito { color: #C0392B !important; }

/* Dataset colors - Orange family */
.color-bird-training { color: #E85D04 !important; }
.color-bird-developer { color: #FF9500 !important; }
.color-spider { color: #FFBB66 !important; }

/* Metric colors - Green family */
.color-schema-precision { color: #0F4C81 !important; }
.color-schema-recall { color: #2E86AB !important; }
.color-cell-value-accuracy { color: #00A878 !important; }
.color-row-set-jaccard { color: #F18F01 !important; }
.color-execution-accuracy { color: #D1495B !important; }
.color-f1-score { color: #6C5CE7 !important; }
            
/* Selector colors Grey/Purple family */
.color-single-selector { color: #6A1B9A !important; }
.color-embedding-selector { color: #8E44AD !important; }
.color-llms-ensemble { color: #AB47BC !important; }

/* Background colors for conditional boxes */
.bg-bird-training { background-color: rgba(232, 93, 4, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-bird-developer { background-color: rgba(255, 149, 0, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-spider { background-color: rgba(255, 187, 102, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-tdex { background-color: rgba(106, 27, 154, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-embedding-selector { background-color: rgba(142, 68, 173, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-llms-ensemble { background-color: rgba(171, 71, 188, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-llm-judge { background-color: rgba(206, 147, 216, 0.15) !important; padding: 10px; border-radius: 5px; }

/* Utility */
.text-item { margin-top: -5px; }

/* Custom tooltips with larger font size */
.tooltip-container {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip-container .tooltip-text {
    visibility: hidden;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 10px 15px;
    position: absolute;
    z-index: 1000;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    white-space: normal;
    width: 300px;
    font-size: 17px;
    line-height: 1.4;
    opacity: 0;
    transition: opacity 0.3s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.tooltip-container .tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #333 transparent transparent transparent;
}

.tooltip-container:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# Helper functions to get CSS class names
def get_model_class(model_name):
    class_map = {
        "DeepSeek Chat": "color-deepseek",
        "Qwen2.5 Coder 32B": "color-qwen25",
        "Qwen3 Coder 30B": "color-qwen3",
        "Cogito 70b": "color-cogito",
    }
    return class_map.get(model_name, "")

def get_dataset_class(dataset_name):
    class_map = {
        "BIRD Training": "color-bird-training",
        "BIRD Developer": "color-bird-developer",
        "SPIDER": "color-spider"
    }
    return class_map.get(dataset_name, "")

def get_dataset_bg_class(dataset_name):
    class_map = {
        "BIRD Training": "bg-bird-training",
        "BIRD Developer": "bg-bird-developer",
        "SPIDER": "bg-spider"
    }
    return class_map.get(dataset_name, "")

def get_metric_class(metric_key):
    class_map = {
        "schema_precision": "color-schema-precision",
        "schema_recall": "color-schema-recall",
        "cell_value_accuracy": "color-cell-value-accuracy",
        "row_set_jaccard": "color-row-set-jaccard",
        "execution_accuracy": "color-execution-accuracy",
        "f1_score": "color-f1-score",
    }
    return class_map.get(metric_key, "")

def get_selector_class(selector_key):
    class_map = {
        "single_selector": "color-single-selector",
        "embedding_selector": "color-embedding-selector",
        "llms_ensemble": "color-llms-ensemble"
    }
    return class_map.get(selector_key, "")

def filter_queries(queries_df, length_range, tables_range, attributes_range=None):
    """
    Filter queries based on length, tables, and attributes involved ranges.
    
    Args:
        queries_df: DataFrame of queries
        length_range: Tuple (min, max) for length
        tables_range: Tuple (min, max) for tables
        attributes_range: Tuple (min, max) for attributes (optional)
    
    Returns:
        Filtered DataFrame of queries
    """
    if queries_df.empty:
        return queries_df
    
    # Use pandas boolean indexing for efficient filtering
    mask = (
        (queries_df['length'] >= length_range[0]) & 
        (queries_df['length'] <= length_range[1]) &
        (queries_df['tables'] >= tables_range[0]) & 
        (queries_df['tables'] <= tables_range[1])
    )
    
    # Check attributes if range is provided
    if attributes_range is not None and 'attributes' in queries_df.columns:
        mask &= (
            (queries_df['attributes'] >= attributes_range[0]) & 
            (queries_df['attributes'] <= attributes_range[1])
        )
    
    return queries_df[mask].copy()

def get_queries_for_databases(dataset_name, database_names):
    """
    Get all queries for the specified databases within a dataset.
    
    Args:
        dataset_name: Name of the dataset
        database_names: List of database names
    
    Returns:
        DataFrame of queries for the specified databases
    """
    if all_queries.empty:
        return pd.DataFrame()
    
    # Use pandas boolean indexing to filter
    mask = (all_queries['dataset'] == dataset_name) & (all_queries['database'].isin(database_names))
    return all_queries[mask].copy()

def collect_all_selected_queries(selected_datasets_list, selected_databases_list):
    """
    Collect all queries from selected datasets and databases.
    
    Args:
        selected_datasets_list: List of selected dataset names
        selected_databases_list: List of selected database names
    
    Returns:
        DataFrame of all selected queries
    """
    if all_queries.empty or not selected_datasets_list:
        return pd.DataFrame()
    
    # Use pandas boolean indexing for efficient filtering
    dataset_mask = all_queries['dataset'].isin(selected_datasets_list)
    
    # If databases are selected, filter by them; otherwise include all databases
    if selected_databases_list:
        database_mask = all_queries['database'].isin(selected_databases_list)
        combined_mask = dataset_mask & database_mask
    else:
        combined_mask = dataset_mask
    
    return all_queries[combined_mask].copy()

def get_query_ranges(queries_df):
    """
    Calculate min/max ranges for query attributes.
    
    Args:
        queries_df: DataFrame of queries
    
    Returns:
        Dictionary with min/max values for each attribute
    """
    if queries_df.empty:
        return {
            'length': (0, 3),
            'tables': (0, 30),
            'attributes': (0, 30)
        }
    
    # Use pandas vectorized min/max operations
    return {
        'length': (int(queries_df['length'].min()), int(queries_df['length'].max())),
        'tables': (int(queries_df['tables'].min()), int(queries_df['tables'].max())),
        'attributes': (int(queries_df['attributes'].min()) if 'attributes' in queries_df.columns else 0, 
                      int(queries_df['attributes'].max()) if 'attributes' in queries_df.columns else 30)
    }

def normalize_slider_bounds(range_tuple):
    """Return safe slider bounds for Streamlit range sliders.

    Streamlit requires min_value < max_value. If a feature has a single
    observed value (e.g., 0..0), expand the upper bound by 1 while keeping
    the selected interval pinned to the original value.
    """
    low, high = int(range_tuple[0]), int(range_tuple[1])
    if low >= high:
        return low, low + 1, (low, low)
    return low, high, (low, high)

st.title("Demo Paper")

columns = st.columns(2)

with columns[0]:
    st.header("Widgets")
    st.write("Hover over the widgets to see the tooltips.")
    
    # Crea tre sottocodonne per i widgets
    sub_cols = st.columns(3)
    
    # Initialize shared state
    selected_databases = []
    
    # PRIMA SOTTO-COLONNA
    with sub_cols[0]:
        # Selezione Modello
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selezione Modello</h3><span class="tooltip-text">Select language models to evaluate</span></div>', unsafe_allow_html=True)
            selected_models = []
            for i, model in enumerate(models):
                css_class = get_model_class(model)
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=False, key=f"model_{i}", label_visibility="collapsed")
                with col2:
                    st.markdown(f'<p class="{css_class} text-item">{model}</p>', unsafe_allow_html=True)
                if checked:
                    selected_models.append(model)
        
        # Selezione Metrica
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selezione Metrica</h3><span class="tooltip-text">Choose evaluation metrics for query assessment</span></div>', unsafe_allow_html=True)
            selected_metrics = {}
            for metric in metrics:
                css_class = get_metric_class(metric["key"])
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=metric["default"], key=f"metric_{metric['key']}", label_visibility="collapsed")
                with col2:
                    st.markdown(f'<div class="tooltip-container"><p class="{css_class} text-item">{metric["name"]}</p><span class="tooltip-text">{metric["tooltip"]}</span></div>', unsafe_allow_html=True)
                selected_metrics[metric["key"]] = checked
        
        # Selezione Dataset
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selezione Dataset</h3><span class="tooltip-text">Select benchmark datasets for evaluation</span></div>', unsafe_allow_html=True)
            selected_datasets = []
            for i, dataset in enumerate(datasets):
                css_class = get_dataset_class(dataset)
                dataset_enabled = DATASET_FLAGS.get(dataset, True)
                dataset_label = dataset if dataset_enabled else f"{dataset} (disabled in prototype)"
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=False, key=f"dataset_{i}", label_visibility="collapsed", disabled=not dataset_enabled)
                with col2:
                    st.markdown(f'<p class="{css_class} text-item">{dataset_label}</p>', unsafe_allow_html=True)
                if checked:
                    selected_datasets.append(dataset)
            st.caption("Prototype flag: only BIRD Developer has real pairwise datapoints right now.")
        
        # Selezione Database (parte 1) - Dinamica basata sui dataset selezionati
        if len(selected_datasets) > 0 and selected_datasets[0] in databases:
            dataset_name = selected_datasets[0]
            css_class = get_dataset_class(dataset_name)
            bg_class = get_dataset_bg_class(dataset_name)
            
            with st.container(border=True):
                st.markdown(f'<div class="tooltip-container"><h3 class="{css_class}">Database - {dataset_name}</h3><span class="tooltip-text">Select specific databases from {dataset_name} dataset</span></div>', unsafe_allow_html=True)
                for i, db in enumerate(databases[dataset_name]):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox("", value=False, key=f"db_part1_{i}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="{bg_class}">{db}</div>', unsafe_allow_html=True)
                    if checked:
                        selected_databases.append(db)
    
    # SECONDA SOTTO-COLONNA
    with sub_cols[1]:
        all_four_models_selected = set(selected_models) == set(models)

        # Selezione Selettore
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selettori</h3><span class="tooltip-text">Select way to select SQL candidates between selected models</span></div>', unsafe_allow_html=True)
            selected_selectors = {}
            for selector in selectors:
                css_class = get_selector_class(selector["key"])
                selector_enabled = selector.get("enabled", True)
                if selector["key"] in ("single_selector", "embedding_selector"):
                    selector_enabled = selector_enabled and all_four_models_selected
                selector_label = selector["name"] if selector_enabled else f"{selector['name']} (disabled)"
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox(
                        "",
                        value=selector["default"],
                        key=f"selector_{selector['key']}",
                        label_visibility="collapsed",
                        disabled=not selector_enabled
                    )
                with col2:
                    st.markdown(f'<div class="tooltip-container"><p class="{css_class} text-item">{selector_label}</p><span class="tooltip-text">{selector["tooltip"]}</span></div>', unsafe_allow_html=True)
                selected_selectors[selector["key"]] = checked

            # Keep selector state consistent if user deselects one of the four candidate models.
            if not all_four_models_selected and selected_selectors.get("single_selector", False):
                selected_selectors["single_selector"] = False
                st.session_state["selector_single_selector"] = False
            if not all_four_models_selected and selected_selectors.get("embedding_selector", False):
                selected_selectors["embedding_selector"] = False
                st.session_state["selector_embedding_selector"] = False

            if not all_four_models_selected:
                st.caption("Select all 4 candidate models to enable selector modes (DeepSeek single-selector and embedding selector).")

        # Modelli da usare come giudici single-selector
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selector Models</h3><span class="tooltip-text">Choose which models act as single LLM selectors</span></div>', unsafe_allow_html=True)
            single_selector_enabled = selected_selectors.get("single_selector", False)
            css_class = get_model_class("DeepSeek Chat")
            st.session_state["selector_model_deepseek"] = single_selector_enabled
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.checkbox(
                    "",
                    key="selector_model_deepseek",
                    label_visibility="collapsed",
                    disabled=True
                )
            with col2:
                st.markdown(f'<p class="{css_class} text-item">DeepSeek Chat only</p>', unsafe_allow_html=True)

            if single_selector_enabled:
                st.warning("Prototype flag: DeepSeek selector bars are available only when all four candidate models are selected.")

        # Selezione Database (parte 2) - Dinamica basata sui dataset selezionati
        if len(selected_datasets) > 1 and selected_datasets[1] in databases:
            dataset_name = selected_datasets[1]
            css_class = get_dataset_class(dataset_name)
            bg_class = get_dataset_bg_class(dataset_name)
            
            with st.container(border=True):
                st.markdown(f'<div class="tooltip-container"><h3 class="{css_class}">Database - {dataset_name}</h3><span class="tooltip-text">Select specific databases from {dataset_name} dataset</span></div>', unsafe_allow_html=True)
                for i, db in enumerate(databases[dataset_name]):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox("", value=False, key=f"db_part2_{i}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="{bg_class}">{db}</div>', unsafe_allow_html=True)
                    if checked:
                        selected_databases.append(db)
        
        
    
    # TERZA SOTTO-COLONNA
    with sub_cols[2]:
        
        # Collect all queries from selected datasets and databases
        all_selected_queries = collect_all_selected_queries(selected_datasets, selected_databases)
        query_ranges = get_query_ranges(all_selected_queries)
        
        # Selettori Query
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selettori Query</h3><span class="tooltip-text">Filter queries by discrete SQL length, table involvement, and attribute involvement</span></div>', unsafe_allow_html=True)
            
            # Show available query info
            if len(all_selected_queries) > 0:
                st.caption(f"📚 Query disponibili: {len(all_selected_queries)} | "
                          f"Range disponibili - L:[{query_ranges['length'][0]}-{query_ranges['length'][1]}] "
                          f"T:[{query_ranges['tables'][0]}-{query_ranges['tables'][1]}] "
                          f"A:[{query_ranges['attributes'][0]}-{query_ranges['attributes'][1]}]")
            else:
                st.info("ℹ️ Seleziona dataset e database per vedere le query disponibili")

            length_min, length_max, length_default = normalize_slider_bounds(query_ranges['length'])
            tables_min, tables_max, tables_default = normalize_slider_bounds(query_ranges['tables'])
            attributes_min, attributes_max, attributes_default = normalize_slider_bounds(query_ranges['attributes'])
            
            # Dynamic sliders based on available query ranges
            length_range = st.slider(
                "Lunghezza", 
                min_value=length_min,
                max_value=length_max,
                value=length_default,
                key="length_slider",
                help="Filtra per lunghezza del testo della query"
            )
            tables_range = st.slider(
                "Tabelle Coinvolte", 
                min_value=tables_min,
                max_value=tables_max,
                value=tables_default,
                key="tables_slider",
                help="Filtra per numero di tabelle coinvolte nella query"
            )
            attributes_range = st.slider(
                "Attributi Coinvolti", 
                min_value=attributes_min,
                max_value=attributes_max,
                value=attributes_default,
                key="attributes_slider",
                help="Filtra per numero di attributi/colonne coinvolte"
            )
            
            # Filter queries based on slider values
            active_queries = filter_queries(
                all_selected_queries, 
                length_range, 
                tables_range, 
                attributes_range
            )
            
            # Display statistics
            total_available = len(all_selected_queries)
            total_active = len(active_queries)
            
            # Visual indicator with color coding
            if total_active == 0:
                st.error(f"⚠️ **Query attive:** {total_active} / {total_available}")
            elif total_active == total_available:
                st.success(f"✅ **Query attive:** {total_active} / {total_available}")
            else:
                st.info(f"🔍 **Query attive:** {total_active} / {total_available}")
            
            # Show breakdown by dataset/database using pandas groupby
            if total_active > 0:
                with st.expander("📊 Ripartizione query attive"):
                    # Use pandas groupby for efficient aggregation
                    breakdown = active_queries.groupby(['dataset', 'database']).size().reset_index(name='count')
                    
                    for _, row in breakdown.iterrows():
                        key = f"{row['dataset']} → {row['database']}"
                        st.markdown(f"- **{key}:** {row['count']} query")
            
            # Store active queries in session state for metric calculations
            st.session_state['active_queries'] = active_queries
            st.session_state['active_queries_count'] = total_active
            st.session_state['active_queries_df'] = active_queries  # active_queries is already a DataFrame
            
            # Show active queries in an expander
            if total_active > 0:
                with st.expander(f"Visualizza query attive ({total_active})"):
                    # Display as dataframe with selected columns
                    display_df = active_queries[['id', 'query', 'dataset', 'database', 'complexity', 'length', 'tables', 'attributes']].copy()
                    display_df.columns = ['ID', 'Query', 'Dataset', 'Database', 'Complessità', 'Lunghezza', 'Tabelle', 'Attributi']
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Download button for active queries
                    csv = active_queries.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Scarica query attive (CSV)",
                        data=csv,
                        file_name="active_queries.csv",
                        mime="text/csv"
                    )
            
            # Collect all active results
            active_results_df = collect_active_results(
                active_queries,
                selected_models,
                selected_metrics,
                selected_selectors_dict=selected_selectors,
            )
            
            # Store active results in session state
            st.session_state['active_results_df'] = active_results_df
            st.session_state['active_results_count'] = len(active_results_df)
            
            # Display active results stats
            if len(active_results_df) > 0:
                with st.expander(f"📊 Risultati attivi ({len(active_results_df)} datapoints)"):
                    # Show summary by model and metric
                    st.markdown("**Ripartizione risultati:**")
                    
                    # Group by model and metric
                    if not active_results_df.empty:
                        summary = active_results_df.groupby(['model', 'metric']).agg({
                            'value': ['count', 'mean', 'min', 'max']
                        }).round(2)
                        summary.columns = ['Count', 'Media (%)', 'Min (%)', 'Max (%)']
                        st.dataframe(summary, use_container_width=True)
                        
                        # Download button
                        csv_results = active_results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Scarica risultati attivi (CSV)",
                            data=csv_results,
                            file_name="active_results.csv",
                            mime="text/csv"
                        )
                        
        # Terzo dataset se presente (in colonna 2 o 3)
        if len(selected_datasets) > 2 and selected_datasets[2] in databases:
            dataset_name = selected_datasets[2]
            css_class = get_dataset_class(dataset_name)
            bg_class = get_dataset_bg_class(dataset_name)
            
            with st.container(border=True):
                st.markdown(f'<div class="tooltip-container"><h3 class="{css_class}">Database - {dataset_name}</h3><span class="tooltip-text">Select specific databases from {dataset_name} dataset</span></div>', unsafe_allow_html=True)
                for i, db in enumerate(databases[dataset_name]):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox("", value=False, key=f"db_part2_alt_{i}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="{bg_class}">{db}</div>', unsafe_allow_html=True)
                    if checked:
                        selected_databases.append(db)
        
with columns[1]:
    st.header("Barplot")

    # Display active results in a bar plot grouped by metric and colored by model
    if 'active_results_df' in st.session_state and not st.session_state['active_results_df'].empty:
        import plotly.graph_objects as go
        
        results_df = st.session_state['active_results_df'].copy()
        
        # Create metric labels that include agent info for complex metrics
        def create_metric_label(row):
            """Create a display label for the metric, including agent info if present"""
            metric_names = {
                "schema_precision": "Schema Precision",
                "schema_recall": "Schema Recall",
                "cell_value_accuracy": "Cell Value Accuracy",
                "row_set_jaccard": "Row Set Jaccard",
                "execution_accuracy": "Execution Accuracy",
                "f1_score": "F1 Score",
            }
            metric_display = metric_names.get(row['metric'], row['metric'])
            return metric_display
        
        results_df['metric_label'] = results_df.apply(create_metric_label, axis=1)
        
        # Calculate average value for each model-metric combination
        grouped_data = results_df.groupby(['metric_label', 'model'])['value'].mean().reset_index()
        
        # Create the bar chart
        fig = go.Figure()
        
        # Get unique metrics in order (preserve ordering)
        unique_metrics = grouped_data['metric_label'].unique()
        
        # Build visible sources: selected candidate models + selector sources
        visible_sources = list(selected_models)
        if selected_selectors.get('single_selector', False) and set(selected_models) == set(models):
            visible_sources.append(SELECTOR_SOURCE_LABEL)
        if selected_selectors.get('embedding_selector', False) and set(selected_models) == set(models):
            visible_sources.append(EMBEDDING_SELECTOR_SOURCE_LABEL)

        # Keep order and drop duplicates
        visible_sources = list(dict.fromkeys(visible_sources))

        # Add a trace for each source
        for source_name in visible_sources:
            model_data = grouped_data[grouped_data['model'] == source_name]

            # Selector bar uses a dedicated color and hatch pattern in prototype mode.
            is_selector_source = source_name == SELECTOR_SOURCE_LABEL

            if is_selector_source:
                if source_name == EMBEDDING_SELECTOR_SOURCE_LABEL:
                    trace_color = selectors_colors.get('embedding_selector', '#8E44AD')
                else:
                    trace_color = selectors_colors.get('single_selector', '#6A1B9A')
                pattern_shape = '/'
            else:
                trace_color = model_colors.get(source_name, '#888888')
                pattern_shape = ''
            
            # Ensure all metrics are represented (fill missing with None)
            plot_values = []
            for metric in unique_metrics:
                metric_row = model_data[model_data['metric_label'] == metric]
                if len(metric_row) > 0:
                    plot_values.append(metric_row['value'].values[0])
                else:
                    plot_values.append(None)
            
            fig.add_trace(go.Bar(
                name=source_name,
                x=unique_metrics,
                y=plot_values,
                marker=dict(
                    color=trace_color,
                    pattern=dict(shape=pattern_shape)
                ),
                text=[f"{v:.1f}%" if v is not None else "" for v in plot_values],
                textposition='outside',
                textfont=dict(size=10)
            ))
        
        # Update layout
        fig.update_layout(
            barmode='group',
            xaxis_title="Metrics",
            yaxis_title="Accuracy (%)",
            yaxis=dict(
                range=[0, 100],
                dtick=10
            ),
            legend=dict(
                title="Sources",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            height=600,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        # Add grid lines
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_xaxes(showgrid=False)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        with st.expander("📊 Summary Statistics"):
            summary_pivot = grouped_data.pivot(index='metric_label', columns='model', values='value').round(2)
            st.dataframe(summary_pivot, use_container_width=True)
    else:
        st.info("👈 Select models, metrics, datasets, and databases from the left panel to see the visualization.")
        st.caption("The bar chart will display model performance grouped by metrics with values ranging from 0% to 100%.")
import streamlit as st
import json
import os
import pandas as pd
import hashlib
import re
import random
import numpy as np

from common_utils import (
    CANONICAL_METRICS,
    collect_models_from_metric_files,
    metric_to_percentage,
    readable_model_label,
)
from embedding import (
    compute_similarity_groups_pairwise,
    get_vector_closest_to_centroid,
    load_embeddings_artifact,
    load_json_artifact,
    load_similarity_matrix_artifact,
)

_SYSTEM_MODELS = collect_models_from_metric_files('metrics_results') or [
    'deepseek-chat',
    'qwen2.5-coder_32b',
    'qwen3-coder_30b',
    'cogito_70b',
    'codellama_70b',
    'codestral_22b',
    'sqlcoder_15b',
]

models = [readable_model_label(m) for m in _SYSTEM_MODELS]
datasets = [
    "BIRD Training",
    "BIRD Developer",
    "SPIDER Training",
    "SPIDER Dev",
    "SPIDER Test",
]

MODEL_TO_SYSTEM_ID = {readable_model_label(system_id): system_id for system_id in _SYSTEM_MODELS}
SYSTEM_ID_TO_MODEL = {v: k for k, v in MODEL_TO_SYSTEM_ID.items()}

PAIRWISE_METRICS = list(CANONICAL_METRICS)

EMBEDDING_SELECTOR_SOURCE_LABEL = "Embedding selector"
FAKE_DATA_DIR = 'fake_data'
CACHE_DIR = 'cache_results'
CACHE_SCHEMA_VERSION = '2026-03-22-v1'

DATASET_FLAGS = {
    "BIRD Training": True,
    "BIRD Developer": True,
    "SPIDER Training": True,
    "SPIDER Dev": True,
    "SPIDER Test": True,
}

SELECTOR_MODEL_OPTIONS = [
    {
        "key": system_id,
        "name": readable_model_label(system_id),
        "tooltip": f"Use {readable_model_label(system_id)} as pairwise-selector judge model.",
    }
    for system_id in _SYSTEM_MODELS
]


def selector_source_label(selector_model_system_id):
    model_name = SYSTEM_ID_TO_MODEL.get(selector_model_system_id, selector_model_system_id)
    return f"{model_name} selector"


def selector_source_to_model_label(source_name):
    suffix = " selector"
    if isinstance(source_name, str) and source_name.endswith(suffix):
        return source_name[:-len(suffix)]
    return source_name


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _build_file_signature(file_paths, extra_tag=""):
    chunks = [CACHE_SCHEMA_VERSION, str(extra_tag)]
    for file_path in file_paths:
        abs_path = os.path.abspath(file_path)
        if os.path.exists(abs_path):
            stat = os.stat(abs_path)
            chunks.append(f"{abs_path}|{stat.st_mtime_ns}|{stat.st_size}")
        else:
            chunks.append(f"{abs_path}|missing")
    raw = "||".join(chunks)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

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
def load_dataset_sql_stats(dataset_name, dataset_path, tables_path):
    """Precompute table/attribute/length stats from SQL text for a dataset split."""
    if not os.path.exists(dataset_path) or not os.path.exists(tables_path):
        return {}

    _ensure_cache_dir()
    cache_id = hashlib.sha256(f"{dataset_name}|{dataset_path}|{tables_path}".encode('utf-8')).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, f"sql_stats_{cache_id}.json")
    cache_signature = _build_file_signature([dataset_path, tables_path], extra_tag=f"sql_stats|{dataset_name}")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if cached.get('signature') == cache_signature and isinstance(cached.get('stats'), dict):
                return cached['stats']
        except Exception:
            pass

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_payload = json.load(f)
    with open(tables_path, 'r', encoding='utf-8') as f:
        dataset_tables_payload = json.load(f)

    schema_lookup = _build_schema_lookup(dataset_tables_payload)
    stats_by_gid = {}
    pending_rows = []
    raw_lengths = []

    for idx, row in enumerate(dataset_payload):
        if not isinstance(row, dict):
            continue

        db_id = row.get('db_id')
        row_id = row.get('question_id')
        if row_id is None:
            row_id = row.get('id')
        if row_id is None:
            row_id = idx

        sql_text = row.get('ground_truth') or row.get('SQL') or row.get('query') or ""

        schema_info = schema_lookup.get(db_id, {})
        table_set = schema_info.get('tables', set())
        column_set = schema_info.get('columns', set())

        tokens = _extract_sql_identifier_candidates(str(sql_text))
        table_count = len(tokens.intersection(table_set)) if table_set else 0
        attribute_count = len(tokens.intersection(column_set)) if column_set else 0
        sql_length = len(re.findall(r'\S+', str(sql_text)))

        gid = string_to_deterministic_int(f"{dataset_name}|{db_id}|{row_id}")
        pending_rows.append((gid, table_count, attribute_count, sql_length))
        raw_lengths.append(sql_length)

    discrete_lengths = _compute_sql_length_bucket(raw_lengths, num_buckets=4)
    for idx, (gid, table_count, attribute_count, _) in enumerate(pending_rows):
        stats_by_gid[gid] = {
            'tables': int(table_count),
            'attributes': int(attribute_count),
            'length': int(discrete_lengths[idx]) if idx < len(discrete_lengths) else 0,
        }

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'signature': cache_signature, 'stats': stats_by_gid}, f)
    except Exception:
        pass

    return stats_by_gid

# Load query datasets
@st.cache_data
def load_queries():
    """Load all supported query datasets from datasets_files and normalize columns."""
    queries_list = []
    dataset_sources = [
        ('BIRD Training', 'datasets_files/BIRD/train.json', 'datasets_files/BIRD/train_tables.json'),
        ('BIRD Developer', 'datasets_files/BIRD/dev.json', 'datasets_files/BIRD/dev_tables.json'),
        ('SPIDER Training', 'datasets_files/SPIDER/train_spider.json', 'datasets_files/SPIDER/tables.json'),
        ('SPIDER Training', 'datasets_files/SPIDER/train_others.json', 'datasets_files/SPIDER/tables.json'),
        ('SPIDER Dev', 'datasets_files/SPIDER/dev.json', 'datasets_files/SPIDER/tables.json'),
        ('SPIDER Test', 'datasets_files/SPIDER/test.json', 'datasets_files/SPIDER/test_tables.json'),
    ]

    def _normalize_queries(raw_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=['id', 'query', 'database', 'length', 'tables', 'attributes', 'dataset'])

        df = raw_df.copy()

        if 'question_id' in df.columns:
            ids = pd.to_numeric(df['question_id'], errors='coerce')
            if ids.isna().any():
                ids = pd.Series(range(len(df)), index=df.index)
        elif 'id' in df.columns:
            ids = pd.to_numeric(df['id'], errors='coerce')
            if ids.isna().any():
                ids = pd.Series(range(len(df)), index=df.index)
        else:
            ids = pd.Series(range(len(df)), index=df.index)

        query_col = ''
        for col in ('question', 'query'):
            if col in df.columns:
                query_col = col
                break

        db_col = ''
        for col in ('db_id', 'database'):
            if col in df.columns:
                db_col = col
                break

        normalized = pd.DataFrame({
            'id': ids.astype(int),
            'query': df[query_col].astype(str) if query_col else '',
            'database': df[db_col].astype(str) if db_col else '',
            'length': pd.to_numeric(df.get('length', pd.Series([0] * len(df), index=df.index)), errors='coerce').fillna(0).astype(int),
            'tables': pd.to_numeric(df.get('tables', pd.Series([0] * len(df), index=df.index)), errors='coerce').fillna(0).astype(int),
            'attributes': pd.to_numeric(df.get('attributes', pd.Series([0] * len(df), index=df.index)), errors='coerce').fillna(0).astype(int),
        })
        normalized['dataset'] = dataset_name
        return normalized
    
    try:
        _ensure_cache_dir()
        signature_paths = []
        for _, dataset_path, tables_path in dataset_sources:
            signature_paths.append(dataset_path)
            signature_paths.append(tables_path)
        queries_signature = _build_file_signature(signature_paths, extra_tag='queries_dataframe')

        queries_cache_meta_path = os.path.join(CACHE_DIR, 'queries_cache_meta.json')
        queries_cache_data_path = os.path.join(CACHE_DIR, 'queries_cache.pkl')

        if os.path.exists(queries_cache_meta_path) and os.path.exists(queries_cache_data_path):
            try:
                with open(queries_cache_meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if meta.get('signature') == queries_signature:
                    cached_df = pd.read_pickle(queries_cache_data_path)
                    if isinstance(cached_df, pd.DataFrame):
                        return cached_df
            except Exception:
                pass

        for dataset_name, dataset_path, tables_path in dataset_sources:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_df = pd.DataFrame(json.load(f))

            normalized_df = _normalize_queries(raw_df, dataset_name)
            split_stats = load_dataset_sql_stats(dataset_name, dataset_path, tables_path)

            normalized_df['g_id'] = (
                normalized_df['dataset'].astype(str) + '|' +
                normalized_df['database'].astype(str) + '|' +
                normalized_df['id'].astype(str)
            ).apply(string_to_deterministic_int)

            normalized_df['tables'] = normalized_df['g_id'].map(
                lambda gid: split_stats.get(gid, {}).get('tables', 0)
            ).astype(int)
            normalized_df['attributes'] = normalized_df['g_id'].map(
                lambda gid: split_stats.get(gid, {}).get('attributes', 0)
            ).astype(int)
            normalized_df['length'] = normalized_df['g_id'].map(
                lambda gid: split_stats.get(gid, {}).get('length', 0)
            ).astype(int)

            queries_list.append(normalized_df)
        
        # Concatenate all dataframes at once for efficiency
        queries_data = pd.concat(queries_list, ignore_index=True) if queries_list else pd.DataFrame()

        try:
            queries_data.to_pickle(queries_cache_data_path)
            with open(queries_cache_meta_path, 'w', encoding='utf-8') as f:
                json.dump({'signature': queries_signature}, f)
        except Exception:
            pass

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
def load_bird_metrics_lookup():
    """Load BIRD Developer true metrics indexed by (db_id, question_id, model_system_id)."""
    lookup = {}
    metrics_dir = 'metrics_results'
    if not os.path.exists(metrics_dir):
        return lookup

    for filename in os.listdir(metrics_dir):
        if not filename.startswith('evaluation_sql_metrics_') or not filename.endswith('_vs_ground_truth.json'):
            continue

        system_model = filename[len('evaluation_sql_metrics_'):-len('_vs_ground_truth.json')]
        path = os.path.join(metrics_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, list):
            continue

        for row in payload:
            if not isinstance(row, dict):
                continue
            db_id = row.get('db_id')
            question_id = row.get('question_id')
            if db_id is None or question_id is None:
                continue

            key = (str(db_id), int(question_id), system_model)
            metric_block = {}
            for metric_key in PAIRWISE_METRICS:
                value = row.get(metric_key)
                pct = metric_to_percentage(value)
                if pct is not None:
                    metric_block[metric_key] = pct
            if metric_block:
                lookup[key] = metric_block

    return lookup


@st.cache_data
def load_precomputed_embedding_assets():
    """Load precomputed similarity index and embeddings vector table."""
    index_path = os.path.join('precomputed', 'similarity', 'bird_dev_similarity_index.json')
    emb_path = os.path.join('precomputed', 'embeddings', 'sql_embeddings.npz')

    if not os.path.exists(index_path) or not os.path.exists(emb_path):
        return {}, None

    try:
        index_payload = load_json_artifact(index_path)
        emb_payload = load_embeddings_artifact(emb_path)
        return index_payload, emb_payload.get('vectors')
    except Exception:
        return {}, None


def _pick_selected_model_from_leaderboard(row, selector_model, dataset_name):
    leaderboard = row.get('leaderboard') or []
    if isinstance(leaderboard, list) and leaderboard:
        top_score = max(x.get('wins', 0) for x in leaderboard if isinstance(x, dict))
        top_models = [x.get('model') for x in leaderboard if isinstance(x, dict) and x.get('wins', 0) == top_score and x.get('model')]
        if top_models:
            seed = f"{selector_model}|{dataset_name}|{row.get('db_id')}|{row.get('question_id')}"
            return random.Random(seed).choice(sorted(top_models))
    return row.get('selected_candidate_model')


@st.cache_data
def load_model_results():
    """Load canonical metrics for candidate models (real BIRD + fake non-BIRD)."""
    all_rows = []
    seen = set()

    try:
        # Real BIRD developer metrics from metrics_results/evaluation_sql_metrics_* files.
        metrics_lookup = load_bird_metrics_lookup()
        for (database, query_id, system_model), metric_block in metrics_lookup.items():
            model_name = SYSTEM_ID_TO_MODEL.get(system_model, readable_model_label(system_model))
            for metric_key in PAIRWISE_METRICS:
                metric_value = metric_block.get(metric_key)
                if not isinstance(metric_value, (int, float)):
                    continue
                dedup_key = ('BIRD Developer', query_id, database, model_name, metric_key)
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

        fake_execution_path = os.path.join(FAKE_DATA_DIR, 'fake_execution_metrics.json')
        if os.path.exists(fake_execution_path):
            with open(fake_execution_path, 'r', encoding='utf-8') as f:
                fake_rows = json.load(f)

            if isinstance(fake_rows, list):
                for row in fake_rows:
                    if not isinstance(row, dict):
                        continue

                    dataset_name = row.get('dataset')
                    query_id = row.get('question_id')
                    database = row.get('db_id')
                    system_model = row.get('model')
                    metrics = row.get('metrics') or {}

                    model_name = SYSTEM_ID_TO_MODEL.get(system_model, readable_model_label(system_model))

                    for metric_key in PAIRWISE_METRICS:
                        metric_value = metrics.get(metric_key)

                        metric_pct = metric_to_percentage(metric_value)
                        if metric_pct is None:
                            continue

                        dedup_key = (dataset_name, query_id, database, model_name, metric_key)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        all_rows.append({
                            'dataset': dataset_name,
                            'database': database,
                            'id': query_id,
                            'model': model_name,
                            'metric': metric_key,
                            'value': metric_pct,
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
    """Load selector-derived metrics from pairwise outcomes plus fake fallback."""
    rows = []
    seen = set()
    bird_lookup = load_bird_metrics_lookup()

    pairwise_dir = 'pairwise_results'
    if os.path.exists(pairwise_dir):
        for filename in os.listdir(pairwise_dir):
            if not filename.endswith('_pairwise_selector_results.json'):
                continue
            path = os.path.join(pairwise_dir, filename)
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            selector_model = payload.get('selector_model', 'deepseek-chat')
            source_label = selector_source_label(selector_model)

            for row in payload.get('results', []):
                if not isinstance(row, dict):
                    continue
                query_id = row.get('question_id')
                database = row.get('db_id')
                selected_model = _pick_selected_model_from_leaderboard(row, selector_model, 'BIRD Developer')
                if not selected_model:
                    continue

                metric_block = bird_lookup.get((str(database), int(query_id), selected_model), {})
                for metric_key in PAIRWISE_METRICS:
                    metric_value = metric_block.get(metric_key)
                    if not isinstance(metric_value, (int, float)):
                        continue
                    dedup_key = ('BIRD Developer', query_id, database, source_label, metric_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    rows.append({
                        'dataset': 'BIRD Developer',
                        'database': database,
                        'id': query_id,
                        'model': source_label,
                        'metric': metric_key,
                        'value': metric_value,
                    })

    fake_pairwise_selector_path = os.path.join(FAKE_DATA_DIR, 'fake_selector_pairwise_results.json')
    if os.path.exists(fake_pairwise_selector_path):
        with open(fake_pairwise_selector_path, 'r', encoding='utf-8') as f:
            fake_pairwise_payload = json.load(f)

        if isinstance(fake_pairwise_payload, list):
            for row in fake_pairwise_payload:
                if not isinstance(row, dict):
                    continue

                dataset_name = row.get('dataset')
                query_id = row.get('question_id')
                database = row.get('db_id')
                selector_model = row.get('judge_model', 'deepseek-chat')
                source_label = selector_source_label(selector_model)
                candidate_models = row.get('candidate_models') or []
                judgments = row.get('pairwise_judgments') or []
                if not candidate_models or not judgments:
                    continue

                wins = {m: 0 for m in candidate_models}
                for judgment in judgments:
                    winner = judgment.get('winner')
                    if winner in wins:
                        wins[winner] += 1

                top_score = max(wins.values()) if wins else 0
                top_models = [m for m, v in wins.items() if v == top_score]
                if not top_models:
                    continue

                seed = f"{selector_model}|{dataset_name}|{database}|{query_id}"
                selected_model = random.Random(seed).choice(sorted(top_models))

                selected_metrics = None
                for judgment in judgments:
                    if judgment.get('model_a') == selected_model:
                        selected_metrics = judgment.get('metrics_a') or {}
                        break
                    if judgment.get('model_b') == selected_model:
                        selected_metrics = judgment.get('metrics_b') or {}
                        break

                if not isinstance(selected_metrics, dict):
                    continue

                for metric_key in PAIRWISE_METRICS:
                    metric_pct = metric_to_percentage(selected_metrics.get(metric_key))
                    if metric_pct is None:
                        continue

                    dedup_key = (dataset_name, query_id, database, source_label, metric_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    rows.append({
                        'dataset': dataset_name,
                        'database': database,
                        'id': query_id,
                        'model': source_label,
                        'metric': metric_key,
                        'value': metric_pct,
                    })

    return pd.DataFrame(rows)


@st.cache_data
def load_embedding_selector_results():
    """Load persisted embedding selector metrics (fallback path)."""
    file_path = 'embedding_pipeline_selection_results.json'

    rows = []
    seen = set()

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue

                query_id = row.get('question_id')
                database = row.get('db_id')
                metrics = row.get('ground_truth_comparison_metrics') or {}

                for metric_key in PAIRWISE_METRICS:
                    metric_value = metrics.get(metric_key)
                    metric_pct = metric_to_percentage(metric_value)
                    if metric_pct is None:
                        continue

                    dedup_key = ('BIRD Developer', query_id, database, metric_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    rows.append({
                        'dataset': 'BIRD Developer',
                        'database': database,
                        'id': query_id,
                        'model': EMBEDDING_SELECTOR_SOURCE_LABEL,
                        'metric': metric_key,
                        'value': metric_pct,
                    })

    fake_embedding_path = os.path.join(FAKE_DATA_DIR, 'fake_embedding_selection.json')
    if os.path.exists(fake_embedding_path):
        with open(fake_embedding_path, 'r', encoding='utf-8') as f:
            fake_payload = json.load(f)

        if isinstance(fake_payload, list):
            for row in fake_payload:
                if not isinstance(row, dict):
                    continue

                dataset_name = row.get('dataset')
                query_id = row.get('question_id')
                database = row.get('db_id')
                metrics = row.get('ground_truth_comparison_metrics') or {}

                for metric_key in PAIRWISE_METRICS:
                    metric_value = metrics.get(metric_key)
                    metric_pct = metric_to_percentage(metric_value)
                    if metric_pct is None:
                        continue

                    dedup_key = (dataset_name, query_id, database, metric_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    rows.append({
                        'dataset': dataset_name,
                        'database': database,
                        'id': query_id,
                        'model': EMBEDDING_SELECTOR_SOURCE_LABEL,
                        'metric': metric_key,
                        'value': metric_pct,
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


def compute_realtime_embedding_selector_rows(active_queries, selected_models, selected_metric_keys):
    """Compute embedding selector rows in real time from precomputed similarity matrices."""
    if active_queries.empty:
        return []

    precomputed_index, embeddings_vectors = load_precomputed_embedding_assets()
    if not precomputed_index or embeddings_vectors is None:
        return []

    queries_index = precomputed_index.get('queries', {}) if isinstance(precomputed_index, dict) else {}
    bird_lookup = load_bird_metrics_lookup()
    selected_systems = {MODEL_TO_SYSTEM_ID.get(name) for name in selected_models if MODEL_TO_SYSTEM_ID.get(name)}

    rows = []
    for _, qrow in active_queries.iterrows():
        dataset_name = qrow.get('dataset')
        if dataset_name != 'BIRD Developer':
            continue

        db_id = str(qrow.get('database', ''))
        question_id = int(qrow.get('id'))
        query_key = f"{db_id}|{question_id}"
        entry = queries_index.get(query_key)
        if not isinstance(entry, dict):
            continue

        models_for_query = entry.get('models', [])
        sql_ids_for_query = entry.get('sql_ids', [])
        matrix_file = entry.get('matrix_file')
        if not isinstance(models_for_query, list) or not isinstance(sql_ids_for_query, list) or not matrix_file:
            continue

        selected_positions = [
            idx for idx, system_model in enumerate(models_for_query)
            if system_model in selected_systems
        ]
        if len(selected_positions) < 2:
            continue

        matrix_path = os.path.join('.', matrix_file)
        if not os.path.exists(matrix_path):
            continue

        try:
            full_matrix = load_similarity_matrix_artifact(matrix_path)
        except Exception:
            continue

        sub_matrix = full_matrix[np.ix_(selected_positions, selected_positions)]
        sub_sql_ids = [int(sql_ids_for_query[i]) for i in selected_positions]
        sub_vectors = [embeddings_vectors[sql_id] for sql_id in sub_sql_ids]

        if len(sub_vectors) < 2:
            continue

        threshold = float(sub_matrix[np.triu_indices_from(sub_matrix, k=1)].mean()) if sub_matrix.shape[0] > 1 else 1.0
        groups = compute_similarity_groups_pairwise(sub_vectors, sub_matrix, verbose=0, threshold=threshold)
        biggest_group = max(groups, key=len)
        selected_local_idx = get_vector_closest_to_centroid(biggest_group)
        if selected_local_idx is None:
            continue

        selected_system = models_for_query[selected_positions[selected_local_idx]]
        metric_block = bird_lookup.get((db_id, question_id, selected_system), {})

        for metric_key in selected_metric_keys:
            metric_value = metric_block.get(metric_key)
            if not isinstance(metric_value, (int, float)):
                continue
            rows.append(
                {
                    'dataset': dataset_name,
                    'database': db_id,
                    'id': question_id,
                    'model': EMBEDDING_SELECTOR_SOURCE_LABEL,
                    'metric': metric_key,
                    'value': metric_value,
                }
            )

    return rows

def collect_active_results(active_queries, selected_models, selected_metrics_dict, selected_selectors_dict=None, selected_selector_models=None):
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
    if selected_selector_models is None:
        selected_selector_models = []

    selector_rows = []
    selector_enabled = selected_selectors_dict.get('single_selector', False)
    embedding_selector_enabled = selected_selectors_dict.get('embedding_selector', False)
    selected_selector_sources = [selector_source_label(s) for s in selected_selector_models]

    if selector_enabled and selected_metric_keys and selected_selector_sources and not all_selector_results.empty:
        selector_df = all_selector_results[
            all_selector_results['g_id'].isin(active_g_ids) &
            all_selector_results['model'].isin(selected_selector_sources) &
            all_selector_results['metric'].isin(selected_metric_keys)
        ].copy()
        if not selector_df.empty:
            selector_rows = selector_df.to_dict(orient='records')

    if embedding_selector_enabled and selected_metric_keys:
        realtime_rows = compute_realtime_embedding_selector_rows(active_queries, selected_models, selected_metric_keys)
        if realtime_rows:
            selector_rows.extend(realtime_rows)
        elif not all_embedding_selector_results.empty:
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
    {"name": "Execution Accuracy", "key": "execution_accuracy", "default": True, "tooltip": "Execution-level correctness compared with ground truth."},
    {"name": "Exact Match", "key": "exact_match", "default": True, "tooltip": "Exact SQL string/structure match against ground truth."},
    {"name": "SQL F1", "key": "sql_f1_score", "default": True, "tooltip": "SQL-level F1 score from precision/recall matching."},
    {"name": "Response F1", "key": "response_schema_f1_score", "default": True, "tooltip": "F1 score on response schema alignment."},
    {"name": "Cell F1", "key": "cell_f1_score", "default": True, "tooltip": "F1 score over cell-level result values."},
]

selectors = [
    {"name": "Pairwise LLM Selector", "key": "single_selector", "default": False, "enabled": True, "tooltip": "Plots pairwise-selector metrics for the selected judge models."},
    {"name": "Embedding Selector", "key": "embedding_selector", "default": False, "enabled": True, "tooltip": "Plots embedding selector metrics from true or generated data."}
]

# Color definitions - organized by color families
model_colors = {
    "DeepSeek Chat": "#246BCE",
    "Qwen2.5 Coder 32B": "#1C9A79",
    "Qwen3 Coder 30B": "#F39C12",
    "Cogito 70B": "#C0392B",
    "CodeLlama 70B": "#8D6E63",
    "Codestral 22B": "#5D6D7E",
    "SQLCoder 15B": "#16A085",
}

dataset_colors = {
    "BIRD Training": "#E85D04",  # Dark orange
    "BIRD Developer": "#FF9500",  # Bright orange
    "SPIDER Training": "#FFBB66",  # Light orange
    "SPIDER Dev": "#FFC98A",  # Pale orange
    "SPIDER Test": "#FFD9AD",  # Soft orange
}

metric_colors = {
    "execution_accuracy": "#D1495B",
    "exact_match": "#0F4C81",
    "sql_f1_score": "#6C5CE7",
    "response_schema_f1_score": "#2E86AB",
    "cell_f1_score": "#00A878",
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
.color-spider-training { color: #FFBB66 !important; }
.color-spider-dev { color: #FFC98A !important; }
.color-spider-test { color: #FFD9AD !important; }

/* Metric colors - Green family */
.color-exact-match { color: #0F4C81 !important; }
.color-response-f1 { color: #2E86AB !important; }
.color-cell-f1 { color: #00A878 !important; }
.color-sql-f1 { color: #6C5CE7 !important; }
.color-execution-accuracy { color: #D1495B !important; }
            
/* Selector colors Grey/Purple family */
.color-single-selector { color: #6A1B9A !important; }
.color-embedding-selector { color: #8E44AD !important; }

/* Background colors for conditional boxes */
.bg-bird-training { background-color: rgba(232, 93, 4, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-bird-developer { background-color: rgba(255, 149, 0, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-spider-training { background-color: rgba(255, 187, 102, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-spider-dev { background-color: rgba(255, 201, 138, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-spider-test { background-color: rgba(255, 217, 173, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-tdex { background-color: rgba(106, 27, 154, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-embedding-selector { background-color: rgba(142, 68, 173, 0.15) !important; padding: 10px; border-radius: 5px; }
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
        "Cogito 70B": "color-cogito",
    }
    return class_map.get(model_name, "")

def get_dataset_class(dataset_name):
    class_map = {
        "BIRD Training": "color-bird-training",
        "BIRD Developer": "color-bird-developer",
        "SPIDER Training": "color-spider-training",
        "SPIDER Dev": "color-spider-dev",
        "SPIDER Test": "color-spider-test",
    }
    return class_map.get(dataset_name, "")

def get_dataset_bg_class(dataset_name):
    class_map = {
        "BIRD Training": "bg-bird-training",
        "BIRD Developer": "bg-bird-developer",
        "SPIDER Training": "bg-spider-training",
        "SPIDER Dev": "bg-spider-dev",
        "SPIDER Test": "bg-spider-test",
    }
    return class_map.get(dataset_name, "")

def get_metric_class(metric_key):
    class_map = {
        "execution_accuracy": "color-execution-accuracy",
        "exact_match": "color-exact-match",
        "sql_f1_score": "color-sql-f1",
        "response_schema_f1_score": "color-response-f1",
        "cell_f1_score": "color-cell-f1",
    }
    return class_map.get(metric_key, "")

def get_selector_class(selector_key):
    class_map = {
        "single_selector": "color-single-selector",
        "embedding_selector": "color-embedding-selector"
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
    
    # Crea quattro sottocolonne: la terza ospita i widget Database.
    sub_cols = st.columns(4)
    
    # Initialize shared state
    selected_databases = []
    selected_selector_models = []
    
    # PRIMA SOTTO-COLONNA
    with sub_cols[0]:
        # Model Selection
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Model Selection</h3><span class="tooltip-text">Select candidate models to display in the chart</span></div>', unsafe_allow_html=True)
            selected_models = []
            for i, model in enumerate(models):
                css_class = get_model_class(model)
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=False, key=f"model_{i}", label_visibility="collapsed")
                with col2:
                    st.markdown(f'<div class="tooltip-container"><p class="{css_class} text-item">{model}</p><span class="tooltip-text">Include {model} as a candidate-model series.</span></div>', unsafe_allow_html=True)
                if checked:
                    selected_models.append(model)
        
        # Metric Selection
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Metric Selection</h3><span class="tooltip-text">Choose evaluation metrics for query assessment</span></div>', unsafe_allow_html=True)
            selected_metrics = {}
            for metric in metrics:
                css_class = get_metric_class(metric["key"])
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=metric["default"], key=f"metric_{metric['key']}", label_visibility="collapsed")
                with col2:
                    st.markdown(f'<div class="tooltip-container"><p class="{css_class} text-item">{metric["name"]}</p><span class="tooltip-text">{metric["tooltip"]}</span></div>', unsafe_allow_html=True)
                selected_metrics[metric["key"]] = checked
        
        # Dataset Selection
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Dataset Selection</h3><span class="tooltip-text">Select benchmark datasets for evaluation</span></div>', unsafe_allow_html=True)
            selected_datasets = []
            for i, dataset in enumerate(datasets):
                css_class = get_dataset_class(dataset)
                dataset_enabled = DATASET_FLAGS.get(dataset, True)
                dataset_label = dataset if dataset_enabled else f"{dataset} (disabled)"
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=False, key=f"dataset_{i}", label_visibility="collapsed", disabled=not dataset_enabled)
                with col2:
                    st.markdown(f'<p class="{css_class} text-item">{dataset_label}</p>', unsafe_allow_html=True)
                if checked:
                    selected_datasets.append(dataset)
            st.caption("Data can come from real or generated sources, depending on availability.")

    # Database
    with sub_cols[1]:
        # Database Selection: one fixed-height scrollable widget per dataset.
        if not selected_datasets:
            with st.container(border=True):
                st.markdown('<h3>Database Selection</h3>', unsafe_allow_html=True)
                st.info("Select at least one dataset to view available databases")
        else:
            for dataset_idx, dataset_name in enumerate(selected_datasets):
                if dataset_name not in databases:
                    continue

                css_class = get_dataset_class(dataset_name)
                bg_class = get_dataset_bg_class(dataset_name)
                dataset_dbs = databases[dataset_name]

                with st.container(border=True, height=250):
                    st.markdown(
                        f'<div class="tooltip-container"><h3 class="{css_class}">Database - {dataset_name}</h3><span class="tooltip-text">Select specific databases from {dataset_name} dataset</span></div>',
                        unsafe_allow_html=True
                    )
                    for db_idx, db in enumerate(dataset_dbs):
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            checked = st.checkbox(
                                "",
                                value=False,
                                key=f"db_scroll_{dataset_idx}_{db_idx}",
                                label_visibility="collapsed"
                            )
                        with col2:
                            st.markdown(f'<div class="{bg_class}">{db}</div>', unsafe_allow_html=True)
                        if checked:
                            selected_databases.append(db)

    # Third sub-column
    with sub_cols[2]:
        # Selector Mode
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selector Modes</h3><span class="tooltip-text">Select which selector pipelines to include in the chart</span></div>', unsafe_allow_html=True)
            selected_selectors = {}
            for selector in selectors:
                css_class = get_selector_class(selector["key"])
                selector_enabled = selector.get("enabled", True)
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

        # Single-selector model choices
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Selector Models</h3><span class="tooltip-text">Choose which models act as single-selector sources</span></div>', unsafe_allow_html=True)
            single_selector_enabled = selected_selectors.get("single_selector", False)
            selected_selector_models = []
            for option in SELECTOR_MODEL_OPTIONS:
                key = f"selector_model_{option['key'].replace('.', '_').replace('-', '_')}"
                css_class = get_model_class(option["name"])
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox(
                        "",
                        value=single_selector_enabled,
                        key=key,
                        label_visibility="collapsed",
                        disabled=not single_selector_enabled
                    )
                with col2:
                    st.markdown(
                        f'<div class="tooltip-container"><p class="{css_class} text-item">{option["name"]}</p><span class="tooltip-text">{option["tooltip"]}</span></div>',
                        unsafe_allow_html=True
                    )
                if checked:
                    selected_selector_models.append(option["key"])

    # Fourth sub-column
    with sub_cols[3]:
        
        # Collect all queries from selected datasets and databases
        all_selected_queries = collect_all_selected_queries(selected_datasets, selected_databases)
        query_ranges = get_query_ranges(all_selected_queries)
        
        # Query Filters
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Query Filters</h3><span class="tooltip-text">Filter queries by discrete SQL length, table involvement, and attribute involvement</span></div>', unsafe_allow_html=True)
            
            # Show available query info
            if len(all_selected_queries) > 0:
                st.caption(f"Available queries: {len(all_selected_queries)} | "
                          f"Ranges - L:[{query_ranges['length'][0]}-{query_ranges['length'][1]}] "
                          f"T:[{query_ranges['tables'][0]}-{query_ranges['tables'][1]}] "
                          f"A:[{query_ranges['attributes'][0]}-{query_ranges['attributes'][1]}]")
            else:
                st.info("Select datasets and databases to view available queries")

            length_min, length_max, length_default = normalize_slider_bounds(query_ranges['length'])
            tables_min, tables_max, tables_default = normalize_slider_bounds(query_ranges['tables'])
            attributes_min, attributes_max, attributes_default = normalize_slider_bounds(query_ranges['attributes'])
            
            # Dynamic sliders based on available query ranges
            length_range = st.slider(
                "Length", 
                min_value=length_min,
                max_value=length_max,
                value=length_default,
                key="length_slider",
                help="Filter by SQL/text length bucket"
            )
            tables_range = st.slider(
                "Tables Involved", 
                min_value=tables_min,
                max_value=tables_max,
                value=tables_default,
                key="tables_slider",
                help="Filter by number of tables involved in the query"
            )
            attributes_range = st.slider(
                "Attributes Involved", 
                min_value=attributes_min,
                max_value=attributes_max,
                value=attributes_default,
                key="attributes_slider",
                help="Filter by number of involved attributes/columns"
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
                st.error(f"Active queries: {total_active} / {total_available}")
            elif total_active == total_available:
                st.success(f"Active queries: {total_active} / {total_available}")
            else:
                st.info(f"Active queries: {total_active} / {total_available}")
            
            # Show breakdown by dataset/database using pandas groupby
            if total_active > 0:
                with st.expander("Active Query Breakdown"):
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
                with st.expander(f"View active queries ({total_active})"):
                    # Display as dataframe with selected columns
                    display_df = active_queries[['id', 'query', 'dataset', 'database', 'length', 'tables', 'attributes']].copy()
                    display_df.columns = ['ID', 'Query', 'Dataset', 'Database', 'Length', 'Tables', 'Attributes']
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Download button for active queries
                    csv = active_queries.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download active queries (CSV)",
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
                selected_selector_models=selected_selector_models,
            )
            
            # Store active results in session state
            st.session_state['active_results_df'] = active_results_df
            st.session_state['active_results_count'] = len(active_results_df)
            
            # Display active results stats
            if len(active_results_df) > 0:
                with st.expander(f"Active results ({len(active_results_df)} datapoints)"):
                    # Show summary by model and metric
                    st.markdown("**Results breakdown:**")
                    
                    # Group by model and metric
                    if not active_results_df.empty:
                        summary = active_results_df.groupby(['model', 'metric']).agg({
                            'value': ['count', 'mean', 'min', 'max']
                        }).round(2)
                        summary.columns = ['Count', 'Mean (%)', 'Min (%)', 'Max (%)']
                        st.dataframe(summary, use_container_width=True)
                        
                        # Download button
                        csv_results = active_results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download active results (CSV)",
                            data=csv_results,
                            file_name="active_results.csv",
                            mime="text/csv"
                        )
                        

with columns[1]:
    st.header("Bar Plot")

    # Display active results in a bar plot grouped by metric and colored by model
    if 'active_results_df' in st.session_state and not st.session_state['active_results_df'].empty:
        import plotly.graph_objects as go
        
        results_df = st.session_state['active_results_df'].copy()
        
        # Create metric labels that include agent info for complex metrics
        def create_metric_label(row):
            """Create a display label for the metric, including agent info if present"""
            metric_names = {
                "execution_accuracy": "Execution Accuracy",
                "exact_match": "Exact Match",
                "sql_f1_score": "SQL F1",
                "response_schema_f1_score": "Response F1",
                "cell_f1_score": "Cell F1",
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
        if selected_selectors.get('single_selector', False):
            visible_sources.extend([selector_source_label(s) for s in selected_selector_models])
        if selected_selectors.get('embedding_selector', False):
            visible_sources.append(EMBEDDING_SELECTOR_SOURCE_LABEL)

        # Keep order and drop duplicates
        visible_sources = list(dict.fromkeys(visible_sources))

        # Add a trace for each source
        for source_name in visible_sources:
            model_data = grouped_data[grouped_data['model'] == source_name]

            # Selector bars inherit their underlying model color and use hatch patterns.
            is_selector_source = source_name.endswith(" selector") and source_name != EMBEDDING_SELECTOR_SOURCE_LABEL

            if is_selector_source:
                base_model_name = selector_source_to_model_label(source_name)
                trace_color = model_colors.get(base_model_name, selectors_colors.get('single_selector', '#6A1B9A'))
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
import streamlit as st
import json
import os
import pandas as pd

models = ["Claude", "GPT", "Cogito 70b", "Llama 70b"]
datasets = ["BIRD Training", "BIRD Developer", "SPIDER"]
databases = {
    "BIRD Training": ["Human Resources", "Football team", "Chicago Crimes"],
    "BIRD Developer": ["European Schools", "Soccer teams", "Altro database"],
    "SPIDER": ["Database 1", "Database 2", "Database 3"]
}

# Load query datasets
@st.cache_data
def load_queries():
    """Load all query datasets from JSON files"""
    queries_data = {}
    
    try:
        # Load BIRD Training queries
        with open('datasets/bird_training_queries.json', 'r', encoding='utf-8') as f:
            queries_data['BIRD Training'] = json.load(f)
        
        # Load BIRD Developer queries
        with open('datasets/bird_developer_queries.json', 'r', encoding='utf-8') as f:
            queries_data['BIRD Developer'] = json.load(f)
        
        # Load SPIDER queries
        with open('datasets/spider_queries.json', 'r', encoding='utf-8') as f:
            queries_data['SPIDER'] = json.load(f)
        
        return queries_data
    except FileNotFoundError as e:
        st.error(f"Error loading query files: {e}")
        return {}

# Load queries on app start
all_queries = load_queries()

# Load model results
@st.cache_data
def load_model_results():
    """Load all model performance results from JSON files"""
    results_data = {}
    
    try:
        # Load individual model results
        for model in ["claude", "gpt", "cogito_70b", "llama_70b"]:
            with open(f'results/{model}_results.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Normalize model name
                model_key = model.replace("_", " ").title()
                if model == "cogito_70b":
                    model_key = "Cogito 70b"
                elif model == "llama_70b":
                    model_key = "Llama 70b"
                results_data[model_key] = data
        
        # Load ensemble results
        with open('results/ensemble_results.json', 'r', encoding='utf-8') as f:
            results_data['ensemble'] = json.load(f)
        
        # Load judge results
        with open('results/judge_results.json', 'r', encoding='utf-8') as f:
            results_data['judge'] = json.load(f)
        
        return results_data
    except FileNotFoundError as e:
        st.error(f"Error loading results files: {e}")
        return {}

# Load results on app start
all_results = load_model_results()

def get_result_for_query(model_name, dataset_name, database_name, query_id, metric_key):
    """
    Get the performance result for a specific query.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        database_name: Name of the database
        query_id: ID of the query
        metric_key: Metric key (exec_acc, exact_match, tdex)
    
    Returns:
        Performance value (0-100) or None if not found
    """
    if model_name not in all_results:
        return None
    
    model_data = all_results[model_name]
    
    # Navigate to the query result
    if 'results' in model_data:
        if dataset_name in model_data['results']:
            if database_name in model_data['results'][dataset_name]:
                queries = model_data['results'][dataset_name][database_name]
                for query_result in queries:
                    if query_result['query_id'] == query_id:
                        if metric_key == 'tdex':
                            # For TDEX, return the model's score from tdex dict
                            return query_result.get('tdex', {}).get(model_name, None)
                        else:
                            return query_result.get(metric_key, None)
    return None

def get_ensemble_result(ensemble_models, dataset_name, database_name, query_id):
    """
    Get ensemble result for a specific query and model combination.
    
    Args:
        ensemble_models: List of model names in the ensemble
        dataset_name: Name of the dataset
        database_name: Name of the database
        query_id: ID of the query
    
    Returns:
        Ensemble performance value (0-100) or None if not found
    """
    if 'ensemble' not in all_results:
        return None
    
    # Create ensemble key from sorted model names
    ensemble_key = "+".join(sorted(ensemble_models))
    
    ensemble_data = all_results['ensemble'].get('ensembles', {})
    
    if ensemble_key in ensemble_data:
        if dataset_name in ensemble_data[ensemble_key]:
            if database_name in ensemble_data[ensemble_key][dataset_name]:
                queries = ensemble_data[ensemble_key][dataset_name][database_name]
                for query_result in queries:
                    if query_result['query_id'] == query_id:
                        return query_result.get('score', None)
    
    return None

def get_judge_result(judge_model, evaluated_model, dataset_name, database_name, query_id):
    """
    Get judge score for a specific model's query result.
    
    Args:
        judge_model: Name of the judge model
        evaluated_model: Name of the model being evaluated
        dataset_name: Name of the dataset
        database_name: Name of the database
        query_id: ID of the query
    
    Returns:
        Judge score (0-100) or None if not found
    """
    if 'judge' not in all_results:
        return None
    
    judge_data = all_results['judge'].get('judge_scores', {})
    
    if evaluated_model in judge_data:
        if dataset_name in judge_data[evaluated_model]:
            if database_name in judge_data[evaluated_model][dataset_name]:
                queries = judge_data[evaluated_model][dataset_name][database_name]
                for query_result in queries:
                    if query_result['query_id'] == query_id:
                        return query_result.get('score', None)
    
    return None

def collect_active_results(active_queries, selected_models, selected_metrics_dict, tdex_models=None, ensemble_models=None, judge_models=None):
    """
    Collect all relevant results for active queries based on selections.
    
    Args:
        active_queries: List of active query dictionaries
        selected_models: List of selected model names
        selected_metrics_dict: Dictionary of selected metrics {metric_key: bool}
        tdex_models: List of models selected for TDEX metric (optional)
        ensemble_models: List of models selected for ensemble (optional)
        judge_models: List of models selected as judges (optional)
    
    Returns:
        DataFrame with columns: query_id, dataset, database, model, metric, value
    """
    results_list = []
    
    for query in active_queries:
        query_id = query['id']
        dataset = query['dataset']
        database = query['database']
        
        # Collect results for each selected model and metric
        for model in selected_models:
            # Execution Accuracy
            if selected_metrics_dict.get('exec_acc', False):
                exec_acc = get_result_for_query(model, dataset, database, query_id, 'exec_acc')
                if exec_acc is not None:
                    results_list.append({
                        'query_id': query_id,
                        'dataset': dataset,
                        'database': database,
                        'model': model,
                        'metric': 'exec_acc',
                        'value': exec_acc
                    })
            
            # Exact Match
            if selected_metrics_dict.get('exact_match', False):
                exact_match = get_result_for_query(model, dataset, database, query_id, 'exact_match')
                if exact_match is not None:
                    results_list.append({
                        'query_id': query_id,
                        'dataset': dataset,
                        'database': database,
                        'model': model,
                        'metric': 'exact_match',
                        'value': exact_match
                    })
            
            # TDEX (uses tdex_models if specified)
            if selected_metrics_dict.get('tdex', False) and tdex_models:
                if model in tdex_models:
                    tdex_score = get_result_for_query(model, dataset, database, query_id, 'tdex')
                    if tdex_score is not None:
                        results_list.append({
                            'query_id': query_id,
                            'dataset': dataset,
                            'database': database,
                            'model': model,
                            'metric': 'tdex',
                            'value': tdex_score
                        })
            
            # Judge (if model is being judged)
            if selected_metrics_dict.get('llm_judge', False) and judge_models:
                for judge in judge_models:
                    judge_score = get_judge_result(judge, model, dataset, database, query_id)
                    if judge_score is not None:
                        results_list.append({
                            'query_id': query_id,
                            'dataset': dataset,
                            'database': database,
                            'model': model,
                            'metric': f'judge_{judge}',
                            'value': judge_score
                        })
        
        # Ensemble (separate from individual models)
        if selected_metrics_dict.get('llms_ensemble', False) and ensemble_models and len(ensemble_models) >= 2:
            ensemble_score = get_ensemble_result(ensemble_models, dataset, database, query_id)
            if ensemble_score is not None:
                ensemble_name = "+".join(sorted(ensemble_models))
                results_list.append({
                    'query_id': query_id,
                    'dataset': dataset,
                    'database': database,
                    'model': ensemble_name,
                    'metric': 'llms_ensemble',
                    'value': ensemble_score
                })
    
    return pd.DataFrame(results_list) if results_list else pd.DataFrame()


metrics = [
    {"name": "Execution Accuracy", "key": "exec_acc", "default": False, "tooltip": "Measures if the generated SQL query executes without errors and produces correct results"},
    {"name": "Exact Match", "key": "exact_match", "default": False, "tooltip": "Checks if the generated SQL exactly matches the reference query"},
    {"name": "TDEX", "key": "tdex", "default": False, "tooltip": "Test-suite-based Database EXecution accuracy - evaluates query validity with test cases"},
    {"name": "LLMs Ensemble", "key": "llms_ensemble", "default": False, "tooltip": "Combines predictions from multiple language models to improve accuracy"},
    {"name": "LLM as a Judge", "key": "llm_judge", "default": False, "tooltip": "Uses a language model to evaluate the quality and correctness of generated queries"}
]

# Color definitions - organized by color families
model_colors = {
    "Claude": "#2E5EAA",  # Deep blue
    "GPT": "#4A90E2",  # Medium blue
    "Cogito 70b": "#7CB3E9",  # Light blue
    "Llama 70b": "#A8D0F0"  # Very light blue
}

dataset_colors = {
    "BIRD Training": "#E85D04",  # Dark orange
    "BIRD Developer": "#FF9500",  # Bright orange
    "SPIDER": "#FFBB66"  # Light orange
}

metric_colors = {
    "exec_acc": "#2E7D32",  # Dark green
    "exact_match": "#66BB6A",  # Light green
    "tdex": "#6A1B9A",  # Dark purple
    "llms_ensemble": "#AB47BC",  # Medium purple
    "llm_judge": "#CE93D8"  # Light purple
}

st.set_page_config(layout="wide")

# Define CSS styles for colors
st.markdown("""
<style>
/* Model colors - Blue family */
.color-claude { color: #2E5EAA !important; }
.color-gpt { color: #4A90E2 !important; }
.color-cogito { color: #7CB3E9 !important; }
.color-llama { color: #A8D0F0 !important; }

/* Dataset colors - Orange family */
.color-bird-training { color: #E85D04 !important; }
.color-bird-developer { color: #FF9500 !important; }
.color-spider { color: #FFBB66 !important; }

/* Metric colors - Green/Purple family */
.color-exec-acc { color: #2E7D32 !important; }
.color-exact-match { color: #66BB6A !important; }
.color-tdex { color: #6A1B9A !important; }
.color-llms-ensemble { color: #AB47BC !important; }
.color-llm-judge { color: #CE93D8 !important; }

/* Background colors for conditional boxes */
.bg-bird-training { background-color: rgba(232, 93, 4, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-bird-developer { background-color: rgba(255, 149, 0, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-spider { background-color: rgba(255, 187, 102, 0.15) !important; padding: 10px; border-radius: 5px; }
.bg-tdex { background-color: rgba(106, 27, 154, 0.15) !important; padding: 10px; border-radius: 5px; }
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
        "Claude": "color-claude",
        "GPT": "color-gpt",
        "Cogito 70b": "color-cogito",
        "Llama 70b": "color-llama"
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
        "exec_acc": "color-exec-acc",
        "exact_match": "color-exact-match",
        "tdex": "color-tdex",
        "llms_ensemble": "color-llms-ensemble",
        "llm_judge": "color-llm-judge"
    }
    return class_map.get(metric_key, "")

def filter_queries(queries, complexity_range, length_range, tables_range, attributes_range=None):
    """
    Filter queries based on complexity, length, tables, and attributes involved ranges.
    
    Args:
        queries: List of query dictionaries
        complexity_range: Tuple (min, max) for complexity
        length_range: Tuple (min, max) for length
        tables_range: Tuple (min, max) for tables
        attributes_range: Tuple (min, max) for attributes (optional)
    
    Returns:
        Filtered list of queries
    """
    filtered = []
    for query in queries:
        if (complexity_range[0] <= query['complexity'] <= complexity_range[1] and
            length_range[0] <= query['length'] <= length_range[1] and
            tables_range[0] <= query['tables'] <= tables_range[1]):
            
            # Check attributes if range is provided
            if attributes_range is not None:
                if attributes_range[0] <= query.get('attributes', 0) <= attributes_range[1]:
                    filtered.append(query)
            else:
                filtered.append(query)
    return filtered

def get_queries_for_databases(dataset_name, database_names):
    """
    Get all queries for the specified databases within a dataset.
    
    Args:
        dataset_name: Name of the dataset
        database_names: List of database names
    
    Returns:
        Dictionary mapping database names to their queries
    """
    if dataset_name not in all_queries:
        return {}
    
    result = {}
    for db_name in database_names:
        if db_name in all_queries[dataset_name]:
            result[db_name] = all_queries[dataset_name][db_name]
    return result

def collect_all_selected_queries(selected_datasets_list, selected_databases_list):
    """
    Collect all queries from selected datasets and databases.
    
    Args:
        selected_datasets_list: List of selected dataset names
        selected_databases_list: List of selected database names
    
    Returns:
        List of all query dictionaries with added metadata (dataset, database)
    """
    all_selected = []
    
    for dataset_name in selected_datasets_list:
        if dataset_name not in all_queries:
            continue
            
        dataset_dbs = databases.get(dataset_name, [])
        
        for db_name in dataset_dbs:
            # Include database if it's in selected list or if no databases are selected yet
            if db_name in selected_databases_list or len(selected_databases_list) == 0:
                if db_name in all_queries[dataset_name]:
                    queries = all_queries[dataset_name][db_name]
                    # Add metadata to each query
                    for query in queries:
                        query_with_meta = query.copy()
                        query_with_meta['dataset'] = dataset_name
                        query_with_meta['database'] = db_name
                        all_selected.append(query_with_meta)
    
    return all_selected

def get_query_ranges(queries_list):
    """
    Calculate min/max ranges for query attributes.
    
    Args:
        queries_list: List of query dictionaries
    
    Returns:
        Dictionary with min/max values for each attribute
    """
    if not queries_list:
        return {
            'complexity': (0, 3),
            'length': (0, 3),
            'tables': (0, 12),
            'attributes': (0, 20)
        }
    
    complexities = [q['complexity'] for q in queries_list]
    lengths = [q['length'] for q in queries_list]
    tables = [q['tables'] for q in queries_list]
    attributes = [q.get('attributes', 0) for q in queries_list]
    
    return {
        'complexity': (min(complexities), max(complexities)),
        'length': (min(lengths), max(lengths)),
        'tables': (min(tables), max(tables)),
        'attributes': (min(attributes), max(attributes))
    }

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
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    checked = st.checkbox("", value=False, key=f"dataset_{i}", label_visibility="collapsed")
                with col2:
                    st.markdown(f'<p class="{css_class} text-item">{dataset}</p>', unsafe_allow_html=True)
                if checked:
                    selected_datasets.append(dataset)
        
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
        # Selezione TDEX (condizionale)
        if selected_metrics.get("tdex", False):
            css_class = get_metric_class("tdex")
            
            with st.container(border=True):
                st.markdown(f'<div class="tooltip-container"><h3 class="{css_class}">TDEX</h3><span class="tooltip-text">Select models to use for TDEX metric evaluation</span></div>', unsafe_allow_html=True)
                selected_tdex = []
                for i, model in enumerate(models):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox("", value=False, key=f"tdex_{i}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="bg-tdex">{model}</div>', unsafe_allow_html=True)
                    if checked:
                        selected_tdex.append(model)

        # Selezione Ensemble (condizionale)
        if selected_metrics.get("llms_ensemble", False):
            css_class = get_metric_class("llms_ensemble")
            
            with st.container(border=True):
                st.markdown(f'<div class="tooltip-container"><h3 class="{css_class}">Selezione Ensemble</h3><span class="tooltip-text">Select models to include in the ensemble prediction</span></div>', unsafe_allow_html=True)
                selected_ensemble = []
                for i, model in enumerate(models):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox("", value=False, key=f"ensemble_{i}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="bg-llms-ensemble">{model}</div>', unsafe_allow_html=True)
                    if checked:
                        selected_ensemble.append(model)
        
        # Selezione Judge (condizionale - appare solo se LLM as a Judge è selezionato)
        if selected_metrics.get("llm_judge", False):
            css_class = get_metric_class("llm_judge")
            
            with st.container(border=True):
                st.markdown(f'<div class="tooltip-container"><h3 class="{css_class}">Selezione Judge</h3><span class="tooltip-text">Select which model will act as the judge for evaluation</span></div>', unsafe_allow_html=True)
                selected_judges = []
                for i, model in enumerate(models):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox("", value=False, key=f"judge_{i}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="bg-llm-judge">{model}</div>', unsafe_allow_html=True)
                    if checked:
                        selected_judges.append(model)

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
            st.markdown('<div class="tooltip-container"><h3>Selettori Query</h3><span class="tooltip-text">Filter queries by complexity, length, and table involvement</span></div>', unsafe_allow_html=True)
            
            # Show available query info
            if len(all_selected_queries) > 0:
                st.caption(f"📚 Query disponibili: {len(all_selected_queries)} | "
                          f"Range disponibili - C:[{query_ranges['complexity'][0]}-{query_ranges['complexity'][1]}] "
                          f"L:[{query_ranges['length'][0]}-{query_ranges['length'][1]}] "
                          f"T:[{query_ranges['tables'][0]}-{query_ranges['tables'][1]}] "
                          f"A:[{query_ranges['attributes'][0]}-{query_ranges['attributes'][1]}]")
            else:
                st.info("ℹ️ Seleziona dataset e database per vedere le query disponibili")
            
            # Dynamic sliders based on available query ranges
            complexity_range = st.slider(
                "Complessità", 
                min_value=query_ranges['complexity'][0], 
                max_value=query_ranges['complexity'][1], 
                value=query_ranges['complexity'], 
                key="complexity_slider",
                help="Filtra per complessità della query (0=semplice, 3=complessa)"
            )
            length_range = st.slider(
                "Lunghezza", 
                min_value=query_ranges['length'][0], 
                max_value=query_ranges['length'][1], 
                value=query_ranges['length'], 
                key="length_slider",
                help="Filtra per lunghezza del testo della query"
            )
            tables_range = st.slider(
                "Tabelle Coinvolte", 
                min_value=query_ranges['tables'][0], 
                max_value=query_ranges['tables'][1], 
                value=query_ranges['tables'], 
                key="tables_slider",
                help="Filtra per numero di tabelle coinvolte nella query"
            )
            attributes_range = st.slider(
                "Attributi Coinvolti", 
                min_value=query_ranges['attributes'][0], 
                max_value=query_ranges['attributes'][1], 
                value=query_ranges['attributes'], 
                key="attributes_slider",
                help="Filtra per numero di attributi/colonne coinvolte"
            )
            
            # Filter queries based on slider values
            active_queries = filter_queries(
                all_selected_queries, 
                complexity_range, 
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
            
            # Show breakdown by dataset/database
            if total_active > 0:
                with st.expander("📊 Ripartizione query attive"):
                    breakdown = {}
                    for query in active_queries:
                        key = f"{query['dataset']} → {query['database']}"
                        breakdown[key] = breakdown.get(key, 0) + 1
                    
                    for key, count in breakdown.items():
                        st.markdown(f"- **{key}:** {count} query")
            
            # Store active queries in session state for metric calculations
            st.session_state['active_queries'] = active_queries
            st.session_state['active_queries_count'] = total_active
            
            # Create dataframe for active queries
            if total_active > 0:
                df_active = pd.DataFrame(active_queries)
                st.session_state['active_queries_df'] = df_active
            else:
                st.session_state['active_queries_df'] = pd.DataFrame()
            
            # Show active queries in an expander
            if total_active > 0:
                with st.expander(f"Visualizza query attive ({total_active})"):
                    # Display as dataframe with selected columns
                    display_df = df_active[['id', 'query', 'dataset', 'database', 'complexity', 'length', 'tables', 'attributes']].copy()
                    display_df.columns = ['ID', 'Query', 'Dataset', 'Database', 'Complessità', 'Lunghezza', 'Tabelle', 'Attributi']
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Download button for active queries
                    csv = df_active.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Scarica query attive (CSV)",
                        data=csv,
                        file_name="active_queries.csv",
                        mime="text/csv"
                    )
            
            # Collect active results based on all selections
            # Get TDEX models if TDEX is selected
            active_tdex_models = []
            if selected_metrics.get("tdex", False):
                # Retrieve selected TDEX models from checkboxes
                for i, model in enumerate(models):
                    if st.session_state.get(f"tdex_{i}", False):
                        active_tdex_models.append(model)
            
            # Get Ensemble models if Ensemble is selected
            active_ensemble_models = []
            if selected_metrics.get("llms_ensemble", False):
                for i, model in enumerate(models):
                    if st.session_state.get(f"ensemble_{i}", False):
                        active_ensemble_models.append(model)
            
            # Get Judge models if Judge is selected
            active_judge_models = []
            if selected_metrics.get("llm_judge", False):
                for i, model in enumerate(models):
                    if st.session_state.get(f"judge_{i}", False):
                        active_judge_models.append(model)
            
            # Collect all active results
            active_results_df = collect_active_results(
                active_queries,
                selected_models,
                selected_metrics,
                tdex_models=active_tdex_models if active_tdex_models else None,
                ensemble_models=active_ensemble_models if active_ensemble_models else None,
                judge_models=active_judge_models if active_judge_models else None
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

        
        # Modificatori Esecuzione
        with st.container(border=True):
            st.markdown('<div class="tooltip-container"><h3>Modificatori Esecuzione</h3><span class="tooltip-text">Modify execution parameters and configurations</span></div>', unsafe_allow_html=True)
            mod_schema_gt = st.checkbox("Schema Ground Truth", value=False, key="mod_schema_gt")
        
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
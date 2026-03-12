import streamlit as st
import json
import os
import pandas as pd
import hashlib

models = ["Claude", "GPT", "Cogito 70b", "Llama 70b"]
datasets = ["BIRD Training", "BIRD Developer", "SPIDER"]
databases = {
    "BIRD Training": ["Human Resources", "Football team", "Chicago Crimes"],
    "BIRD Developer": ["European Schools", "Soccer teams", "Altro database"],
    "SPIDER": ["Database 1", "Database 2", "Database 3"]
}

def compute_all_agents(models):
    #compute all combinations of agents for Ensemble metric
    from itertools import combinations
    all_agents = []
    for r in range(1, len(models) + 1):
        for combo in combinations(models, r):
            all_agents.append(" + ".join(combo))
    return all_agents

agents = compute_all_agents(models)

# Load query datasets
@st.cache_data
def load_queries():
    """Load all query datasets from JSON files"""
    queries_list = []
    
    try:
        # Load BIRD Training queries
        with open('datasets/bird_training_queries.json', 'r', encoding='utf-8') as f:
            q_data = pd.DataFrame(json.load(f))
            q_data['dataset'] = 'BIRD Training'
            queries_list.append(q_data)
        
        # Load BIRD Developer queries
        with open('datasets/bird_developer_queries.json', 'r', encoding='utf-8') as f:
            q_data_developer = pd.DataFrame(json.load(f))
            q_data_developer['dataset'] = 'BIRD Developer'
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

# Load model results
@st.cache_data
def load_model_results():
    """Load all model performance results from JSON files and flatten nested structure"""
    all_rows = []
    
    # Metric name mapping
    metric_mapping = {
        "Execution Accuracy": "exec_acc",
        "Exact Match": "exact_match",
        "TDEX": "tdex",
        "Ensemble": "llms_ensemble",
        "LLMs as a Judge": "llm_judge"
    }
    
    try:
        # Load individual model results
        for model in models:
            lower_model = model.lower().replace(" ", "_")
            with open(f'results/{lower_model}_results.json', 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                # Iterate through datasets
                for dataset_name, queries in json_data.get('datasets', {}).items():
                    # Iterate through queries in this dataset
                    for query_data in queries:
                        query_id = query_data['id']
                        database = query_data['database']
                        
                        # Iterate through metrics for this query
                        for metric_name, metric_value in query_data.get('metrics', {}).items():
                            metric_key = metric_mapping.get(metric_name, metric_name)
                            
                            # Handle simple metrics (exec_acc, exact_match)
                            if isinstance(metric_value, (int, float)):
                                all_rows.append({
                                    'dataset': dataset_name,
                                    'database': database,
                                    'id': query_id,
                                    'model': model,
                                    'metric': metric_key,
                                    'value': metric_value
                                })
                            # Handle complex metrics (TDEX, Ensemble, Judge) - nested dicts
                            elif isinstance(metric_value, dict):
                                # For now, we'll store the main model's value
                                # You can extend this to handle ensemble combinations
                                agents = metric_value.keys()
                                for agent in agents:
                                    all_rows.append({
                                        'dataset': dataset_name,
                                        'database': database,
                                        'id': query_id,
                                        'model': model,
                                        'metric': metric_key,
                                        'agent': agent,
                                        'value': metric_value[agent]
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

# Load results on app start
all_results = load_model_results()

def string_to_deterministic_int(s):
    """Convert a string to a deterministic integer using SHA256 hash.
    This allows faster equality comparisons than string comparisons.
    """
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)

# Create a general_id column in both dataframes for quick lookup using vectorized operations
if not all_queries.empty:
    # Vectorized string concatenation and hash creation
    all_queries['g_id'] = (all_queries['dataset'].astype(str) + '|' + 
                           all_queries['database'].astype(str) + '|' + 
                           all_queries['id'].astype(str)).apply(string_to_deterministic_int)
if not all_results.empty:    
    # Vectorized string concatenation and hash creation
    all_results['g_id'] = (all_results['dataset'].astype(str) + '|' + 
                           all_results['database'].astype(str) + '|' + 
                           all_results['id'].astype(str)).apply(string_to_deterministic_int)

def collect_active_results(active_queries, selected_models, selected_metrics_dict, tdex_agents=None, ensemble_agents=None, judge_agents=None):
    """
    Collect all relevant results for active queries based on selections.
    
    Args:
        active_queries: DataFrame of active queries
        selected_models: List of selected model names
        selected_metrics_dict: Dictionary of selected metrics {metric_key: bool}
        tdex_agents: List of agents selected for TDEX metric (optional)
        ensemble_agents: List of agents selected for ensemble (optional)
        judge_agents: List of agents selected as judges (optional)
    
    Returns:
        DataFrame with columns: query_id, dataset, database, model, metric, agent, value
    """
    if all_results.empty or active_queries.empty:
        return pd.DataFrame()
    
    # Get active query IDs using pandas Series
    active_g_ids = active_queries['g_id'] if 'g_id' in active_queries.columns else pd.Series(dtype='int64')
    
    # Get selected metrics using pandas-friendly list comprehension
    selected_metric_keys = [m for m, selected in selected_metrics_dict.items() if selected]
    
    # Use pandas boolean indexing with & instead of 'and'
    active_results = all_results[
        all_results['g_id'].isin(active_g_ids) & 
        all_results['model'].isin(selected_models) & 
        all_results['metric'].isin(selected_metric_keys)
    ].copy()

    # Filter by agents for complex metrics if specified
    if tdex_agents is not None:
        active_results = active_results[~((active_results['metric'] == 'tdex') & ~active_results['agent'].isin(tdex_agents))]
    if ensemble_agents is not None:
        active_results = active_results[~((active_results['metric'] == 'llms_ensemble') & ~active_results['agent'].isin(ensemble_agents))]
    if judge_agents is not None:
        active_results = active_results[~((active_results['metric'] == 'llm_judge') & ~active_results['agent'].isin(judge_agents))]

    return active_results


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

def filter_queries(queries_df, complexity_range, length_range, tables_range, attributes_range=None):
    """
    Filter queries based on complexity, length, tables, and attributes involved ranges.
    
    Args:
        queries_df: DataFrame of queries
        complexity_range: Tuple (min, max) for complexity
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
        (queries_df['complexity'] >= complexity_range[0]) & 
        (queries_df['complexity'] <= complexity_range[1]) &
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
            'complexity': (0, 3),
            'length': (0, 3),
            'tables': (0, 30),
            'attributes': (0, 30)
        }
    
    # Use pandas vectorized min/max operations
    return {
        'complexity': (int(queries_df['complexity'].min()), int(queries_df['complexity'].max())),
        'length': (int(queries_df['length'].min()), int(queries_df['length'].max())),
        'tables': (int(queries_df['tables'].min()), int(queries_df['tables'].max())),
        'attributes': (int(queries_df['attributes'].min()) if 'attributes' in queries_df.columns else 0, 
                      int(queries_df['attributes'].max()) if 'attributes' in queries_df.columns else 30)
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

            # Calculate ensemble string - join selected models with " + "
            ensemble_string = " + ".join(active_ensemble_models) if active_ensemble_models else None
            
            # Collect all active results
            active_results_df = collect_active_results(
                active_queries,
                selected_models,
                selected_metrics,
                tdex_agents=active_tdex_models if active_tdex_models else None,
                ensemble_agents=[ensemble_string] if ensemble_string else None,
                judge_agents=active_judge_models if active_judge_models else None
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

    # Display active results in a bar plot grouped by metric and colored by model
    if 'active_results_df' in st.session_state and not st.session_state['active_results_df'].empty:
        import plotly.graph_objects as go
        
        results_df = st.session_state['active_results_df'].copy()
        
        # Create metric labels that include agent info for complex metrics
        def create_metric_label(row):
            """Create a display label for the metric, including agent info if present"""
            metric_names = {
                "exec_acc": "Execution Accuracy",
                "exact_match": "Exact Match",
                "tdex": "TDEX",
                "llms_ensemble": "LLMs Ensemble",
                "llm_judge": "LLM as Judge"
            }
            metric_display = metric_names.get(row['metric'], row['metric'])
            
            # Add agent info for complex metrics
            if 'agent' in row and pd.notna(row['agent']):
                return f"{metric_display} ({row['agent']})"
            return metric_display
        
        results_df['metric_label'] = results_df.apply(create_metric_label, axis=1)
        
        # Calculate average value for each model-metric combination
        grouped_data = results_df.groupby(['metric_label', 'model'])['value'].mean().reset_index()
        
        # Create the bar chart
        fig = go.Figure()
        
        # Get unique metrics in order (preserve ordering)
        unique_metrics = grouped_data['metric_label'].unique()
        
        # Add a trace for each model
        for model in selected_models:
            model_data = grouped_data[grouped_data['model'] == model]
            
            # Ensure all metrics are represented (fill missing with None)
            plot_values = []
            for metric in unique_metrics:
                metric_row = model_data[model_data['metric_label'] == metric]
                if len(metric_row) > 0:
                    plot_values.append(metric_row['value'].values[0])
                else:
                    plot_values.append(None)
            
            fig.add_trace(go.Bar(
                name=model,
                x=unique_metrics,
                y=plot_values,
                marker_color=model_colors.get(model, '#888888'),
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
                title="Models",
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
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


# Resolve project root (parent folder of this file's directory)
BASE_DIR = Path(__file__).resolve().parent


def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON data from a file, resolving relative to project root."""
    path = Path(filepath)
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_schema_info(db_id: str, tables_file: str = 'train_tables.json') -> Optional[Dict[str, Any]]:
    """
    Retrieve schema information for a specific database.
    
    Args:
        db_id: The database identifier (e.g., 'human_resources')
        tables_file: Path to the tables JSON file
    
    Returns:
        Dictionary containing schema information or None if not found
    """
    tables_data = load_json_file(tables_file)
    
    for schema in tables_data:
        if schema.get('db_id') == db_id:
            return schema
    
    return None


def get_queries(db_id: str, queries_file: str = 'train.json') -> List[Dict[str, str]]:
    """
    Retrieve all queries for a specific database.
    
    Args:
        db_id: The database identifier (e.g., 'human_resources')
        queries_file: Path to the queries JSON file
    
    Returns:
        List of dictionaries containing question, evidence, and SQL query
    """
    queries_data = load_json_file(queries_file)
    
    db_queries = [
        {
            'question_id': query.get('question_id', -1),
            'question': query.get('question', ''),
            'evidence': query.get('evidence', ''),
            'SQL': query.get('SQL', '')
        }
        for query in queries_data
        if query.get('db_id') == db_id
    ]
    
    return db_queries

def format_schema_info_for_prompt(schema_info: Dict[str, Any]) -> str:
    """
    Format the schema information into a readable string for prompt usage.
    
    Args:
        schema_info: Dictionary containing schema information
    """
    
    keys = schema_info.keys()
    new_schema = {}
    
    # Copy relevant keys
    new_schema['table_names'] = schema_info['table_names_original']
    
    # Build column_by_table mapping
    new_schema['column_by_table'] = {}
    for col in schema_info['column_names_original']:
        table_idx, col_name = col
        if table_idx < 0 or table_idx >= len(schema_info['table_names_original']):
            continue
        table_name = schema_info['table_names_original'][table_idx]
        if table_name not in new_schema['column_by_table']:
            new_schema['column_by_table'][table_name] = []
        column_type = schema_info['column_types'][schema_info['column_names_original'].index(col)]
        #check if primary key
        primary = False
        if schema_info['column_names_original'].index(col) in schema_info['primary_keys']:
            primary = True
        new_schema['column_by_table'][table_name].append([col_name, column_type, primary])
        
    # traslate foreign keys to names
    new_schema['foreign_keys'] = []
    for fk in schema_info['foreign_keys']:
        col1_idx, col2_idx = fk
        col1 = schema_info['column_names_original'][col1_idx]
        col2 = schema_info['column_names_original'][col2_idx]
        table1 = schema_info['table_names_original'][col1[0]]
        table2 = schema_info['table_names_original'][col2[0]]
        new_schema['foreign_keys'].append([[table1, col1[1]], [table2, col2[1]]])
    
    return new_schema

def extract_relevant_tables_from_schema(schema_info, necessary_tables):
    #Extract only the necessary tables and their columns from the full schema.
    extracted_schema = {
        'table_names': [],
        'column_by_table': {},
        'foreign_keys': []
    }
    for table in necessary_tables:
        if table in schema_info['table_names']:
            extracted_schema['table_names'].append(table)
            extracted_schema['column_by_table'][table] = schema_info['column_by_table'][table]
    # Extract foreign keys that involve only the necessary tables
    for fk in schema_info['foreign_keys']:
        table1, table2 = fk[0][0], fk[1][0]
        if table1 in necessary_tables and table2 in necessary_tables:
            extracted_schema['foreign_keys'].append(fk)

    return extracted_schema
        

def format_schema_info(schema_info: Dict[str, Any]) -> str:
    """
    Format the schema information into a readable string.
    
    Args:
        schema_info: Dictionary containing schema information
    
    Returns:
        Formatted string representation of the schema
    """
    if not schema_info:
        return "No schema information available."
    
    output = []
    output.append(f"Database ID: {schema_info.get('db_id', 'N/A')}")
    output.append("\n" + "="*80)
    
    # Tables section
    output.append("\nTABLE NAMES:")
    output.append("-" * 80)
    table_names_original = schema_info.get('table_names_original', [])
    table_names = schema_info.get('table_names', [])
    for i, (orig, normalized) in enumerate(zip(table_names_original, table_names)):
        output.append(f"\n  [{i}] {orig} (normalized: {normalized})")
    
    # Columns section
    output.append("\n\n" + "="*80)
    output.append("\nCOLUMNS:")
    output.append("-" * 80)
    
    column_names_original = schema_info.get('column_names_original', [])
    column_names = schema_info.get('column_names', [])
    column_types = schema_info.get('column_types', [])
    
    for idx, (orig_col, norm_col, col_type) in enumerate(zip(column_names_original, column_names, column_types)):
        table_idx, col_name = orig_col
        norm_table_idx, norm_col_name = norm_col
        
        if table_idx == -1:
            output.append(f"\n  [{idx}] {col_name} -> {norm_col_name}")
        else:
            table_name = table_names_original[table_idx] if table_idx < len(table_names_original) else "Unknown"
            output.append(f"\n  [{idx}] {table_name}.{col_name} -> {norm_col_name} ({col_type})")
    
    # Primary keys section
    output.append("\n\n" + "="*80)
    output.append("\nPRIMARY KEYS:")
    output.append("-" * 80)
    primary_keys = schema_info.get('primary_keys', [])
    if primary_keys:
        for pk in primary_keys:
            if pk < len(column_names_original):
                col_info = column_names_original[pk]
                table_idx, col_name = col_info
                output.append(f"\n  Column Index {pk}: {col_name}")
    else:
        output.append("\n  None")
    
    # Foreign keys section
    output.append("\n\n" + "="*80)
    output.append("\nFOREIGN KEYS:")
    output.append("-" * 80)
    foreign_keys = schema_info.get('foreign_keys', [])
    if foreign_keys:
        for fk in foreign_keys:
            output.append(f"\n  {fk}")
    else:
        output.append("\n  None")
    
    # Additional metadata
    output.append("\n\n" + "="*80)
    output.append("\nMETADATA:")
    output.append("-" * 80)
    output.append(f"\n  Total Tables: {len(table_names_original)}")
    output.append(f"  Total Columns: {len(column_names_original)}")
    output.append(f"  Total Column Types: {len(column_types)}")
    output.append(f"  Primary Keys Count: {len(primary_keys)}")
    output.append(f"  Foreign Keys Count: {len(foreign_keys)}")
    
    return "\n".join(output)
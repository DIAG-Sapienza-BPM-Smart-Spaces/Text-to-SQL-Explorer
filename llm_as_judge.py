import sys
import json
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from langchain.schema.messages import HumanMessage

# Calculate paths and ensure project root is in sys.path
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]  # /DEMO-PAPER

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    
from retrieve_database_info import get_schema_info, format_schema_info_for_prompt, get_queries

models = ["deepseek-chat", "cogito_70b", "qwen2.5-coder_32b", "qwen3-coder_30b"]

llm_prompt_binary_choice = """
You are an expert SQL programmer. Given a natural language query, an evidence, the schema of the database, and one SQL candidate, determine if the candidate is correct and answer the natural language query.
---
Natural language query:
---
{question}
---
Evidence:
---
{evidence}
---
Database schema (with information about tables, column names/types, primary keys and foreign keys):
---
{schema}
---
Candidate:
---
{query}
---
Your answer should be in the following format (either ACCEPT or REJECT as the choice):
---
<your reasoning>
``choice
<CHOICE>
```
---
Example of an answer:
---
The natural language query is asking for the names of all customers who have made a purchase in the last month. The evidence provided includes information about the "customers" table, which contains columns such as "customer_id", "name", and "purchase_date". Based on the schema, we can see that the "customers" table has the necessary columns to answer the query. The candidate SQL correctly selects the "name" column from the "customers" table and filters the results based on the "purchase_date" column to include only those customers who made a purchase in the last month. Therefore, the candidate is correct and should be accepted.
```choice
ACCEPT
```
---
"""
    
llm_prompt_candidate_choice = """
You are an expert SQL programmer. Given a natural language query, an evidence, the schema of the database, and two SQL candidates, choose the one that is more correct and answer the natural language query using the evidence and the schema.
---
Evidence:
---
{evidence}
---
Database schema (with information about tables, column names/types, primary keys and foreign keys):
---
{schema}
---
Candidate 1:
---
{query_1}
---
Candidate 2:
---
{query_2}
---
Your answer should be in the following format:
---
<your reasoning>
``choice
<CANDIDATE>
```
---
Example of an answer:
---
The natural language query is asking for the names of all customers who have made a purchase in the last month. The evidence provided includes information about the "customers" table, which contains columns such as "customer_id", "name", and "purchase_date". Based on the schema, we can see that the "customers" table has the necessary columns to answer the query. Candidate 1 correctly selects the "name" column from the "customers" table and filters the results based on the "purchase_date" column to include only those customers who made a purchase in the last month. Candidate 2, on the other hand, does not include the necessary filter on the "purchase_date" column, which means it would return all customers regardless of when they made a purchase. Therefore, Candidate 1 is the correct choice.
```choice
CANDIDATE 1
```
---
"""

def initialize_llm(model: str, api_key: str):
        if model == "deepseek-chat":
            from langchain_deepseek.chat_models import ChatDeepSeek
            llm = ChatDeepSeek(api_key=api_key, model=model)
            return llm

def judge(database, query, query_id, schema_file, evidence, judge_arguments, mode = "binary_choice"):
    
    model = judge_arguments["model"]
    llm = judge_arguments["llm"]
    
    schema_info = get_schema_info(database, tables_file=schema_file)
    schema_formatted = format_schema_info_for_prompt(schema_info)
    
    if mode == "binary_choice":
        candidate = judge_arguments["candidate"]
        candidate_model = judge_arguments["candidate_model"]
        llm_prompt = llm_prompt_binary_choice
        prompt = llm_prompt.format(question=query, evidence=evidence, schema=schema_formatted, query=candidate)
    elif mode == "candidate_choice":
        candidate_model_1 = judge_arguments["candidate_1_model"]
        candidate_1 = judge_arguments["candidate_1"]
        candidate_model_2 = judge_arguments["candidate_2_model"]
        candidate_2 = judge_arguments["candidate_2"]
        llm_prompt = llm_prompt_candidate_choice
        prompt = llm_prompt.format(query=query, evidence=evidence, schema=schema_formatted, query_1=candidate_1, query_2=candidate_2)
        
    '''
    print("Prompt for llm:")
    print(prompt)
    print("\n\n")
    '''

    messages = [HumanMessage(content=prompt)]
    
    response = llm.invoke(messages)
    response_content = response.content

    #'''
    print("LLM Response:")
    print(response_content)
    print("\n\n")
    #'''
    
    # Extract schema from response
    if '```' in response_content:
        parts = response_content.split('```')
        reasoning = parts[0].strip()
        choice = parts[1] if len(parts) > 2 else parts[0]
    else:
        choice = response_content
        
    if choice.startswith("choice\n"):
        choice = choice[len("choice\n"):].strip()
    
    choice = choice.strip()
    
    print("Extracted Choice:")
    print(choice)
    print("\n\n")
        
    '''
    #save the sql and results to a file
    with open(_PROJECT_ROOT / "judge_output.txt", "w", encoding="utf-8") as f:
        f.write("Extracted Choice:\n")
        f.write(choice + "\n\n")
    '''
    
    if mode == "binary_choice":
        result = {
            "Query ID": query_id,
            "Database": database,
            "Judge Model": model,
            "Reasoning": reasoning,
            "choice": choice
        }
    elif mode == "candidate_choice":
        result = {
            "Query ID": query_id,
            "Database": database,
            "Judge Model": model,
            "Model 1": candidate_model_1,
            "Model 2": candidate_model_2,
            "Reasoning": reasoning,
            "choice": choice
        }
    
    return result


def _resolve_dataset_files(database: str, schema_file: str = None, query_file: str = None) -> Tuple[str, str]:
    """Resolve schema/query files from explicit inputs or known dataset aliases."""
    default_files = {
        "bird_developer": ("dev_tables.json", "dev.json"),
    }

    if schema_file and query_file:
        return schema_file, query_file

    alias = (database or "").strip().lower()
    if alias in default_files:
        default_schema, default_query = default_files[alias]
        return schema_file or default_schema, query_file or default_query

    if not schema_file or not query_file:
        raise ValueError(
            "schema_file and query_file are required unless database is a known alias like 'bird_developer'."
        )

    return schema_file, query_file


def _build_question_index_for_db(db_id: str, query_file: str) -> Dict[int, Dict[str, Any]]:
    """Build a lookup by question_id for one db_id using retrieve_database_info.get_queries."""
    db_queries = get_queries(db_id=db_id, queries_file=query_file)
    index: Dict[int, Dict[str, Any]] = {}
    for row in db_queries:
        question_id = row.get("question_id")
        if question_id is None:
            continue
        index[int(question_id)] = {
            "question": row.get("question", ""),
            "evidence": row.get("evidence", ""),
        }
    return index


def _atomic_dump_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to reduce risk of corrupted files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def compute_binary_choices_for_sqls(
    judge_model: str = "deepseek-chat",
    llm=None,
    database: str = "bird_developer",
    schema_file: str = None,
    query_file: str = None,
    sqls_dir: str = "sqls",
    output_dir: str = "binary_choices",
    resume: bool = True,
    save_every: int = 10,
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate all SQL outputs in `sqls_dir` with the binary-choice judge and save one file per candidate model.

    Resume strategy:
    - If `resume=True` and an output file already exists for a candidate model, previously judged
      (db_id, question_id) pairs are skipped.
    - Progress is checkpointed every `save_every` items using atomic writes, so interruption only
      loses at most a small tail of unsaved items.
    """
    schema_file, query_file = _resolve_dataset_files(database, schema_file, query_file)

    if llm is None:
        raise ValueError("An instantiated llm must be provided, or initialize it before calling this function.")

    sqls_path = _PROJECT_ROOT / sqls_dir
    output_path = _PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    sql_files = sorted(sqls_path.glob("*_query_results.json"))
    summary: Dict[str, Any] = {
        "judge_model": judge_model,
        "database": database,
        "schema_file": schema_file,
        "query_file": query_file,
        "processed_models": {},
    }

    # Cache question lookups per database to avoid re-reading query files repeatedly.
    per_db_question_index: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for sql_file in sql_files:
        candidate_model = sql_file.name.replace("_query_results.json", "")
        output_file = output_path / f"{candidate_model}_binary_choices.json"

        with open(sql_file, "r", encoding="utf-8") as f:
            sql_rows = json.load(f)

        existing_results: List[Dict[str, Any]] = []
        processed_keys = set()

        if resume and output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            existing_results = existing_payload.get("results", [])
            for r in existing_results:
                db_id = r.get("db_id")
                qid = r.get("question_id")
                if db_id is not None and qid is not None:
                    processed_keys.add((str(db_id), int(qid)))

        results = list(existing_results)
        processed_since_save = 0
        errors_count = 0

        for row in sql_rows:
            db_id = row.get("db_id")
            question_id = row.get("question_id")
            if db_id is None or question_id is None:
                continue

            key = (str(db_id), int(question_id))
            if key in processed_keys:
                continue

            db_key = str(db_id)
            if db_key not in per_db_question_index:
                per_db_question_index[db_key] = _build_question_index_for_db(db_key, query_file)

            query_meta = per_db_question_index[db_key].get(int(question_id))
            if query_meta is None:
                result = {
                    "question_id": int(question_id),
                    "db_id": str(db_id),
                    "candidate_model": candidate_model,
                    "candidate_sql": row.get("extracted_sql") or row.get("generated_sql") or "",
                    "choice": "ERROR",
                    "Reasoning": "No matching question/evidence found in query_file.",
                }
                results.append(result)
                processed_keys.add(key)
                processed_since_save += 1
                errors_count += 1
            else:
                candidate_sql = row.get("extracted_sql") or row.get("generated_sql") or ""
                try:
                    judge_result = judge(
                        database=str(db_id),
                        query=query_meta["question"],
                        query_id=int(question_id),
                        schema_file=schema_file,
                        evidence=query_meta.get("evidence", ""),
                        judge_arguments={
                            "model": judge_model,
                            "llm": llm,
                            "candidate_model": candidate_model,
                            "candidate": candidate_sql,
                        },
                        mode="binary_choice",
                    )

                    judge_result["question_id"] = int(question_id)
                    judge_result["db_id"] = str(db_id)
                    judge_result["candidate_model"] = candidate_model
                    judge_result["candidate_sql"] = candidate_sql
                    results.append(judge_result)
                    processed_keys.add(key)
                    processed_since_save += 1
                except Exception as exc:
                    error_result = {
                        "question_id": int(question_id),
                        "db_id": str(db_id),
                        "candidate_model": candidate_model,
                        "candidate_sql": candidate_sql,
                        "choice": "ERROR",
                        "Reasoning": str(exc),
                    }
                    results.append(error_result)
                    processed_keys.add(key)
                    processed_since_save += 1
                    errors_count += 1
                    if stop_on_error:
                        payload = {
                            "judge_model": judge_model,
                            "candidate_model": candidate_model,
                            "schema_file": schema_file,
                            "query_file": query_file,
                            "total_sql_rows": len(sql_rows),
                            "results": results,
                        }
                        _atomic_dump_json(output_file, payload)
                        raise

            if processed_since_save >= save_every:
                payload = {
                    "judge_model": judge_model,
                    "candidate_model": candidate_model,
                    "schema_file": schema_file,
                    "query_file": query_file,
                    "total_sql_rows": len(sql_rows),
                    "results": results,
                }
                _atomic_dump_json(output_file, payload)
                processed_since_save = 0

        payload = {
            "judge_model": judge_model,
            "candidate_model": candidate_model,
            "schema_file": schema_file,
            "query_file": query_file,
            "total_sql_rows": len(sql_rows),
            "results": results,
        }
        _atomic_dump_json(output_file, payload)

        summary["processed_models"][candidate_model] = {
            "output_file": str(output_file),
            "total_sql_rows": len(sql_rows),
            "saved_results": len(results),
            "errors": errors_count,
        }

    return summary

if __name__ == "__main__":
    model = "deepseek-chat"
    llm = initialize_llm(model=model, api_key="sk-2675b255bc084d70b188e7fccd0aed15")
    
    
    
    
    
    

    
    
    
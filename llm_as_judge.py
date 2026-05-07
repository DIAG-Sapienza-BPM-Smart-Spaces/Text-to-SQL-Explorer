import sys
import json
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langchain.messages import HumanMessage
from common_utils import (
    atomic_dump_json,
)

# Calculate paths and ensure project root is in sys.path
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[0]  # /DEMO-PAPER

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    
from retrieve_database_info import get_schema_info, format_schema_info_for_prompt, get_queries, load_json_file

models = ["deepseek-chat", "cogito_70b", "qwen2.5-coder_32b", "qwen3-coder_30b"]

_FILE_WRITE_LOCK = threading.Lock()

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
<ACCEPT|REJECT>
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

def initialize_llm(model: str, api_key: str, verbose: int = 1):
    if verbose >= 1:
        print(f"[init] Initializing LLM: {model}")
    if model == "deepseek-chat":
        from langchain_deepseek.chat_models import ChatDeepSeek
        llm = ChatDeepSeek(api_key=api_key, model=model)
        if verbose >= 2:
            print(f"[init] ChatDeepSeek instance created for model '{model}'")
        return llm
    if verbose >= 0:
        print(f"[init] WARNING: unsupported model '{model}', returning None")
    return None

def judge(database, query, query_id, schema_file, evidence, judge_arguments, verbose: int = 1):

    model = judge_arguments["model"]
    llm = judge_arguments["llm"]

    if verbose >= 1:
        print(f"[judge] query_id={query_id} | db={database} | mode=binary | judge={model}")

    schema_info = judge_arguments.get("schema_info")
    if schema_info is None:
        schema_info = get_schema_info(database, tables_file=schema_file)
        if verbose >= 2:
            print(f"[judge] Schema source: get_schema_info(db_id='{database}', tables_file='{schema_file}')")
    elif verbose >= 2:
        print(f"[judge] Schema source: preloaded schema_info from batch cache for db_id='{database}'")

    if schema_info is None:
        raise ValueError(f"Schema not found for db_id='{database}' in '{schema_file}'")

    schema_formatted = format_schema_info_for_prompt(schema_info)

    if verbose >= 2:
        print(f"[judge] Schema loaded for '{database}': {list(schema_formatted.get('column_by_table', {}).keys())} tables")

    candidate = judge_arguments["candidate"]
    candidate_model = judge_arguments["candidate_model"]
    llm_prompt = llm_prompt_binary_choice
    prompt = llm_prompt.format(question=query, evidence=evidence, schema=schema_formatted, query=candidate)
    if verbose >= 2:
        print(f"[judge] Question  : {query}")
        print(f"[judge] Evidence  : {evidence}")
        print(f"[judge] Candidate SQL ({candidate_model}): {candidate}")

    if verbose >= 2:
        print("[judge] --- Prompt sent to LLM ---")
        print(prompt)
        print("[judge] --- End of prompt ---\n")

    messages = [HumanMessage(content=prompt)]

    response = llm.invoke(messages)
    response_content = response.content

    if verbose >= 2:
        print("[judge] --- LLM raw response ---")
        print(response_content)
        print("[judge] --- End of response ---\n")

    # Extract choice from response
    if '```' in response_content:
        parts = response_content.split('```')
        reasoning = parts[0].strip()
        choice = parts[1] if len(parts) > 2 else parts[0]
    else:
        choice = response_content
        reasoning = ""

    if choice.startswith("choice\n"):
        choice = choice[len("choice\n"):].strip()

    choice = choice.strip().upper()

    if choice not in ("ACCEPT", "REJECT"):
        choice = "INVALID CHOICE"

    if verbose >= 1:
        print(f"[judge] → choice: {choice}")
        
    '''
    #save the sql and results to a file
    with open(_PROJECT_ROOT / "judge_output.txt", "w", encoding="utf-8") as f:
        f.write("Extracted Choice:\n")
        f.write(choice + "\n\n")
    '''
    
    result = {
        "Query ID": query_id,
        "Database": database,
        "Judge Model": model,
        "Reasoning": reasoning,
        "choice": choice
    }
    
    return result


def _judge_single_row_job(
    *,
    judge_model: str,
    llm,
    schema_file: str,
    candidate_model: str,
    db_id: str,
    question_id: int,
    question: str,
    evidence: str,
    candidate_sql: str,
    schema_info: Dict[str, Any],
    candidate_metrics: Optional[Dict[str, Any]],
    verbose: int,
) -> Tuple[Dict[str, Any], bool]:
    """Run one binary judgement job and return (result, is_error)."""
    try:
        judge_result = judge(
            database=str(db_id),
            query=question,
            query_id=int(question_id),
            schema_file=schema_file,
            evidence=evidence,
            judge_arguments={
                "model": judge_model,
                "llm": llm,
                "candidate_model": candidate_model,
                "candidate": candidate_sql,
                "schema_info": schema_info,
            },
            verbose=verbose,
        )

        judge_result["question_id"] = int(question_id)
        judge_result["db_id"] = str(db_id)
        judge_result["candidate_model"] = candidate_model
        judge_result["candidate_sql"] = candidate_sql
        judge_result["candidate_metrics"] = candidate_metrics
        return judge_result, False
    except Exception as exc:
        if verbose >= 0:
            print(f"[batch] ERROR judging db={db_id} question_id={question_id}: {exc}")
        error_result = {
            "question_id": int(question_id),
            "db_id": str(db_id),
            "candidate_model": candidate_model,
            "candidate_sql": candidate_sql,
            "choice": "ERROR",
            "Reasoning": str(exc),
            "candidate_metrics": candidate_metrics,
        }
        return error_result, True




def compute_binary_choices_for_sqls(
    judge_model: str = "deepseek-chat",
    llm=None,
    database: str = "bird_developer",
    schema_file: str = None,
    query_file: str = None,
    sqls_dir: str = "candidates",
    output_dir: str = "binary_choices",
    resume: bool = True,
    save_every: int = 10,
    num_threads: int = 1,
    stop_on_error: bool = False,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate all SQL outputs in `sqls_dir` with the binary-choice judge and save one file per candidate model.

    Resume strategy:
    - If `resume=True` and an output file already exists for a candidate model, previously judged
      (db_id, question_id) pairs are skipped.
    - Progress is checkpointed every `save_every` items using atomic writes, so interruption only
      loses at most a small tail of unsaved items.
    """
    default_files = {
        "bird_developer": ("datasets_files/BIRD/dev_tables.json", "datasets_files/BIRD/dev.json"),
    }
    if not schema_file or not query_file:
        alias = (database or "").strip().lower()
        if alias in default_files:
            default_schema, default_query = default_files[alias]
            schema_file = schema_file or default_schema
            query_file = query_file or default_query
            if verbose >= 1:
                print(f"[dataset] Resolved files for '{alias}': schema_file='{schema_file}', query_file='{query_file}'")
        else:
            raise ValueError(
                "schema_file and query_file are required unless database is a known alias like 'bird_developer'."
            )

    if llm is None:
        raise ValueError("An instantiated llm must be provided, or initialize it before calling this function.")
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")

    sqls_path = _PROJECT_ROOT / sqls_dir
    output_path = _PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose >= 1:
        print(f"[batch] Current project root: {_PROJECT_ROOT}")
        print(f"[batch] Looking for SQL files in: {sqls_path}")

    sql_files = sorted(sqls_path.glob("evaluation_sql_metrics_*_vs_ground_truth.json"))

    if verbose >= 0:
        print(f"[batch] Starting binary choice evaluation — judge={judge_model}, database={database}")
        print(f"[batch] Found {len(sql_files)} model file(s) in '{sqls_dir}'")
        print(f"[batch] Using num_threads={num_threads}")
    if verbose >= 2:
        print(f"[batch] schema_file={schema_file} | query_file={query_file}")
        print(f"[batch] output_dir={output_path}")

    summary: Dict[str, Any] = {
        "judge_model": judge_model,
        "database": database,
        "schema_file": schema_file,
        "query_file": query_file,
        "num_threads": num_threads,
        "processed_models": {},
    }

    # Cache question lookups per database to avoid re-reading query files repeatedly.
    per_db_question_index: Dict[str, Dict[int, Dict[str, Any]]] = {}
    # Cache schema per database to avoid reloading schema JSON for every query.
    per_db_schema_info: Dict[str, Dict[str, Any]] = {}

    for sql_file in sql_files:
        candidate_model = sql_file.name
        if candidate_model.startswith("evaluation_sql_metrics_") and candidate_model.endswith("_vs_ground_truth.json"):
            candidate_model = candidate_model[len("evaluation_sql_metrics_") : -len("_vs_ground_truth.json")]
        else:
            candidate_model = sql_file.stem
        output_file = output_path / f"{candidate_model}_binary_choices.json"

        sql_rows = load_json_file(str(sql_file))

        existing_results: List[Dict[str, Any]] = []
        processed_keys = set()

        if resume and output_file.exists():
            existing_payload = load_json_file(str(output_file))
            existing_results = existing_payload.get("results", [])
            for r in existing_results:
                db_id = r.get("db_id")
                qid = r.get("question_id")
                if db_id is not None and qid is not None:
                    processed_keys.add((str(db_id), int(qid)))
            if verbose >= 1:
                print(f"[batch] model={candidate_model} | resuming: skipping {len(processed_keys)} already judged row(s)")
        elif verbose >= 1:
            print(f"[batch] model={candidate_model} | starting fresh ({len(sql_rows)} row(s) to judge)")

        if verbose >= 0:
            print(f"[batch] ── Processing model: {candidate_model} ({len(sql_rows)} total rows) ──")

        results = list(existing_results)
        processed_since_save = 0
        errors_count = 0
        metrics_found = 0
        metrics_missing = 0

        pending_jobs: List[Dict[str, Any]] = []

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
                if verbose >= 2:
                    print(f"[index] Building question index for db_id='{db_key}' from '{query_file}'")
                db_queries = get_queries(db_id=db_key, queries_file=query_file)
                question_index: Dict[int, Dict[str, Any]] = {}
                for query_row in db_queries:
                    qid = query_row.get("question_id")
                    if qid is None:
                        continue
                    question_index[int(qid)] = {
                        "question": query_row.get("question", ""),
                        "evidence": query_row.get("evidence", ""),
                    }
                per_db_question_index[db_key] = question_index
                if verbose >= 2:
                    print(f"[index] Found {len(question_index)} questions for db_id='{db_key}'")
            if db_key not in per_db_schema_info:
                per_db_schema_info[db_key] = get_schema_info(db_key, tables_file=schema_file)
                if per_db_schema_info[db_key] is None:
                    if verbose >= 0:
                        print(f"[batch] ERROR: schema not found for db={db_id} in schema_file={schema_file}")
                    error_result = {
                        "question_id": int(question_id),
                        "db_id": str(db_id),
                        "candidate_model": candidate_model,
                        "candidate_sql": row.get("extracted_sql") or row.get("generated_sql") or "",
                        "choice": "ERROR",
                        "Reasoning": f"Schema not found for db_id '{db_id}' in '{schema_file}'.",
                    }
                    results.append(error_result)
                    processed_keys.add(key)
                    processed_since_save += 1
                    errors_count += 1
                    continue
                if verbose >= 2:
                    print(f"[batch] Cached schema for db={db_id}")

            query_meta = per_db_question_index[db_key].get(int(question_id))
            candidate_metrics = row if isinstance(row, dict) else None
            if candidate_metrics is None:
                metrics_missing += 1
            else:
                metrics_found += 1

            if query_meta is None:
                if verbose >= 0:
                    print(f"[batch] ERROR: no question found for db={db_id} question_id={question_id} — skipping")
                result = {
                    "question_id": int(question_id),
                    "db_id": str(db_id),
                    "candidate_model": candidate_model,
                    "candidate_sql": row.get("extracted_sql") or row.get("generated_sql") or "",
                    "choice": "ERROR",
                    "Reasoning": "No matching question/evidence found in query_file.",
                    "candidate_metrics": candidate_metrics,
                }
                results.append(result)
                processed_keys.add(key)
                processed_since_save += 1
                errors_count += 1
            else:
                candidate_sql = row.get("candidate_sql") or row.get("clean_sql") or row.get("extracted_sql") or row.get("generated_sql") or ""
                if verbose >= 2:
                    print(f"[batch] db={db_id} | question_id={question_id}")
                    print(f"[batch] question : {query_meta['question']}")
                    print(f"[batch] evidence : {query_meta.get('evidence', '')}")
                    print(f"[batch] sql      : {candidate_sql}")
                pending_jobs.append(
                    {
                        "db_id": str(db_id),
                        "question_id": int(question_id),
                        "question": query_meta["question"],
                        "evidence": query_meta.get("evidence", ""),
                        "candidate_sql": candidate_sql,
                        "schema_info": per_db_schema_info[db_key],
                        "candidate_metrics": candidate_metrics,
                    }
                )

            if processed_since_save >= save_every:
                payload = {
                    "judge_model": judge_model,
                    "candidate_model": candidate_model,
                    "schema_file": schema_file,
                    "query_file": query_file,
                    "total_sql_rows": len(sql_rows),
                    "results": results,
                }
                atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
                if verbose >= 1:
                    print(f"[batch] checkpoint saved — {len(results)}/{len(sql_rows)} done for model={candidate_model}")
                processed_since_save = 0

        if pending_jobs and verbose >= 1:
            print(f"[batch] model={candidate_model} | dispatching {len(pending_jobs)} row(s) with {num_threads} thread(s)")

        if pending_jobs:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(
                        _judge_single_row_job,
                        judge_model=judge_model,
                        llm=llm,
                        schema_file=schema_file,
                        candidate_model=candidate_model,
                        db_id=job["db_id"],
                        question_id=job["question_id"],
                        question=job["question"],
                        evidence=job["evidence"],
                        candidate_sql=job["candidate_sql"],
                        schema_info=job["schema_info"],
                        candidate_metrics=job["candidate_metrics"],
                        verbose=verbose,
                    )
                    for job in pending_jobs
                ]

                for future in as_completed(futures):
                    row_result, is_error = future.result()
                    results.append(row_result)
                    processed_keys.add((str(row_result["db_id"]), int(row_result["question_id"])))
                    processed_since_save += 1
                    if is_error:
                        errors_count += 1

                    if stop_on_error and is_error:
                        payload = {
                            "judge_model": judge_model,
                            "candidate_model": candidate_model,
                            "schema_file": schema_file,
                            "query_file": query_file,
                            "total_sql_rows": len(sql_rows),
                            "results": results,
                        }
                        atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
                        raise RuntimeError(
                            f"Stopping on first error for model={candidate_model} "
                            f"db={row_result.get('db_id')} question_id={row_result.get('question_id')}"
                        )

                    if processed_since_save >= save_every:
                        payload = {
                            "judge_model": judge_model,
                            "candidate_model": candidate_model,
                            "schema_file": schema_file,
                            "query_file": query_file,
                            "total_sql_rows": len(sql_rows),
                            "results": results,
                        }
                        atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
                        if verbose >= 1:
                            print(f"[batch] checkpoint saved — {len(results)}/{len(sql_rows)} done for model={candidate_model}")
                        processed_since_save = 0

        payload = {
            "judge_model": judge_model,
            "candidate_model": candidate_model,
            "schema_file": schema_file,
            "query_file": query_file,
            "total_sql_rows": len(sql_rows),
            "results": results,
        }
        atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)

        if verbose >= 0:
            print(f"[batch] ✓ model={candidate_model} complete — {len(results)}/{len(sql_rows)} judged, {errors_count} error(s) → {output_file.name}")
        if verbose >= 1:
            print(
                f"[batch] model={candidate_model} | candidate metrics attached={metrics_found}, "
                f"missing={metrics_missing}"
            )

        summary["processed_models"][candidate_model] = {
            "output_file": str(output_file),
            "total_sql_rows": len(sql_rows),
            "saved_results": len(results),
            "errors": errors_count,
        }

    if verbose >= 0:
        print(f"[batch] Evaluation complete — {len(sql_files)} model(s) processed")

    return summary

if __name__ == "__main__":
    model = "deepseek-chat"
    num_threads = 8
    llm = initialize_llm(model=model, api_key="")
    summary = compute_binary_choices_for_sqls(
        judge_model=model,
        llm=llm,
        num_threads=num_threads,
        verbose=1
    )
    print("Summary of binary choice judgments:")
    print(json.dumps(summary, indent=2))
    
    

    
    
    
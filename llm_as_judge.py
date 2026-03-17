import sys
import json
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langchain.schema.messages import HumanMessage

# Calculate paths and ensure project root is in sys.path
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[0]  # /DEMO-PAPER

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    
from retrieve_database_info import get_schema_info, format_schema_info_for_prompt, get_queries, load_json_file

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

def judge(database, query, query_id, schema_file, evidence, judge_arguments, mode = "binary_choice", verbose: int = 1):

    model = judge_arguments["model"]
    llm = judge_arguments["llm"]

    if verbose >= 1:
        print(f"[judge] query_id={query_id} | db={database} | mode={mode} | judge={model}")

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

    if mode == "binary_choice":
        candidate = judge_arguments["candidate"]
        candidate_model = judge_arguments["candidate_model"]
        llm_prompt = llm_prompt_binary_choice
        prompt = llm_prompt.format(question=query, evidence=evidence, schema=schema_formatted, query=candidate)
        if verbose >= 2:
            print(f"[judge] Question  : {query}")
            print(f"[judge] Evidence  : {evidence}")
            print(f"[judge] Candidate SQL ({candidate_model}): {candidate}")
    elif mode == "candidate_choice":
        candidate_model_1 = judge_arguments["candidate_1_model"]
        candidate_1 = judge_arguments["candidate_1"]
        candidate_model_2 = judge_arguments["candidate_2_model"]
        candidate_2 = judge_arguments["candidate_2"]
        llm_prompt = llm_prompt_candidate_choice
        prompt = llm_prompt.format(query=query, evidence=evidence, schema=schema_formatted, query_1=candidate_1, query_2=candidate_2)
        if verbose >= 2:
            print(f"[judge] Candidate 1 ({candidate_model_1}): {candidate_1}")
            print(f"[judge] Candidate 2 ({candidate_model_2}): {candidate_2}")

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

    choice = choice.strip()

    if verbose >= 1:
        print(f"[judge] → choice: {choice}")
        
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


def _atomic_dump_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to reduce risk of corrupted files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _build_pairwise_index(pairwise_payload: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Index pairwise entries by (db_id, question_id)."""
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for row in pairwise_payload.values():
        if not isinstance(row, dict):
            continue
        db_id = row.get("db_id")
        question_id = row.get("question_id")
        if db_id is None or question_id is None:
            continue
        index[(str(db_id), int(question_id))] = row

    return index


def _extract_ground_truth_comparison(
    pairwise_entry: Optional[Dict[str, Any]],
    candidate_model: str,
) -> Optional[Dict[str, Any]]:
    """Extract the comparison block for ground_truth vs candidate_model, if present."""
    if not pairwise_entry:
        return None

    comparisons = pairwise_entry.get("comparisons", {})
    if not isinstance(comparisons, dict):
        return None

    expected_key = f"ground_truth_vs_{candidate_model}"
    comparison = comparisons.get(expected_key)
    if isinstance(comparison, dict):
        return comparison

    # Fallback if key naming varies: locate by system names.
    for comp in comparisons.values():
        if not isinstance(comp, dict):
            continue
        if comp.get("system1") == "ground_truth" and comp.get("system2") == candidate_model:
            return comp

    return None


def compute_binary_choices_for_sqls(
    judge_model: str = "deepseek-chat",
    llm=None,
    database: str = "bird_developer",
    schema_file: str = None,
    query_file: str = None,
    sqls_dir: str = "sqls",
    output_dir: str = "binary_choices",
    pairwise_file: str = "all_pairwise_comparisons.json",
    resume: bool = True,
    save_every: int = 10,
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
        "bird_developer": ("dev_tables.json", "dev.json"),
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

    sqls_path = _PROJECT_ROOT / sqls_dir
    output_path = _PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    pairwise_path = _PROJECT_ROOT / pairwise_file
    pairwise_payload = load_json_file(str(pairwise_path))
    pairwise_index = _build_pairwise_index(pairwise_payload)
    
    if verbose >= 1:
        print(f"[batch] Current project root: {_PROJECT_ROOT}")
        print(f"[batch] Looking for SQL files in: {sqls_path}")

    sql_files = sorted(sqls_path.glob("*_query_results.json"))

    if verbose >= 0:
        print(f"[batch] Starting binary choice evaluation — judge={judge_model}, database={database}")
        print(f"[batch] Found {len(sql_files)} model file(s) in '{sqls_dir}'")
        print(f"[batch] Loaded pairwise comparison index with {len(pairwise_index)} row(s) from '{pairwise_file}'")
    if verbose >= 2:
        print(f"[batch] schema_file={schema_file} | query_file={query_file}")
        print(f"[batch] output_dir={output_path}")

    summary: Dict[str, Any] = {
        "judge_model": judge_model,
        "database": database,
        "schema_file": schema_file,
        "query_file": query_file,
        "pairwise_file": pairwise_file,
        "processed_models": {},
    }

    # Cache question lookups per database to avoid re-reading query files repeatedly.
    per_db_question_index: Dict[str, Dict[int, Dict[str, Any]]] = {}
    # Cache schema per database to avoid reloading schema JSON for every query.
    per_db_schema_info: Dict[str, Dict[str, Any]] = {}

    for sql_file in sql_files:
        candidate_model = sql_file.name.replace("_query_results.json", "")
        output_file = output_path / f"{candidate_model}_binary_choices.json"

        sql_rows = load_json_file(str(sql_file))

        existing_results: List[Dict[str, Any]] = []
        processed_keys = set()

        if resume and output_file.exists():
            existing_payload = load_json_file(str(output_file))
            existing_results = existing_payload.get("results", [])
            backfilled_count = 0
            for r in existing_results:
                db_id = r.get("db_id")
                qid = r.get("question_id")
                if db_id is None or qid is None:
                    continue
                if "execution_vs_ground_truth" not in r:
                    pairwise_entry = pairwise_index.get((str(db_id), int(qid)))
                    r["execution_vs_ground_truth"] = _extract_ground_truth_comparison(pairwise_entry, candidate_model)
                    backfilled_count += 1
            for r in existing_results:
                db_id = r.get("db_id")
                qid = r.get("question_id")
                if db_id is not None and qid is not None:
                    processed_keys.add((str(db_id), int(qid)))
            if verbose >= 1:
                print(f"[batch] model={candidate_model} | resuming: skipping {len(processed_keys)} already judged row(s)")
                if backfilled_count > 0:
                    print(f"[batch] model={candidate_model} | backfilled execution metadata for {backfilled_count} resumed row(s)")
        elif verbose >= 1:
            print(f"[batch] model={candidate_model} | starting fresh ({len(sql_rows)} row(s) to judge)")

        if verbose >= 0:
            print(f"[batch] ── Processing model: {candidate_model} ({len(sql_rows)} total rows) ──")

        results = list(existing_results)
        processed_since_save = 0
        errors_count = 0
        gt_metadata_found = 0
        gt_metadata_missing = 0

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
            pairwise_entry = pairwise_index.get((db_key, int(question_id)))
            gt_execution = _extract_ground_truth_comparison(pairwise_entry, candidate_model)
            if gt_execution is None:
                gt_metadata_missing += 1
                if verbose >= 2:
                    print(
                        f"[pairwise] Missing ground_truth comparison for model={candidate_model} "
                        f"db={db_id} question_id={question_id}"
                    )
            else:
                gt_metadata_found += 1
                if verbose >= 2:
                    print(
                        f"[pairwise] Found ground_truth comparison for model={candidate_model} "
                        f"db={db_id} question_id={question_id} | "
                        f"performed={gt_execution.get('comparison_performed')}"
                    )

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
                    "execution_vs_ground_truth": gt_execution,
                }
                results.append(result)
                processed_keys.add(key)
                processed_since_save += 1
                errors_count += 1
            else:
                candidate_sql = row.get("extracted_sql") or row.get("generated_sql") or ""
                if verbose >= 2:
                    print(f"[batch] db={db_id} | question_id={question_id}")
                    print(f"[batch] question : {query_meta['question']}")
                    print(f"[batch] evidence : {query_meta.get('evidence', '')}")
                    print(f"[batch] sql      : {candidate_sql}")
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
                            "schema_info": per_db_schema_info[db_key],
                        },
                        mode="binary_choice",
                        verbose=verbose,
                    )

                    judge_result["question_id"] = int(question_id)
                    judge_result["db_id"] = str(db_id)
                    judge_result["candidate_model"] = candidate_model
                    judge_result["candidate_sql"] = candidate_sql
                    judge_result["execution_vs_ground_truth"] = gt_execution
                    results.append(judge_result)
                    processed_keys.add(key)
                    processed_since_save += 1
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
                        "execution_vs_ground_truth": gt_execution,
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
                            "pairwise_file": pairwise_file,
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
                    "pairwise_file": pairwise_file,
                    "total_sql_rows": len(sql_rows),
                    "results": results,
                }
                _atomic_dump_json(output_file, payload)
                if verbose >= 1:
                    print(f"[batch] checkpoint saved — {len(results)}/{len(sql_rows)} done for model={candidate_model}")
                processed_since_save = 0

        payload = {
            "judge_model": judge_model,
            "candidate_model": candidate_model,
            "schema_file": schema_file,
            "query_file": query_file,
            "pairwise_file": pairwise_file,
            "total_sql_rows": len(sql_rows),
            "results": results,
        }
        _atomic_dump_json(output_file, payload)

        if verbose >= 0:
            print(f"[batch] ✓ model={candidate_model} complete — {len(results)}/{len(sql_rows)} judged, {errors_count} error(s) → {output_file.name}")
        if verbose >= 1:
            print(
                f"[pairwise] model={candidate_model} | execution metadata found={gt_metadata_found}, "
                f"missing={gt_metadata_missing}"
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
    llm = initialize_llm(model=model, api_key="sk-2675b255bc084d70b188e7fccd0aed15")
    summary = compute_binary_choices_for_sqls(
        judge_model=model,
        llm=llm,
        verbose=1
    )
    print("Summary of binary choice judgments:")
    print(json.dumps(summary, indent=2))
    
    

    
    
    
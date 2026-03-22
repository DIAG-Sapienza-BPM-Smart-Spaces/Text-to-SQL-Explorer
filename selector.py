import json
import os
import re
import sys
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema.messages import HumanMessage
from common_utils import (
	atomic_dump_json,
	build_pairwise_index,
	extract_ground_truth_comparison,
)

# Calculate paths and ensure project root is in sys.path
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[0]  # /DEMO-PAPER

if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from retrieve_database_info import (  # noqa: E402
	format_schema_info_for_prompt,
	get_queries,
	get_schema_info,
	load_json_file,
)

SELECTOR_MODELS = ["deepseek-chat", "cogito_70b", "qwen2.5-coder_32b", "qwen3-coder_30b"]

_FILE_WRITE_LOCK = threading.Lock()

# Single-model selector prompt.
single_model_selector_prompt = """
You are an expert SQL programmer. Given a natural language query, an evidence, the schema of the database, and four SQL candidates, choose the one that is more correct and answer the natural language query using the evidence and the schema. There can be more than one correct candidate, but you should choose the one that is more correct. You should provide a reasoning for your choice, and then select the candidate that you think is the most correct.
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
Candidate 1:
---
{query_1}
---
Candidate 2:
---
{query_2}
---
Candidate 3:
---
{query_3}
---
Candidate 4:
---
{query_4}
---
Your answer should be in the following format (either CANDIDATE 1, CANDIDATE 2, CANDIDATE 3, or CANDIDATE 4 as CHOICE):
---
<your reasoning>
``choice
<CHOICE>
```
---
Here is an example of an answer:
---
I will sistematically analyze the candidates, undestanding which is the most correct one.
Candidate 1 is correct because it correctly uses the JOIN clause to combine the tables and retrieves the correct columns. It also correctly uses the WHERE clause to filter the results based on the conditions specified in the natural language query.
Candidate 2 is incorrect because it does not use the JOIN clause correctly, and it retrieves incorrect columns. It also does not use the WHERE clause correctly, which leads to incorrect results.
Candidate 3 is partially incorrect because it uses the JOIN clause correctly, but it retrieves incorrect columns. It also does not use the WHERE clause correctly, which leads to incorrect results.
Candidate 4 is also correct because it correctly uses the JOIN clause to combine the tables and retrieves the correct columns. It also correctly uses the WHERE clause to filter the results based on the conditions specified in the natural language query.
Yet, I think that Candidate 1 is more correct than Candidate 4 because it is more efficient and it retrieves the results in a more organized way. Therefore, I will choose Candidate 1 as the most correct one.
```choice
CANDIDATE 1
```
---
"""

#Multi model selector

multi_model_selector_prompt_second_round = """
You are an expert SQL programmer that is working togheter with other experts. Given a natural language query, an evidence, the schema of the database, and four SQL candidates, choose the one that is more correct and answer the natural language query using the evidence and the schema. There can be more than one correct candidate, but you should choose the one that is more correct. You should provide a reasoning for your choice, and then select the candidate that you think is the most correct.
You are provided your past reasoning and choice as well as the choice and reasoning of the other experts, you should take them into consideration when making your choice.
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
Candidate 3:
---
{query_3}
---
Candidate 4:
---
{query_4}
---
Choices and reasoning of the other experts:
---
{choices_and_reasoning}
---
Your answer should be in the following format (either CANDIDATE 1, CANDIDATE 2, CANDIDATE 3, or CANDIDATE 4 as CHOICE). 
It is crucial that you put exactly CANDIDATE 1, CANDIDATE 2, CANDIDATE 3, or CANDIDATE 4 as CHOICE, not sql code or any other text.
---
<your reasoning>
``choice
<CHOICE>
```
---
Example of an answer:
---
I will sistematically analyze the candidates, undestanding which is the most correct one.
Candidate 1 is correct because it correctly uses the JOIN clause to combine the tables and retrieves the correct columns. It also correctly uses the WHERE clause to filter the results based on the conditions specified in the natural language query. Expert 1 chose Candidate 1 and providing a fair reasoning.
Candidate 2 is incorrect because it does not use the JOIN clause correctly, and it retrieves incorrect columns. It also does not use the WHERE clause correctly, which leads to incorrect results.
Candidate 3 is partially incorrect because it uses the JOIN clause correctly, but it retrieves incorrect columns. Expert 2 and 3 chose Candidate 3 stating that the columns are renamed but actually correct. Their reasoning is sound and is worth considering, and point out that Candidate 1 and 4 could be incosistent.
Candidate 4 is also correct because it correctly uses the JOIN clause to combine the tables and retrieves the correct columns. It also correctly uses the WHERE clause to filter the results based on the conditions specified in the natural language query. My initial choice was Candidate 1.
Following the reasoning of Expert 2 and 3, I will reconsider Candidate 3 due having an actual correct approach and I will change my initial choice. Candidate 3 is actually correct and it is more efficient than Candidate 1 and 4. Therefore, I will choose Candidate 3 as the most correct one.
```choice
CANDIDATE 3
```
---
"""


def initialize_llm(model: str, api_key: str, verbose: int = 1):
	if verbose >= 1:
		print(f"[init] Initializing selector LLM: {model}")
	if model == "deepseek-chat":
		from langchain_deepseek.chat_models import ChatDeepSeek

		llm = ChatDeepSeek(api_key=api_key, model=model)
		if verbose >= 2:
			print(f"[init] ChatDeepSeek instance created for model '{model}'")
		return llm

	raise ValueError(
		f"Unsupported selector model '{model}'. "
		"At the moment initialize_llm supports only 'deepseek-chat'. "
		"You can still pass a custom pre-initialized llm object to compute_single_selector_choices()."
	)


def _extract_reasoning_and_choice(response_content: str) -> Tuple[str, str]:
	"""Extract reasoning text and raw choice text from LLM response."""
	if "```" in response_content:
		parts = response_content.split("```")
		reasoning = parts[0].strip()
		choice = parts[1] if len(parts) > 1 else ""
	else:
		reasoning = ""
		choice = response_content

	choice = choice.strip()
	if choice.lower().startswith("choice\n"):
		choice = choice.split("\n", 1)[1].strip()

	return reasoning, choice.strip()


def _normalize_candidate_choice(choice_raw: str) -> Optional[int]:
	"""Return candidate index 1..4 from a free-form LLM choice, if possible."""
	upper = choice_raw.upper().strip()

	match = re.search(r"CANDIDATE\s*([1-4])", upper)
	if match:
		return int(match.group(1))

	only_digit = re.fullmatch(r"\s*([1-4])\s*", upper)
	if only_digit:
		return int(only_digit.group(1))

	return None


def _normalize_sql_for_match(sql_text: str) -> str:
	"""Normalize SQL text for robust textual matching across formatting variants."""
	cleaned = (sql_text or "").strip()
	cleaned = re.sub(r"```(?:sql)?", "", cleaned, flags=re.IGNORECASE)
	cleaned = cleaned.replace("```", " ")
	cleaned = re.sub(r"\s+", " ", cleaned)
	return cleaned.strip().rstrip(";").strip().lower()


def _match_candidate_from_sql_response(
	response_content: str,
	ordered_candidates: List[Tuple[str, str]],
) -> Optional[int]:
	"""Try to infer the chosen candidate by matching SQL present in the model answer."""
	fenced_blocks = re.findall(r"```(?:sql)?\s*(.*?)```", response_content, flags=re.IGNORECASE | re.DOTALL)
	search_spaces = [response_content] + fenced_blocks
	normalized_spaces = [_normalize_sql_for_match(chunk) for chunk in search_spaces if chunk and chunk.strip()]

	for idx, (_, candidate_sql) in enumerate(ordered_candidates, start=1):
		norm_candidate = _normalize_sql_for_match(candidate_sql)
		if not norm_candidate:
			continue
		for norm_space in normalized_spaces:
			if norm_candidate == norm_space or norm_candidate in norm_space:
				return idx

	return None


def _load_candidate_sqls(
	sqls_dir: str,
	verbose: int = 1,
) -> Tuple[List[str], Dict[Tuple[str, int], Dict[str, str]]]:
	"""
	Load all *_query_results.json files and build:
	- ordered candidate model list
	- index keyed by (db_id, question_id) -> {model_name: sql}
	"""
	sqls_path = _PROJECT_ROOT / sqls_dir
	sql_files = sorted(sqls_path.glob("*_query_results.json"))

	if not sql_files:
		raise FileNotFoundError(f"No *_query_results.json files found in '{sqls_path}'.")

	candidate_models = [p.name.replace("_query_results.json", "") for p in sql_files]
	candidate_index: Dict[Tuple[str, int], Dict[str, str]] = {}

	if verbose >= 1:
		print(f"[candidates] Found {len(sql_files)} candidate model file(s) in '{sqls_dir}'")

	for sql_file in sql_files:
		model_name = sql_file.name.replace("_query_results.json", "")
		rows = load_json_file(str(sql_file))

		for row in rows:
			db_id = row.get("db_id")
			question_id = row.get("question_id")
			if db_id is None or question_id is None:
				continue

			key = (str(db_id), int(question_id))
			sql = row.get("extracted_sql") or row.get("generated_sql") or ""

			if key not in candidate_index:
				candidate_index[key] = {}
			candidate_index[key][model_name] = sql

	return candidate_models, candidate_index


def select_candidate_for_query(
	llm,
	selector_model: str,
	question: str,
	evidence: str,
	schema_formatted: Dict[str, Any],
	ordered_candidates: List[Tuple[str, str]],
	verbose: int = 1,
) -> Dict[str, Any]:
	"""Run the single-model selector prompt for one query and parse the choice."""
	if len(ordered_candidates) != 4:
		raise ValueError(
			"single_model_selector_prompt expects exactly 4 candidates, "
			f"but got {len(ordered_candidates)}"
		)

	prompt = single_model_selector_prompt.format(
		question=question,
		evidence=evidence,
		schema=schema_formatted,
		query_1=ordered_candidates[0][1],
		query_2=ordered_candidates[1][1],
		query_3=ordered_candidates[2][1],
		query_4=ordered_candidates[3][1],
	)

	if verbose >= 2:
		print("[selector] --- Prompt sent to LLM ---")
		print(prompt)
		print("[selector] --- End of prompt ---\n")

	response = llm.invoke([HumanMessage(content=prompt)])
	response_content = response.content if hasattr(response, "content") else str(response)
	original_answer = (response_content or "").strip()

	if verbose >= 2:
		print("[selector] --- LLM raw response ---")
		print(response_content)
		print("[selector] --- End of response ---\n")

	reasoning, choice_raw = _extract_reasoning_and_choice(response_content)
	selected_index = _normalize_candidate_choice(choice_raw)

	selected_model = None
	selected_sql = None
	normalized_choice = "UNPARSEABLE"
	used_sql_fallback = False

	if selected_index is None:
		matched_index = _match_candidate_from_sql_response(response_content, ordered_candidates)
		if matched_index is not None:
			selected_index = matched_index
			used_sql_fallback = True

	if selected_index is not None:
		normalized_choice = f"CANDIDATE {selected_index}"
		selected_model, selected_sql = ordered_candidates[selected_index - 1]

	if verbose >= 1:
		print(
			f"[selector] selector={selector_model} | raw_choice='{choice_raw}' "
			f"| normalized='{normalized_choice}' | sql_fallback={used_sql_fallback}"
		)

	return {
		"Reasoning": reasoning,
		"choice_raw": choice_raw,
		"original_answer": original_answer,
		"choice": normalized_choice,
		"selected_candidate_index": selected_index,
		"selected_candidate_model": selected_model,
		"selected_candidate_sql": selected_sql,
	}


def _select_single_row_job(
	*,
	llm,
	selector_model: str,
	db_id: str,
	question_id: int,
	question: str,
	evidence: str,
	ordered_candidates: List[Tuple[str, str]],
	schema_info: Dict[str, Any],
	pairwise_entry: Optional[Dict[str, Any]],
	verbose: int,
) -> Tuple[Dict[str, Any], bool]:
	"""Run one selector job and return (result, is_error)."""
	try:
		schema_formatted = format_schema_info_for_prompt(schema_info)
		selection_result = select_candidate_for_query(
			llm=llm,
			selector_model=selector_model,
			question=question,
			evidence=evidence,
			schema_formatted=schema_formatted,
			ordered_candidates=ordered_candidates,
			verbose=verbose,
		)

		selected_model = selection_result["selected_candidate_model"]
		selected_vs_gt = (
			extract_ground_truth_comparison(pairwise_entry, selected_model)
			if selected_model
			else None
		)

		selected_metrics = None
		if isinstance(selected_vs_gt, dict):
			selected_metrics = {
				"schema_precision": selected_vs_gt.get("schema_precision"),
				"schema_recall": selected_vs_gt.get("schema_recall"),
				"cell_value_accuracy": selected_vs_gt.get("cell_value_accuracy"),
				"row_set_jaccard": selected_vs_gt.get("row_set_jaccard"),
				"execution_accuracy": selected_vs_gt.get("execution_accuracy"),
				"f1_score": selected_vs_gt.get("f1_score"),
				"comparison_performed": selected_vs_gt.get("comparison_performed"),
			}

		result = {
			"question_id": int(question_id),
			"db_id": str(db_id),
			"selector_model": selector_model,
			"question": question,
			"evidence": evidence,
			"choice": selection_result["choice"],
			"choice_raw": selection_result["choice_raw"],
			"original_answer": selection_result.get("original_answer"),
			"Reasoning": selection_result["Reasoning"],
			"selected_candidate_index": selection_result["selected_candidate_index"],
			"selected_candidate_model": selected_model,
			"selected_candidate_sql": selection_result["selected_candidate_sql"],
			"candidate_models": [m for m, _ in ordered_candidates],
			"candidate_sqls": [
				{
					"candidate": f"CANDIDATE {idx + 1}",
					"model": model_name,
					"sql": sql,
				}
				for idx, (model_name, sql) in enumerate(ordered_candidates)
			],
			"selected_vs_ground_truth": selected_vs_gt,
			"selected_metrics_vs_ground_truth": selected_metrics,
		}
		return result, False
	except Exception as exc:
		if verbose >= 0:
			print(f"[batch] ERROR db={db_id} question_id={question_id}: {exc}")
		result = {
			"question_id": int(question_id),
			"db_id": str(db_id),
			"selector_model": selector_model,
			"question": question,
			"evidence": evidence,
			"choice": "ERROR",
			"choice_raw": "ERROR",
			"original_answer": None,
			"Reasoning": str(exc),
			"selected_candidate_index": None,
			"selected_candidate_model": None,
			"selected_candidate_sql": None,
			"candidate_models": [m for m, _ in ordered_candidates],
			"candidate_sqls": [
				{
					"candidate": f"CANDIDATE {idx + 1}",
					"model": model_name,
					"sql": sql,
				}
				for idx, (model_name, sql) in enumerate(ordered_candidates)
			],
			"selected_vs_ground_truth": None,
			"selected_metrics_vs_ground_truth": None,
		}
		return result, True


def compute_single_selector_choices(
	selector_model: str = "deepseek-chat",
	llm=None,
	database: str = "bird_developer",
	schema_file: str = None,
	query_file: str = None,
	sqls_dir: str = "sqls",
	pairwise_file: str = "all_pairwise_comparisons.json",
	output_dir: str = "selectors",
	output_filename: Optional[str] = None,
	resume: bool = True,
	save_every: int = 10,
	num_threads: int = 1,
	stop_on_error: bool = False,
	verbose: int = 1,
) -> Dict[str, Any]:
	"""
	Run single-model selection over all queries and choose one of four SQL candidates.
	Save one JSON file for the selector model.
	Also enrich each selected choice with ground-truth comparison metrics from pairwise_file.
	"""
	if llm is None:
		raise ValueError("An instantiated llm must be provided.")
	if num_threads < 1:
		raise ValueError("num_threads must be >= 1")

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

	candidate_models, candidate_index = _load_candidate_sqls(sqls_dir=sqls_dir, verbose=verbose)
	if len(candidate_models) != 4:
		raise ValueError(
			f"This selector currently supports exactly 4 candidate models. Found: {len(candidate_models)}"
		)

	pairwise_payload = load_json_file(str(_PROJECT_ROOT / pairwise_file))
	pairwise_index = build_pairwise_index(pairwise_payload)

	query_rows = load_json_file(str(_PROJECT_ROOT / query_file))
	query_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
	for row in query_rows:
		db_id = row.get("db_id")
		qid = row.get("question_id")
		if db_id is None or qid is None:
			continue
		query_index[(str(db_id), int(qid))] = {
			"question": row.get("question", ""),
			"evidence": row.get("evidence", ""),
		}

	output_path = _PROJECT_ROOT / output_dir
	output_path.mkdir(parents=True, exist_ok=True)
	if output_filename is None:
		output_filename = f"{selector_model}_single_selector_choices.json"
	output_file = output_path / output_filename

	results: List[Dict[str, Any]] = []
	processed_keys = set()
	if resume and output_file.exists():
		existing_payload = load_json_file(str(output_file))
		existing_results = existing_payload.get("results", [])
		for r in existing_results:
			db_id = r.get("db_id")
			qid = r.get("question_id")
			if db_id is None or qid is None:
				continue
			results.append(r)
			processed_keys.add((str(db_id), int(qid)))
		if verbose >= 1:
			print(f"[resume] Loaded {len(processed_keys)} already processed row(s) from '{output_file.name}'")

	per_db_schema_info: Dict[str, Dict[str, Any]] = {}

	total = len(candidate_index)
	errors_count = 0
	processed_since_save = 0

	if verbose >= 0:
		print(f"[batch] Starting single-model selection with selector={selector_model}")
		print(f"[batch] Candidate models: {candidate_models}")
		print(f"[batch] Total query keys from SQL outputs: {total}")
		print(f"[batch] Using num_threads={num_threads}")

	pending_jobs: List[Dict[str, Any]] = []

	for key in sorted(candidate_index.keys()):
		db_id, question_id = key
		if key in processed_keys:
			continue

		query_meta = query_index.get(key)
		candidate_sqls_by_model = candidate_index.get(key, {})

		ordered_candidates: List[Tuple[str, str]] = []
		missing_models = []
		for model_name in candidate_models:
			sql = candidate_sqls_by_model.get(model_name)
			if sql is None:
				missing_models.append(model_name)
				sql = ""
			ordered_candidates.append((model_name, sql))

		if db_id not in per_db_schema_info:
			per_db_schema_info[db_id] = get_schema_info(db_id, tables_file=schema_file)

		schema_info = per_db_schema_info[db_id]

		if query_meta is None or schema_info is None or missing_models:
			reason_parts = []
			if query_meta is None:
				reason_parts.append("No matching question/evidence found in query_file.")
			if schema_info is None:
				reason_parts.append(f"Schema not found for db_id '{db_id}' in '{schema_file}'.")
			if missing_models:
				reason_parts.append(f"Missing candidate SQL for model(s): {missing_models}")

			selected_model = None
			selected_vs_gt = None
			result = {
				"question_id": int(question_id),
				"db_id": str(db_id),
				"selector_model": selector_model,
				"question": query_meta.get("question", "") if query_meta else "",
				"evidence": query_meta.get("evidence", "") if query_meta else "",
				"choice": "ERROR",
				"choice_raw": "ERROR",
				"original_answer": None,
				"Reasoning": " ".join(reason_parts),
				"selected_candidate_index": None,
				"selected_candidate_model": selected_model,
				"selected_candidate_sql": None,
				"candidate_models": candidate_models,
				"candidate_sqls": [
					{
						"candidate": f"CANDIDATE {idx + 1}",
						"model": model_name,
						"sql": sql,
					}
					for idx, (model_name, sql) in enumerate(ordered_candidates)
				],
				"selected_vs_ground_truth": selected_vs_gt,
				"selected_metrics_vs_ground_truth": None,
			}
			results.append(result)
			processed_keys.add(key)
			processed_since_save += 1
			errors_count += 1
		else:
			if verbose >= 1:
				print(f"[batch] db={db_id} | question_id={question_id}")

			pending_jobs.append(
				{
					"db_id": str(db_id),
					"question_id": int(question_id),
					"question": query_meta["question"],
					"evidence": query_meta.get("evidence", ""),
					"ordered_candidates": ordered_candidates,
					"schema_info": schema_info,
					"pairwise_entry": pairwise_index.get(key),
				}
			)

		if processed_since_save >= save_every:
			payload = {
				"selector_model": selector_model,
				"mode": "single_selector",
				"database": database,
				"schema_file": schema_file,
				"query_file": query_file,
				"pairwise_file": pairwise_file,
				"candidate_models": candidate_models,
				"total_queries": total,
				"results": results,
			}
			atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
			if verbose >= 1:
				print(f"[batch] checkpoint saved — {len(results)}/{total} processed")
			processed_since_save = 0

	if pending_jobs and verbose >= 1:
		print(f"[batch] dispatching {len(pending_jobs)} row(s) with {num_threads} thread(s)")

	if pending_jobs:
		with ThreadPoolExecutor(max_workers=num_threads) as executor:
			futures = [
				executor.submit(
					_select_single_row_job,
					llm=llm,
					selector_model=selector_model,
					db_id=job["db_id"],
					question_id=job["question_id"],
					question=job["question"],
					evidence=job["evidence"],
					ordered_candidates=job["ordered_candidates"],
					schema_info=job["schema_info"],
					pairwise_entry=job["pairwise_entry"],
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
						"selector_model": selector_model,
						"mode": "single_selector",
						"database": database,
						"schema_file": schema_file,
						"query_file": query_file,
						"pairwise_file": pairwise_file,
						"candidate_models": candidate_models,
						"total_queries": total,
						"results": results,
					}
					atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
					raise RuntimeError(
						f"Stopping on first error for db={row_result.get('db_id')} "
						f"question_id={row_result.get('question_id')}"
					)

				if processed_since_save >= save_every:
					payload = {
						"selector_model": selector_model,
						"mode": "single_selector",
						"database": database,
						"schema_file": schema_file,
						"query_file": query_file,
						"pairwise_file": pairwise_file,
						"candidate_models": candidate_models,
						"total_queries": total,
						"results": results,
					}
					atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
					if verbose >= 1:
						print(f"[batch] checkpoint saved — {len(results)}/{total} processed")
					processed_since_save = 0

	payload = {
		"selector_model": selector_model,
		"mode": "single_selector",
		"database": database,
		"schema_file": schema_file,
		"query_file": query_file,
		"pairwise_file": pairwise_file,
		"candidate_models": candidate_models,
		"total_queries": total,
		"results": results,
	}
	atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)

	if verbose >= 0:
		print(f"[batch] Done — {len(results)}/{total} processed, errors={errors_count}")
		print(f"[batch] Output: {output_file}")

	return {
		"selector_model": selector_model,
		"output_file": str(output_file),
		"candidate_models": candidate_models,
		"total_queries": total,
		"saved_results": len(results),
		"num_threads": num_threads,
		"errors": errors_count,
	}


def _resolve_api_key_for_model(selector_model: str) -> str:
	"""Resolve API key from environment variables for a selector model."""
	if selector_model == "deepseek-chat":
		return os.getenv("DEEPSEEK_API_KEY", "sk-2675b255bc084d70b188e7fccd0aed15")

	# Optional generic fallback if additional providers are wired in initialize_llm.
	return os.getenv("SELECTOR_API_KEY", "")


def run_selector_batch_for_dataset(
	selector_models: List[str],
	dataset: str = "bird_developer",
	num_threads: int = 3,
	verbose: int = 1,
) -> Dict[str, Any]:
	"""Run selector inference for each selector model on the selected dataset."""
	batch_summary: Dict[str, Any] = {
		"dataset": dataset,
		"runs": {},
		"skipped": {},
	}

	for selector_model in selector_models:
		api_key = _resolve_api_key_for_model(selector_model)
		if not api_key:
			message = (
				f"Missing API key for selector model '{selector_model}'. "
				"Set the required environment variable and retry."
			)
			batch_summary["skipped"][selector_model] = message
			if verbose >= 0:
				print(f"[main] SKIP {selector_model}: {message}")
			continue

		try:
			llm = initialize_llm(model=selector_model, api_key=api_key, verbose=verbose)
		except Exception as exc:
			batch_summary["skipped"][selector_model] = str(exc)
			if verbose >= 0:
				print(f"[main] SKIP {selector_model}: {exc}")
			continue

		if verbose >= 0:
			print(
				f"[main] Running four-candidate selector for model='{selector_model}' "
				f"on dataset='{dataset}'"
			)

		summary = compute_single_selector_choices(
			selector_model=selector_model,
			llm=llm,
			database=dataset,
			num_threads=num_threads,
			verbose=verbose,
		)
		batch_summary["runs"][selector_model] = summary

	return batch_summary


if __name__ == "__main__":

	summary = run_selector_batch_for_dataset(
		selector_models=["deepseek-chat"],
		dataset="bird_developer",
		num_threads=7,
		verbose=1,
	)

	print("Summary of selector runs (four-candidate selection):")
	print(json.dumps(summary, indent=2))
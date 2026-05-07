import json
import os
import random
import re
import sys
import threading
import argparse
import concurrent.futures
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.messages import HumanMessage
from common_utils import (
	atomic_dump_json,
)
from sql_cleaner import clean_sql_query

# Calculate paths and ensure project root is in sys.path
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[0]  # /DEMO-PAPER

if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from retrieve_database_info import (  # noqa: E402
	format_schema_info_for_prompt,
	get_schema_info,
	load_json_file,
)

_FILE_WRITE_LOCK = threading.Lock()
_THREAD_LOCAL = threading.local()


pairwise_selector_prompt = """
You are an expert SQL programmer.
Given a natural language query, evidence, database schema, and exactly two SQL candidates, choose which candidate is better.

Return your decision as exactly one of: CANDIDATE A or CANDIDATE B.

Natural language query:
{question}

Evidence:
{evidence}

Database schema:
{schema}

Candidate A:
{query_a}

Candidate B:
{query_b}

Answer format:
<reasoning>
```choice
<CANDIDATE A|CANDIDATE B>
```
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
		"At the moment initialize_llm supports only 'deepseek-chat'."
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


def _canonical_sql_for_match(sql_text: str) -> str:
	"""Build a lightweight canonical SQL string for equality matching."""
	cleaned = clean_sql_query(sql_text)
	cleaned = cleaned.strip()

	# Some responses begin with a language hint (e.g. "sql\nSELECT ...").
	if "\n" in cleaned:
		first_line, rest = cleaned.split("\n", 1)
		if first_line.strip().lower() in {"sql", "query"}:
			cleaned = rest.strip()

	return cleaned.rstrip(";").strip().lower()


def _normalize_pairwise_choice(
	choice_raw: str,
	candidate_sql_a: str = "",
	candidate_sql_b: str = "",
) -> str:
	"""Normalize pairwise selector output to CANDIDATE A/B/TIE or UNPARSEABLE."""
	upper = (choice_raw or "").upper().strip()
	if re.search(r"CANDIDATE\s*A", upper) or re.fullmatch(r"\s*A\s*", upper):
		return "CANDIDATE A"
	if re.search(r"CANDIDATE\s*B", upper) or re.fullmatch(r"\s*B\s*", upper):
		return "CANDIDATE B"

	# Fallback: if the model pasted SQL instead of A/B, try exact canonical match.
	choice_sql = _canonical_sql_for_match(choice_raw)
	if not choice_sql:
		return "UNPARSEABLE"

	a_sql = _canonical_sql_for_match(candidate_sql_a)
	b_sql = _canonical_sql_for_match(candidate_sql_b)

	if choice_sql == a_sql and choice_sql != b_sql:
		return "CANDIDATE A"
	if choice_sql == b_sql and choice_sql != a_sql:
		return "CANDIDATE B"
	if choice_sql == a_sql and choice_sql == b_sql:
		return random.choice(["CANDIDATE A", "CANDIDATE B"])
	return "UNPARSEABLE"



def _load_candidate_sqls(
	sqls_dir: str,
	verbose: int = 1,
) -> Tuple[List[str], Dict[Tuple[str, int], Dict[str, str]], Dict[Tuple[str, int], Dict[str, Dict[str, Any]]]]:
	"""
	Load candidate metric/result files and build:
	- ordered candidate model list
	- index keyed by (db_id, question_id) -> {model_name: sql}
	- index keyed by (db_id, question_id) -> {model_name: source row payload}
	"""
	sqls_path = _PROJECT_ROOT / sqls_dir
	sql_files = sorted(sqls_path.glob("evaluation_sql_metrics_*_vs_ground_truth.json"))

	if not sql_files:
		raise FileNotFoundError(
			f"No candidate metric/result JSON files found in '{sqls_path}'. "
			"Expected files like 'evaluation_sql_metrics_*_vs_ground_truth.json'."
		)

	candidate_model_set: set[str] = set()
	candidate_index: Dict[Tuple[str, int], Dict[str, str]] = {}
	candidate_row_index: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}

	if verbose >= 1:
		print(f"[candidates] Found {len(sql_files)} candidate model file(s) in '{sqls_dir}'")

	for sql_file in sql_files:
		model_name = sql_file.name
		if model_name.startswith("evaluation_sql_metrics_") and model_name.endswith("_vs_ground_truth.json"):
			model_name = model_name[len("evaluation_sql_metrics_") : -len("_vs_ground_truth.json")]
		else:
			model_name = Path(model_name).stem

		rows = load_json_file(str(sql_file))
		if verbose >= 2:
			print(f"[candidates] Loaded {len(rows)} row(s) from {sql_file.name}")

		for row in rows:
			db_id = row.get("db_id")
			question_id = row.get("question_id")
			if db_id is None or question_id is None:
				continue

			key = (str(db_id), int(question_id))
			row_model = str(row.get("model") or model_name)
			sql = row.get("candidate_sql") or row.get("clean_sql") or row.get("extracted_sql") or row.get("generated_sql") or ""

			if key not in candidate_index:
				candidate_index[key] = {}
			if key not in candidate_row_index:
				candidate_row_index[key] = {}

			candidate_index[key][row_model] = sql
			candidate_row_index[key][row_model] = row
			candidate_model_set.add(row_model)

	if verbose >= 1:
		print(f"[candidates] Built candidate index for {len(candidate_index)} query key(s)")
		print(f"[candidates] Discovered {len(candidate_model_set)} candidate model(s)")

	return sorted(candidate_model_set), candidate_index, candidate_row_index


def _resolve_api_key_for_model(selector_model: str) -> str:
	"""Resolve API key from environment variables for a selector model."""
	if selector_model == "deepseek-chat":
		return os.getenv("DEEPSEEK_API_KEY", "")

	# Optional generic fallback if additional providers are wired in initialize_llm.
	return os.getenv("SELECTOR_API_KEY", "")


def _get_worker_llm(*, selector_model: str, api_key: str, fallback_llm: Any = None):
	"""Return a per-thread LLM client to avoid shared-client contention."""
	llm = getattr(_THREAD_LOCAL, "llm", None)
	if llm is not None:
		return llm

	if api_key:
		llm = initialize_llm(model=selector_model, api_key=api_key, verbose=0)
	elif fallback_llm is not None:
		llm = fallback_llm
	else:
		raise RuntimeError("No worker LLM available: provide either api_key or llm.")

	_THREAD_LOCAL.llm = llm
	return llm


def _derive_pairwise_leaderboard(
	candidate_models: List[str],
	pairwise_judgments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	stats = {
		model: {"wins": 0, "losses": 0, "ties": 0, "played": 0}
		for model in candidate_models
	}

	for row in pairwise_judgments:
		model_a = row.get("model_a")
		model_b = row.get("model_b")
		winner = row.get("winner")
		if model_a not in stats or model_b not in stats:
			continue

		stats[model_a]["played"] += 1
		stats[model_b]["played"] += 1

		if winner == model_a:
			stats[model_a]["wins"] += 1
			stats[model_b]["losses"] += 1
		elif winner == model_b:
			stats[model_b]["wins"] += 1
			stats[model_a]["losses"] += 1
		else:
			stats[model_a]["ties"] += 1
			stats[model_b]["ties"] += 1

	leaderboard = []
	for model in candidate_models:
		played = max(1, stats[model]["played"])
		wins = stats[model]["wins"]
		losses = stats[model]["losses"]
		ties = stats[model]["ties"]
		leaderboard.append(
			{
				"model": model,
				"wins": wins,
				"losses": losses,
				"ties": ties,
				"played": stats[model]["played"],
				"win_rate": wins / played,
			}
		)

	leaderboard.sort(key=lambda x: (x["wins"], x["win_rate"], -x["losses"], x["model"]), reverse=True)
	return leaderboard


def _save_pairwise_payload(
	*,
	output_file: Path,
	selector_model: str,
	database: str,
	schema_file: str,
	query_file: str,
	candidate_models: List[str],
	total_queries: int,
	results: List[Dict[str, Any]],
) -> None:
	"""Persist selector payload through atomic, lock-protected write."""
	payload = {
		"selector_model": selector_model,
		"mode": "pairwise_selector",
		"database": database,
		"schema_file": schema_file,
		"query_file": query_file,
		"candidate_models": candidate_models,
		"total_queries": total_queries,
		"results": results,
	}
	atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)


def _build_pairwise_result_for_key(
	*,
	key: Tuple[str, int],
	candidate_models: List[str],
	candidate_rows_for_key: Dict[str, Dict[str, Any]],
	query_meta: Dict[str, Any],
	schema_formatted: str,
	candidates_for_key: Dict[str, str],
	selector_model: str,
	api_key: str,
	llm: Any,
	verbose: int,
) -> Dict[str, Any]:
	"""Build pairwise judgments and leaderboard for a single (db_id, question_id)."""

	if verbose >= 1:
		print(f"[pairwise] Processing db_id={key[0]}, qid={key[1]}")

	db_id, question_id = key
	pairwise_rows: List[Dict[str, Any]] = []

	for model_a, model_b in combinations(candidate_models, 2):

		if verbose >= 1:
				print(f"[pairwise] Calculating pairwise judgment for db_id={db_id}, qid={question_id}, {model_a} vs {model_b}")

		sql_a = candidates_for_key.get(model_a, "")
		sql_b = candidates_for_key.get(model_b, "")
		prompt = pairwise_selector_prompt.format(
			question=query_meta["question"],
			evidence=query_meta.get("evidence", ""),
			schema=schema_formatted,
			query_a=sql_a,
			query_b=sql_b,
		)

		if verbose >= 2:
			print(f"[pairwise] db_id={db_id}, qid={question_id}, comparing {model_a} vs {model_b}")
			print(f"[pairwise] Prompt:\n{prompt}\n---")

		worker_llm = _get_worker_llm(selector_model=selector_model, api_key=api_key, fallback_llm=llm)
		response = worker_llm.invoke([HumanMessage(content=prompt)])

		if verbose >= 2:
			print(f"[pairwise] Raw response for db_id={db_id}, qid={question_id}, {model_a} vs {model_b}")
			print(str(response))
			print("---")

		response_content = response.content if hasattr(response, "content") else str(response)
		reasoning, choice_raw = _extract_reasoning_and_choice(str(response_content))
		normalized = _normalize_pairwise_choice(
			choice_raw,
			candidate_sql_a=sql_a,
			candidate_sql_b=sql_b,
		)

		if normalized == "CANDIDATE A":
			winner = model_a
		elif normalized == "CANDIDATE B":
			winner = model_b
		elif normalized == "UNPARSEABLE":
			winner = "UNPARSEABLE"

		if verbose >= 2:
			print(f"[pairwise] Normalized choice for db_id={db_id}, qid={question_id}, {model_a} vs {model_b}: '{normalized}' (raw: '{choice_raw}') -> Winner: '{winner}'")

		pairwise_rows.append(
			{
				"model_a": model_a,
				"model_b": model_b,
				"candidate_sql_a": sql_a,
				"candidate_sql_b": sql_b,
				"choice_raw": choice_raw,
				"choice": normalized,
				"winner": winner,
				"Reasoning": reasoning,
				"original_answer": str(response_content).strip(),
				"candidate_metrics_model_a": candidate_rows_for_key.get(model_a, {}),
				"candidate_metrics_model_b": candidate_rows_for_key.get(model_b, {}),
			}
		)
	
	if verbose >= 2:
		print(f"[pairwise] Completed pairwise judgments for db_id={db_id}, qid={question_id}")

	leaderboard = _derive_pairwise_leaderboard(candidate_models, pairwise_rows)
	selected_candidate_model = leaderboard[0]["model"] if leaderboard else None

	if verbose >= 2:
		print(f"[pairwise] Finished db_id={db_id}, qid={question_id}, selected={selected_candidate_model}")

	return {
		"question_id": int(question_id),
		"db_id": str(db_id),
		"selector_model": None,
		"question": query_meta.get("question", ""),
		"evidence": query_meta.get("evidence", ""),
		"candidate_models": candidate_models,
		"pairwise_judgments": pairwise_rows,
		"leaderboard": leaderboard,
		"selected_candidate_model": selected_candidate_model,
	}


def compute_pairwise_selector_choices(
	selector_model: str = "deepseek-chat",
	llm=None,
	api_key: str = "",
	database: str = "bird_developer",
	schema_file: str = None,
	query_file: str = None,
	sqls_dir: str = "candidates",
	output_dir: str = "pairwise_results",
	output_filename: Optional[str] = None,
	resume: bool = True,
	save_every: int = 10,
	max_workers: Optional[int] = None,
	verbose: int = 1,
) -> Dict[str, Any]:
	"""Run pairwise judge selection for all candidate model pairs and persist raw outcomes."""
	if llm is None and not api_key:
		raise ValueError("Provide either an instantiated llm or api_key.")

	if verbose >= 1:
		print(
			"[pairwise] Starting selection "
			f"(selector_model={selector_model}, database={database}, sqls_dir={sqls_dir}, resume={resume}, save_every={save_every}, max_workers={max_workers})"
		)

	default_files = {
		"bird_developer": ("datasets_files/BIRD/dev_tables.json", "datasets_files/BIRD/dev.json"),
	}
	if not schema_file or not query_file:
		alias = (database or "").strip().lower()
		if alias in default_files:
			default_schema, default_query = default_files[alias]
			schema_file = schema_file or default_schema
			query_file = query_file or default_query
		else:
			raise ValueError("schema_file and query_file are required unless database='bird_developer'.")

	if verbose >= 1:
		print(f"[pairwise] Using schema_file='{schema_file}' and query_file='{query_file}'")

	candidate_row_index: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]]
	all_models, candidate_index, candidate_row_index = _load_candidate_sqls(sqls_dir=sqls_dir, verbose=verbose)
	candidate_models = [m for m in all_models if m != "ground_truth"]
	if len(candidate_models) < 2:
		raise ValueError("Need at least two candidate models to run pairwise selector.")
	if verbose >= 1:
		print(f"[pairwise] Candidate models considered: {candidate_models}")

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
	if verbose >= 1:
		print(f"[pairwise] Built query index with {len(query_index)} row(s)")

	output_path = _PROJECT_ROOT / output_dir
	output_path.mkdir(parents=True, exist_ok=True)
	if output_filename is None:
		output_filename = f"{selector_model}_pairwise_selector_results.json"
	output_file = output_path / output_filename

	results: List[Dict[str, Any]] = []
	processed_keys = set()
	if resume and output_file.exists():
		existing_payload = load_json_file(str(output_file))
		for row in existing_payload.get("results", []):
			key = (str(row.get("db_id")), int(row.get("question_id", -1)))
			if key[1] >= 0:
				results.append(row)
				processed_keys.add(key)
		if verbose >= 1:
			print(f"[pairwise] Resume enabled: loaded {len(processed_keys)} previously processed query key(s)")
	elif verbose >= 2:
		print("[pairwise] Resume skipped (no prior output file found or resume disabled)")

	per_db_schema_info: Dict[str, Dict[str, Any]] = {}
	processed_since_save = 0
	processed_now = 0
	# Counters to make skip reasons visible in logs.
	skipped_already_done = 0
	skipped_missing_query_meta = 0
	skipped_missing_schema = 0
	skipped_missing_candidates = 0
	tasks: List[Tuple[Tuple[str, int], Dict[str, Any], str, Dict[str, str], Dict[str, Dict[str, Any]]]] = []

	for key in sorted(candidate_index.keys()):
		db_id, question_id = key
		if key in processed_keys:
			skipped_already_done += 1
			continue

		query_meta = query_index.get(key)
		if query_meta is None:
			skipped_missing_query_meta += 1
			continue

		if db_id not in per_db_schema_info:
			# Cache schema per DB to avoid repeated disk parsing across questions.
			per_db_schema_info[db_id] = get_schema_info(db_id, tables_file=schema_file)
		schema_info = per_db_schema_info[db_id]
		if schema_info is None:
			skipped_missing_schema += 1
			continue

		candidates_for_key = candidate_index.get(key, {})
		missing = [m for m in candidate_models if m not in candidates_for_key]
		if missing:
			skipped_missing_candidates += 1
			if verbose >= 2:
				print(f"[pairwise] Skipping db_id={db_id}, qid={question_id}; missing candidate(s): {missing}")
			continue

		schema_formatted = format_schema_info_for_prompt(schema_info)
		candidate_rows_for_key = candidate_row_index.get(key, {})
		tasks.append((key, query_meta, schema_formatted, candidates_for_key, candidate_rows_for_key))

	if verbose >= 1:
		print(f"[pairwise] Queued {len(tasks)} query key(s) for threaded processing")

	workers = max(1, int(max_workers or min(8, (os.cpu_count() or 4))))
	if workers > 1 and api_key and verbose >= 1:
		print("[pairwise] Worker mode: per-thread LLM clients enabled")
	if tasks:
		if verbose >= 1:
			print(f"[pairwise] Running with {workers} worker thread(s)")

		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			future_to_key = {
				pool.submit(
					_build_pairwise_result_for_key,
					key=key,
					candidate_models=candidate_models,
					candidate_rows_for_key=candidate_rows_for_key,
					query_meta=query_meta,
					schema_formatted=schema_formatted,
					candidates_for_key=candidates_for_key,
					selector_model=selector_model,
					api_key=api_key,
					llm=llm,
					verbose=verbose,
				): key
				for key, query_meta, schema_formatted, candidates_for_key, candidate_rows_for_key in tasks
			}

			pending = set(future_to_key.keys())
			while pending:
				done, pending = concurrent.futures.wait(
					pending,
					timeout=20,
					return_when=concurrent.futures.FIRST_COMPLETED,
				)

				if not done:
					if verbose >= 1:
						print(
							f"[pairwise] Heartbeat: completed {processed_now}/{len(tasks)} "
							f"query key(s), waiting on {len(pending)} task(s)"
						)
					continue

				for future in done:
					key = future_to_key[future]
					row = future.result()
					row["selector_model"] = selector_model
					results.append(row)
					processed_keys.add(key)
					processed_since_save += 1
					processed_now += 1

					if verbose >= 2 and (processed_now % 25 == 0 or processed_now == len(tasks)):
						print(f"[pairwise] Processed {processed_now}/{len(tasks)} queued query key(s)")

					if processed_since_save >= save_every:
						# Checkpoint partial progress to reduce work loss on long jobs.
						_save_pairwise_payload(
							output_file=output_file,
							selector_model=selector_model,
							database=database,
							schema_file=schema_file,
							query_file=query_file,
							candidate_models=candidate_models,
							total_queries=len(candidate_index),
							results=results,
						)
						if verbose >= 2:
							print(f"[pairwise] Checkpoint saved ({len(results)} total result rows)")
						processed_since_save = 0

	_save_pairwise_payload(
		output_file=output_file,
		selector_model=selector_model,
		database=database,
		schema_file=schema_file,
		query_file=query_file,
		candidate_models=candidate_models,
		total_queries=len(candidate_index),
		results=results,
	)
	if verbose >= 1:
		print(
			"[pairwise] Run stats: "
			f"processed_now={processed_now}, skipped_already_done={skipped_already_done}, "
			f"skipped_missing_query_meta={skipped_missing_query_meta}, "
			f"skipped_missing_schema={skipped_missing_schema}, "
			f"skipped_missing_candidates={skipped_missing_candidates}"
		)

	if verbose >= 0:
		print(f"[pairwise] Done - {len(results)} query judgments saved to {output_file}")

	return {
		"selector_model": selector_model,
		"output_file": str(output_file),
		"candidate_models": candidate_models,
		"total_queries": len(candidate_index),
		"saved_results": len(results),
	}


if __name__ == "__main__":
	selector_model = "deepseek-chat"
	api_key = _resolve_api_key_for_model(selector_model)
	if not api_key:
		raise RuntimeError("Missing API key for selector model 'deepseek-chat'.")

	llm = initialize_llm(model=selector_model, api_key=api_key, verbose=1)
	summary = compute_pairwise_selector_choices(
		selector_model=selector_model,
		llm=llm,
		api_key=api_key,
		database="bird_developer",
		max_workers= 15,
		verbose=1,
		save_every=1
	)

	print("Summary of selector runs (pairwise judgments):")
	print(json.dumps(summary, indent=2))
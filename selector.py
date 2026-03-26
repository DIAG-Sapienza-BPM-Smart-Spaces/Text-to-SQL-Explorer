import json
import os
import re
import sys
import threading
import argparse
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema.messages import HumanMessage
from common_utils import (
	atomic_dump_json,
	build_pairwise_index,
	extract_pairwise_comparison,
)

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


pairwise_selector_prompt = """
You are an expert SQL programmer and judge.
Given a natural language query, evidence, database schema, and exactly two SQL candidates,
choose which candidate is better.

Return your decision as exactly one of: CANDIDATE A, CANDIDATE B, or TIE.

Natural language query:
{question}

Evidence:
{evidence}

Database schema:
{schema}

Candidate A ({model_a}):
{query_a}

Candidate B ({model_b}):
{query_b}

Answer format:
<brief reasoning>
```choice
<CANDIDATE A|CANDIDATE B|TIE>
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


def _normalize_pairwise_choice(choice_raw: str) -> str:
	"""Normalize pairwise selector output to CANDIDATE A/B/TIE or UNPARSEABLE."""
	upper = (choice_raw or "").upper().strip()
	if "TIE" in upper:
		return "TIE"
	if re.search(r"CANDIDATE\s*A", upper) or re.fullmatch(r"\s*A\s*", upper):
		return "CANDIDATE A"
	if re.search(r"CANDIDATE\s*B", upper) or re.fullmatch(r"\s*B\s*", upper):
		return "CANDIDATE B"
	return "UNPARSEABLE"



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


def _resolve_api_key_for_model(selector_model: str) -> str:
	"""Resolve API key from environment variables for a selector model."""
	if selector_model == "deepseek-chat":
		return os.getenv("DEEPSEEK_API_KEY", "sk-2675b255bc084d70b188e7fccd0aed15")

	# Optional generic fallback if additional providers are wired in initialize_llm.
	return os.getenv("SELECTOR_API_KEY", "")


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


def compute_pairwise_selector_choices(
	selector_model: str = "deepseek-chat",
	llm=None,
	database: str = "bird_developer",
	schema_file: str = None,
	query_file: str = None,
	sqls_dir: str = "candidates",
	pairwise_file: str = "all_pairwise_comparisons.json",
	output_dir: str = "pairwise_results",
	output_filename: Optional[str] = None,
	resume: bool = True,
	save_every: int = 10,
	verbose: int = 1,
) -> Dict[str, Any]:
	"""Run pairwise judge selection for all candidate model pairs and persist raw outcomes."""
	if llm is None:
		raise ValueError("An instantiated llm must be provided.")

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

	all_models, candidate_index = _load_candidate_sqls(sqls_dir=sqls_dir, verbose=verbose)
	candidate_models = [m for m in all_models if m != "ground_truth"]
	if len(candidate_models) < 2:
		raise ValueError("Need at least two candidate models to run pairwise selector.")

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

	per_db_schema_info: Dict[str, Dict[str, Any]] = {}
	processed_since_save = 0

	for key in sorted(candidate_index.keys()):
		db_id, question_id = key
		if key in processed_keys:
			continue

		query_meta = query_index.get(key)
		if query_meta is None:
			continue

		if db_id not in per_db_schema_info:
			per_db_schema_info[db_id] = get_schema_info(db_id, tables_file=schema_file)
		schema_info = per_db_schema_info[db_id]
		if schema_info is None:
			continue

		candidates_for_key = candidate_index.get(key, {})
		missing = [m for m in candidate_models if m not in candidates_for_key]
		if missing:
			continue

		schema_formatted = format_schema_info_for_prompt(schema_info)
		pairwise_rows: List[Dict[str, Any]] = []

		for model_a, model_b in combinations(candidate_models, 2):
			sql_a = candidates_for_key.get(model_a, "")
			sql_b = candidates_for_key.get(model_b, "")
			prompt = pairwise_selector_prompt.format(
				question=query_meta["question"],
				evidence=query_meta.get("evidence", ""),
				schema=schema_formatted,
				model_a=model_a,
				model_b=model_b,
				query_a=sql_a,
				query_b=sql_b,
			)

			response = llm.invoke([HumanMessage(content=prompt)])
			response_content = response.content if hasattr(response, "content") else str(response)
			reasoning, choice_raw = _extract_reasoning_and_choice(str(response_content))
			normalized = _normalize_pairwise_choice(choice_raw)

			winner = "tie"
			if normalized == "CANDIDATE A":
				winner = model_a
			elif normalized == "CANDIDATE B":
				winner = model_b

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
					"ground_truth_vs_model_a": extract_pairwise_comparison(pairwise_index.get(key), "ground_truth", model_a),
					"ground_truth_vs_model_b": extract_pairwise_comparison(pairwise_index.get(key), "ground_truth", model_b),
				}
			)

		leaderboard = _derive_pairwise_leaderboard(candidate_models, pairwise_rows)
		selected_candidate_model = leaderboard[0]["model"] if leaderboard else None

		results.append(
			{
				"question_id": int(question_id),
				"db_id": str(db_id),
				"selector_model": selector_model,
				"question": query_meta.get("question", ""),
				"evidence": query_meta.get("evidence", ""),
				"candidate_models": candidate_models,
				"pairwise_judgments": pairwise_rows,
				"leaderboard": leaderboard,
				"selected_candidate_model": selected_candidate_model,
			}
		)
		processed_keys.add(key)
		processed_since_save += 1

		if processed_since_save >= save_every:
			payload = {
				"selector_model": selector_model,
				"mode": "pairwise_selector",
				"database": database,
				"schema_file": schema_file,
				"query_file": query_file,
				"pairwise_file": pairwise_file,
				"candidate_models": candidate_models,
				"total_queries": len(candidate_index),
				"results": results,
			}
			atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)
			processed_since_save = 0

	payload = {
		"selector_model": selector_model,
		"mode": "pairwise_selector",
		"database": database,
		"schema_file": schema_file,
		"query_file": query_file,
		"pairwise_file": pairwise_file,
		"candidate_models": candidate_models,
		"total_queries": len(candidate_index),
		"results": results,
	}
	atomic_dump_json(output_file, payload, lock=_FILE_WRITE_LOCK)

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
		database="bird_developer",
		verbose=1,
	)

	print("Summary of selector runs (pairwise judgments):")
	print(json.dumps(summary, indent=2))
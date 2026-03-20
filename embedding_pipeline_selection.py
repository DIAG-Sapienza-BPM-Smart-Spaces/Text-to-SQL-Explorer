import json
from pathlib import Path
from typing import Any, Optional

from embedding import (
	sql_selection_trough_embedding_similarities,
)


ROOT = Path(__file__).resolve().parent
SQLS_DIR = ROOT / "sqls"
PAIRWISE_PATH = ROOT / "all_pairwise_comparisons.json"
OUTPUT_PATH = ROOT / "embedding_pipeline_selection_results.json"
DEFAULT_SAVE_EVERY = 10


def log_message(verbose: int, level: int, message: str) -> None:
	"""Print a message only when the active verbose level allows it."""
	if verbose >= level:
		print(message)


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def save_json(path: Path, payload: Any) -> None:
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def load_existing_results(path: Path, verbose: int = 1) -> list[dict[str, Any]]:
	"""Load previous pipeline output to support resume mode."""
	if not path.exists():
		return []

	try:
		payload = load_json(path)
	except Exception as exc:
		log_message(verbose, 1, f"Failed to load existing output {path.name}: {exc}")
		return []

	if not isinstance(payload, list):
		log_message(verbose, 1, f"Existing output {path.name} is not a list. Starting fresh.")
		return []

	valid_rows: list[dict[str, Any]] = []
	for row in payload:
		if isinstance(row, dict) and row.get("general_id") is not None:
			valid_rows.append(row)

	log_message(verbose, 1, f"Loaded {len(valid_rows)} existing rows from {path.name}")
	return valid_rows


def to_int_or_none(value: Any) -> Optional[int]:
	"""Safely convert a value to int, returning None if conversion fails."""
	try:
		return int(value)
	except Exception:
		return None


def model_name_from_file(file_path: Path) -> str:
	suffix = "_query_results"
	stem = file_path.stem
	return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def load_sql_candidates(sqls_dir: Path, verbose: int = 1) -> dict[int, list[dict[str, Any]]]:
	grouped: dict[int, list[dict[str, Any]]] = {}

	json_files = sorted(sqls_dir.glob("*_query_results.json"))
	log_message(verbose, 1, f"Found {len(json_files)} SQL result files")

	for file_path in json_files:
		model_name = model_name_from_file(file_path)
		log_message(verbose, 1, f"Loading candidates from {file_path.name}")
		records = load_json(file_path)
		if not isinstance(records, list):
			log_message(verbose, 2, f"[DEBUG] Skipping {file_path.name}: top-level JSON is not a list")
			continue

		for rec in records:
			if not isinstance(rec, dict):
				continue

			general_id = rec.get("general_id")
			if general_id is None:
				log_message(verbose, 2, f"[DEBUG] Skipping record without general_id in {file_path.name}")
				continue

			clean_sql = rec.get("clean_sql")
			if not clean_sql:
				log_message(verbose, 2, f"[DEBUG] Skipping general_id={general_id} from {file_path.name}: missing clean_sql")
				continue

			candidate = {
				"model": model_name,
				"general_id": general_id,
				"question_id": rec.get("question_id"),
				"db_id": rec.get("db_id"),
				"clean_sql": clean_sql,
			}
			grouped.setdefault(int(general_id), []).append(candidate)

	return grouped


def load_pairwise_index(path: Path, verbose: int = 1) -> dict[int, dict[str, Any]]:
	raw = load_json(path)
	index: dict[int, dict[str, Any]] = {}

	if isinstance(raw, dict):
		log_message(verbose, 2, "[DEBUG] Pairwise JSON is an object keyed by general_id")
		for key, value in raw.items():
			if not isinstance(value, dict):
				continue

			gid = to_int_or_none(key)
			if gid is None:
				gid = to_int_or_none(value.get("general_id"))

			if gid is not None and gid >= 0:
				index[gid] = value
	elif isinstance(raw, list):
		log_message(verbose, 2, "[DEBUG] Pairwise JSON is a list of objects")
		for value in raw:
			if not isinstance(value, dict):
				continue

			gid = to_int_or_none(value.get("general_id"))
			if gid is None:
				continue

			index[gid] = value

	return index


def get_ground_truth_metrics(entry: dict[str, Any], selected_model: str) -> Optional[dict[str, Any]]:
	comparisons = entry.get("comparisons") if isinstance(entry, dict) else None
	if not isinstance(comparisons, dict):
		return None

	for comp_key, comp in comparisons.items():
		if not isinstance(comp, dict):
			continue
		s1 = comp.get("system1")
		s2 = comp.get("system2")
		if {s1, s2} == {"ground_truth", selected_model}:
			metrics = {
				"comparison_key": comp_key,
				"system1": s1,
				"system2": s2,
				"schema_precision": comp.get("schema_precision"),
				"schema_recall": comp.get("schema_recall"),
				"cell_value_accuracy": comp.get("cell_value_accuracy"),
				"row_set_jaccard": comp.get("row_set_jaccard"),
				"execution_accuracy": comp.get("execution_accuracy"),
				"f1_score": comp.get("f1_score"),
				"comparison_performed": comp.get("comparison_performed"),
				"fast_path": comp.get("fast_path"),
			}
			return metrics

	return None


def build_selection_results(
	candidates_by_general_id: dict[int, list[dict[str, Any]]],
	pairwise_index: dict[int, dict[str, Any]],
	output_path: Path,
	existing_results: Optional[list[dict[str, Any]]] = None,
	save_every: int = 1,
	verbose: int = 1,
) -> list[dict[str, Any]]:
	output: list[dict[str, Any]] = list(existing_results or [])
	processed_ids = {int(row["general_id"]) for row in output if isinstance(row, dict) and row.get("general_id") is not None}
	new_rows_since_checkpoint = 0

	log_message(verbose, 1, f"Resume mode: {len(processed_ids)} general_id values already completed")

	for general_id in sorted(candidates_by_general_id.keys()):
		if general_id in processed_ids:
			log_message(verbose, 2, f"[DEBUG] Skipping already processed general_id={general_id}")
			continue

		candidates = candidates_by_general_id[general_id]
		if not candidates:
			continue

		sql_queries = [c["clean_sql"] for c in candidates]

		log_message(verbose, 1, f"Selecting query for general_id={general_id} from {len(candidates)} candidates")

		embedding_verbose = 0 if verbose == 0 else verbose - 1
		selected_sql, stats = sql_selection_trough_embedding_similarities(sql_queries, verbose=embedding_verbose)

		selected_index = stats["selected_index"]
		if selected_index < 0 or selected_index >= len(candidates):
			continue

		selected_candidate = next(
			(c for c in candidates if c["clean_sql"] == selected_sql),
			candidates[selected_index],
		)

		pairwise_entry = pairwise_index.get(general_id, {})
		gt_metrics = get_ground_truth_metrics(pairwise_entry, selected_candidate["model"])
		if verbose >= 2 and gt_metrics is None:
			log_message(verbose, 2, f"[DEBUG] No ground_truth metrics found for general_id={general_id}, model={selected_candidate['model']}")

		result_row = {
			"general_id": general_id,
			"question_id": selected_candidate.get("question_id"),
			"db_id": selected_candidate.get("db_id"),
			"selected_model": selected_candidate["model"],
			"selected_clean_sql": selected_candidate["clean_sql"],
			"num_clusters": stats.get("num_clusters"),
			"cluster_sizes": stats.get("cluster_sizes"),
			"biggest_cluster_size": stats.get("biggest_cluster_size"),
			"selection_statistics": stats,
			"ground_truth_comparison_metrics": gt_metrics,
			"candidate_models": [c["model"] for c in candidates],
			"num_model_candidates": len(candidates),
		}
		output.append(result_row)
		processed_ids.add(general_id)
		new_rows_since_checkpoint += 1

		if new_rows_since_checkpoint >= max(1, save_every):
			save_json(output_path, output)
			log_message(verbose, 1, f"Checkpoint saved at general_id={general_id} ({len(output)} total rows)")
			new_rows_since_checkpoint = 0

		if verbose >= 2:
			log_message(verbose, 2, f"[DEBUG] Selected model={selected_candidate['model']} index={stats['selected_index']} mean={stats['similarity_mean']:.4f}")

		if verbose >= 1 and general_id % 100 == 0:
			log_message(verbose, 1, f"Processed general_id={general_id}")

	if new_rows_since_checkpoint > 0:
		save_json(output_path, output)
		log_message(verbose, 1, f"Final checkpoint saved ({len(output)} total rows)")

	return output


def main(verbose: int = 1) -> None:
	if not SQLS_DIR.exists():
		raise FileNotFoundError(f"Missing sqls directory: {SQLS_DIR}")
	if not PAIRWISE_PATH.exists():
		raise FileNotFoundError(f"Missing pairwise file: {PAIRWISE_PATH}")

	log_message(verbose, 1, "Loading SQL candidates...")
	candidates_by_general_id = load_sql_candidates(SQLS_DIR, verbose=verbose)
	log_message(verbose, 1, f"Loaded candidates for {len(candidates_by_general_id)} general_id values")

	log_message(verbose, 1, "Loading pairwise comparisons...")
	pairwise_index = load_pairwise_index(PAIRWISE_PATH, verbose=verbose)
	log_message(verbose, 1, f"Loaded pairwise entries: {len(pairwise_index)}")

	log_message(verbose, 1, "Loading existing output for resume...")
	existing_results = load_existing_results(OUTPUT_PATH, verbose=verbose)

	log_message(verbose, 1, "Running embedding-based selection pipeline...")
	output_rows = build_selection_results(
		candidates_by_general_id,
		pairwise_index,
		output_path=OUTPUT_PATH,
		existing_results=existing_results,
		save_every=DEFAULT_SAVE_EVERY,
		verbose=verbose,
	)
	log_message(verbose, 1, f"Selection completed for {len(output_rows)} rows")

	log_message(verbose, 1, f"Output kept up-to-date incrementally at {OUTPUT_PATH}")


if __name__ == "__main__":
	# 0 = silent, 1 = step information, 2 = detailed debug output
	main(verbose=1)

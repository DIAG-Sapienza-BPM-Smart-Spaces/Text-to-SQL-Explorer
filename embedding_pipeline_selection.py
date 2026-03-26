from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

from embedding import (
	compute_average_and_std_of_similarities,
	compute_cosine_similarity,
	compute_similarity_groups_pairwise,
	compute_similarity_matrix,
	get_vector_closest_to_centroid,
	load_embedder,
	load_json_artifact,
	load_similarity_matrix_artifact,
	parse_and_normalize_sql,
)
from common_utils import atomic_dump_json, load_json as load_json_file


ROOT = Path(__file__).resolve().parent
SQLS_DIR = ROOT / "candidates"
OUTPUT_PATH = ROOT / "embedding_pipeline_selection_results.json"
GROUND_TRUTH_PATH = ROOT / "datasets_files" / "BIRD" / "dev.json"
METRICS_RESULTS_DIR = ROOT / "metrics_results"
GROUND_TRUTH_AGGREGATE_PATH = ROOT / "embedding_ground_truth_cluster_stats.json"
DEFAULT_SAVE_EVERY = 10
DEFAULT_MAX_WORKERS = 7
PRECOMPUTED_SIMILARITY_INDEX = ROOT / "precomputed" / "similarity" / "bird_dev_similarity_index.json"


def log_message(verbose: int, level: int, message: str) -> None:
	"""Print a message only when the active verbose level allows it."""
	if verbose >= level:
		print(message)


def load_json(path: Path) -> Any:
	return load_json_file(path)


def save_json(path: Path, payload: Any) -> None:
	atomic_dump_json(path, payload)


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


def pick_candidate_sql(rec: dict[str, Any]) -> Optional[str]:
	"""Pick the best available SQL text from a result record."""
	for key in ("clean_sql", "candidate_sql", "extracted_sql", "generated_sql"):
		value = rec.get(key)
		if isinstance(value, str):
			text = value.strip()
			if text:
				return text
	return None


def db_qid_key(db_id: Any, question_id: Any) -> Optional[tuple[str, int]]:
	qid = to_int_or_none(question_id)
	if db_id is None or qid is None:
		return None
	return str(db_id), qid


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

			clean_sql = pick_candidate_sql(rec)
			if not clean_sql:
				log_message(verbose, 2, f"[DEBUG] Skipping general_id={general_id} from {file_path.name}: missing SQL text")
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


def build_general_id_index(
	candidates_by_general_id: dict[int, list[dict[str, Any]]],
) -> dict[tuple[str, int], int]:
	"""Build (db_id, question_id) -> general_id mapping from loaded candidates."""
	index: dict[tuple[str, int], int] = {}
	for general_id, candidates in candidates_by_general_id.items():
		for candidate in candidates:
			key = db_qid_key(candidate.get("db_id"), candidate.get("question_id"))
			if key is not None:
				index[key] = general_id
	return index


def load_ground_truth_sql_by_general_id(
	ground_truth_path: Path,
	candidates_by_general_id: dict[int, list[dict[str, Any]]],
	verbose: int = 1,
) -> dict[int, dict[str, Any]]:
	"""
	Load ground truth SQL from dev.json and map rows to general_id.

	General IDs are reconstructed by matching (db_id, question_id)
	against the index recovered from candidate result files.
	"""
	payload = load_json(ground_truth_path)
	if not isinstance(payload, list):
		raise ValueError(f"Ground truth file is not a list: {ground_truth_path}")

	db_qid_to_general_id = build_general_id_index(candidates_by_general_id)
	ground_truth_by_general_id: dict[int, dict[str, Any]] = {}
	unmatched = 0

	for row in payload:
		if not isinstance(row, dict):
			continue

		gt_sql = row.get("SQL")
		if not isinstance(gt_sql, str) or not gt_sql.strip():
			continue

		qid = to_int_or_none(row.get("question_id"))
		db_id = row.get("db_id")
		if qid is None or db_id is None:
			continue

		key = db_qid_key(db_id, qid)
		general_id = db_qid_to_general_id.get(key) if key is not None else None

		# Fallback for datasets where general_id aligns with question_id.
		if general_id is None and qid in candidates_by_general_id:
			general_id = qid

		if general_id is None:
			unmatched += 1
			continue

		ground_truth_by_general_id[int(general_id)] = {
			"general_id": int(general_id),
			"question_id": int(qid),
			"db_id": str(db_id),
			"ground_truth_sql": gt_sql,
		}

	log_message(
		verbose,
		1,
		f"Loaded ground truth SQL for {len(ground_truth_by_general_id)} general_id values "
		f"({unmatched} unmatched rows in {ground_truth_path.name})",
	)
	return ground_truth_by_general_id


def load_ground_truth_metrics_lookup(metrics_dir: Path, verbose: int = 1) -> dict[tuple[str, int, str], dict[str, Any]]:
	"""Load BIRD dev canonical metrics keyed by (db_id, question_id, model)."""
	lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
	if not metrics_dir.exists():
		log_message(verbose, 1, f"Metrics directory not found: {metrics_dir}")
		return lookup

	pattern = "evaluation_sql_metrics_*_vs_ground_truth.json"
	for path in sorted(metrics_dir.glob(pattern)):
		model_name = path.name[len("evaluation_sql_metrics_"):-len("_vs_ground_truth.json")]
		payload = load_json(path)
		if not isinstance(payload, list):
			continue

		for row in payload:
			if not isinstance(row, dict):
				continue
			db_id = row.get("db_id")
			qid = to_int_or_none(row.get("question_id"))
			if db_id is None or qid is None:
				continue

			lookup[(str(db_id), int(qid), model_name)] = {
				"system1": "ground_truth",
				"system2": model_name,
				"execution_accuracy": row.get("execution_accuracy"),
				"exact_match": row.get("exact_match"),
				"sql_f1_score": row.get("sql_f1_score"),
				"response_schema_f1_score": row.get("response_schema_f1_score"),
				"cell_f1_score": row.get("cell_f1_score"),
			}

	log_message(verbose, 1, f"Loaded ground-truth metrics lookup entries: {len(lookup)}")
	return lookup


def get_ground_truth_metrics(
	*,
	db_id: Any,
	question_id: Any,
	selected_model: str,
	metrics_lookup: dict[tuple[str, int, str], dict[str, Any]],
) -> Optional[dict[str, Any]]:
	"""Resolve selected-model metrics versus ground truth from metrics lookup."""
	qid = to_int_or_none(question_id)
	if db_id is None or qid is None or not selected_model:
		return None
	return metrics_lookup.get((str(db_id), int(qid), str(selected_model)))


def select_candidate_from_embeddings(
	sql_queries: list[str],
	precomputed_similarity_matrix: Optional[np.ndarray] = None,
	verbose: int = 0,
) -> dict[str, Any]:
	"""Run candidate-only embedding selection and expose clustering internals."""
	if not sql_queries:
		raise ValueError("sql_queries must contain at least one query")

	embedder = load_embedder(verbose=verbose)
	vectors = [embedder.encode(parse_and_normalize_sql(sql, verbose=verbose)) for sql in sql_queries]
	similarity_matrix = (
		precomputed_similarity_matrix
		if isinstance(precomputed_similarity_matrix, np.ndarray) and precomputed_similarity_matrix.size > 0
		else compute_similarity_matrix(vectors)
	)
	avg_sim, std_sim = compute_average_and_std_of_similarities(similarity_matrix)
	groups = compute_similarity_groups_pairwise(vectors, similarity_matrix, verbose=verbose, threshold=avg_sim)
	biggest_group = max(groups, key=len)
	selected_index = get_vector_closest_to_centroid(biggest_group)
	selected_sql = sql_queries[selected_index]

	selection_stats = {
		"selected_index": int(selected_index),
		"num_candidates": len(sql_queries),
		"num_clusters": len(groups),
		"cluster_sizes": [len(group) for group in groups],
		"biggest_cluster_size": len(biggest_group),
		"similarity_mean": float(avg_sim),
		"similarity_std": float(std_sim),
	}

	return {
		"selected_sql": selected_sql,
		"selection_stats": selection_stats,
		"vectors": vectors,
		"similarity_threshold": float(avg_sim),
		"biggest_cluster_indices": [int(index) for index, _ in biggest_group],
	}


def load_precomputed_similarity_index(path: Path, verbose: int = 1) -> dict[str, Any]:
	"""Load precomputed similarity index if available."""
	if not path.exists():
		log_message(verbose, 1, f"Precomputed similarity index not found: {path}")
		return {}
	try:
		return load_json_artifact(path)
	except Exception as exc:
		log_message(verbose, 1, f"Failed to load precomputed similarity index: {exc}")
		return {}


def load_query_precomputed_similarity(
	precomputed_index: dict[str, Any],
	db_id: str,
	question_id: int,
	verbose: int = 0,
) -> Optional[np.ndarray]:
	"""Load query-specific precomputed similarity matrix for active candidates."""
	queries = precomputed_index.get("queries", {}) if isinstance(precomputed_index, dict) else {}
	key = f"{db_id}|{int(question_id)}"
	entry = queries.get(key)
	if not isinstance(entry, dict):
		return None

	matrix_rel = entry.get("matrix_file")
	if not isinstance(matrix_rel, str) or not matrix_rel:
		return None

	matrix_path = ROOT / matrix_rel
	if not matrix_path.exists():
		return None

	try:
		return load_similarity_matrix_artifact(matrix_path)
	except Exception as exc:
		log_message(verbose, 2, f"[DEBUG] Failed loading matrix {matrix_path}: {exc}")
		return None


def analyze_ground_truth_embedding(
	ground_truth_sql: str,
	vectors: list[Any],
	biggest_cluster_indices: list[int],
	selected_index: int,
	similarity_threshold: float,
	verbose: int = 0,
) -> dict[str, Any]:
	"""Compare GT embedding against the already computed biggest candidate cluster."""
	embedder = load_embedder(verbose=verbose)
	gt_vector = embedder.encode(parse_and_normalize_sql(ground_truth_sql, verbose=verbose))

	if not biggest_cluster_indices:
		return {
			"would_join_biggest_cluster": False,
			"similarity_threshold": float(similarity_threshold),
			"similarity_to_biggest_cluster_centroid": None,
			"euclidean_distance_to_biggest_cluster_centroid": None,
			"similarity_to_selected_candidate": float(compute_cosine_similarity(gt_vector, vectors[selected_index])),
			"euclidean_distance_to_selected_candidate": float(np.linalg.norm(np.array(gt_vector) - np.array(vectors[selected_index]))),
			"max_similarity_to_biggest_cluster_member": None,
			"closest_biggest_cluster_member_index": None,
			"closest_biggest_cluster_member_similarity": None,
		}

	biggest_cluster_vectors = [vectors[idx] for idx in biggest_cluster_indices]
	centroid = np.mean([np.array(v) for v in biggest_cluster_vectors], axis=0)

	similarities_to_biggest_cluster = [compute_cosine_similarity(gt_vector, vectors[idx]) for idx in biggest_cluster_indices]
	closest_local_idx = int(np.argmax(similarities_to_biggest_cluster))
	closest_global_idx = biggest_cluster_indices[closest_local_idx]
	max_similarity = float(similarities_to_biggest_cluster[closest_local_idx])

	would_join_biggest_cluster = bool(max_similarity >= similarity_threshold)
	similarity_to_centroid = float(compute_cosine_similarity(gt_vector, centroid))
	distance_to_centroid = float(np.linalg.norm(np.array(gt_vector) - np.array(centroid)))
	similarity_to_selected = float(compute_cosine_similarity(gt_vector, vectors[selected_index]))
	distance_to_selected = float(np.linalg.norm(np.array(gt_vector) - np.array(vectors[selected_index])))

	return {
		"would_join_biggest_cluster": would_join_biggest_cluster,
		"similarity_threshold": float(similarity_threshold),
		"similarity_to_biggest_cluster_centroid": similarity_to_centroid,
		"euclidean_distance_to_biggest_cluster_centroid": distance_to_centroid,
		"similarity_to_selected_candidate": similarity_to_selected,
		"euclidean_distance_to_selected_candidate": distance_to_selected,
		"max_similarity_to_biggest_cluster_member": max_similarity,
		"closest_biggest_cluster_member_index": int(closest_global_idx),
		"closest_biggest_cluster_member_similarity": max_similarity,
	}


def build_ground_truth_cluster_aggregate(output_rows: list[dict[str, Any]]) -> dict[str, Any]:
	"""Compute global aggregate statistics for GT-vs-cluster analysis."""
	rows_with_gt = [
		row for row in output_rows
		if isinstance(row, dict)
		and isinstance(row.get("ground_truth_cluster_analysis"), dict)
	]

	def mean_or_none(values: list[float]) -> Optional[float]:
		return float(sum(values) / len(values)) if values else None

	total_rows = len(output_rows)
	total_with_gt = len(rows_with_gt)
	join_flags = [
		bool(row["ground_truth_cluster_analysis"].get("would_join_biggest_cluster"))
		for row in rows_with_gt
	]

	centroid_similarities = [
		float(row["ground_truth_cluster_analysis"]["similarity_to_biggest_cluster_centroid"])
		for row in rows_with_gt
		if row["ground_truth_cluster_analysis"].get("similarity_to_biggest_cluster_centroid") is not None
	]
	selected_similarities = [
		float(row["ground_truth_cluster_analysis"]["similarity_to_selected_candidate"])
		for row in rows_with_gt
		if row["ground_truth_cluster_analysis"].get("similarity_to_selected_candidate") is not None
	]
	centroid_distances = [
		float(row["ground_truth_cluster_analysis"]["euclidean_distance_to_biggest_cluster_centroid"])
		for row in rows_with_gt
		if row["ground_truth_cluster_analysis"].get("euclidean_distance_to_biggest_cluster_centroid") is not None
	]
	selected_distances = [
		float(row["ground_truth_cluster_analysis"]["euclidean_distance_to_selected_candidate"])
		for row in rows_with_gt
		if row["ground_truth_cluster_analysis"].get("euclidean_distance_to_selected_candidate") is not None
	]

	per_model: dict[str, dict[str, Any]] = {}
	for row in rows_with_gt:
		model = str(row.get("selected_model"))
		model_entry = per_model.setdefault(
			model,
			{
				"count": 0,
				"joins_biggest_cluster": 0,
				"avg_similarity_to_biggest_cluster_centroid": [],
				"avg_similarity_to_selected_candidate": [],
			},
		)
		analysis = row["ground_truth_cluster_analysis"]
		model_entry["count"] += 1
		if analysis.get("would_join_biggest_cluster"):
			model_entry["joins_biggest_cluster"] += 1
		if analysis.get("similarity_to_biggest_cluster_centroid") is not None:
			model_entry["avg_similarity_to_biggest_cluster_centroid"].append(float(analysis["similarity_to_biggest_cluster_centroid"]))
		if analysis.get("similarity_to_selected_candidate") is not None:
			model_entry["avg_similarity_to_selected_candidate"].append(float(analysis["similarity_to_selected_candidate"]))

	for model, data in per_model.items():
		count = data["count"]
		joins = data["joins_biggest_cluster"]
		data["join_rate"] = float(joins / count) if count else 0.0
		data["avg_similarity_to_biggest_cluster_centroid"] = mean_or_none(data["avg_similarity_to_biggest_cluster_centroid"])
		data["avg_similarity_to_selected_candidate"] = mean_or_none(data["avg_similarity_to_selected_candidate"])

	return {
		"total_rows": total_rows,
		"rows_with_ground_truth_analysis": total_with_gt,
		"rows_without_ground_truth_analysis": total_rows - total_with_gt,
		"ground_truth_joins_biggest_cluster_count": int(sum(1 for x in join_flags if x)),
		"ground_truth_joins_biggest_cluster_rate": float(sum(1 for x in join_flags if x) / total_with_gt) if total_with_gt else 0.0,
		"avg_similarity_ground_truth_to_biggest_cluster_centroid": mean_or_none(centroid_similarities),
		"avg_similarity_ground_truth_to_selected_candidate": mean_or_none(selected_similarities),
		"avg_distance_ground_truth_to_biggest_cluster_centroid": mean_or_none(centroid_distances),
		"avg_distance_ground_truth_to_selected_candidate": mean_or_none(selected_distances),
		"per_selected_model": per_model,
	}


def process_single_general_id(
	general_id: int,
	candidates: list[dict[str, Any]],
	ground_truth_entry: Optional[dict[str, Any]],
	metrics_lookup: dict[tuple[str, int, str], dict[str, Any]],
	precomputed_index: Optional[dict[str, Any]] = None,
	verbose: int = 1,
) -> Optional[dict[str, Any]]:
	if not candidates:
		return None

	sql_queries = [c["clean_sql"] for c in candidates]
	embedding_verbose = 0 if verbose == 0 else max(0, verbose - 1)
	first_candidate = candidates[0]
	query_matrix = load_query_precomputed_similarity(
		precomputed_index or {},
		str(first_candidate.get("db_id", "")),
		int(first_candidate.get("question_id", -1)),
		verbose=embedding_verbose,
	)
	selection = select_candidate_from_embeddings(
		sql_queries,
		precomputed_similarity_matrix=query_matrix,
		verbose=embedding_verbose,
	)
	stats = selection["selection_stats"]
	selected_index = stats["selected_index"]

	if selected_index < 0 or selected_index >= len(candidates):
		return None

	selected_sql = selection["selected_sql"]
	selected_candidate = next(
		(c for c in candidates if c["clean_sql"] == selected_sql),
		candidates[selected_index],
	)

	gt_metrics = get_ground_truth_metrics(
		db_id=selected_candidate.get("db_id"),
		question_id=selected_candidate.get("question_id"),
		selected_model=selected_candidate["model"],
		metrics_lookup=metrics_lookup,
	)

	ground_truth_cluster_analysis = None
	ground_truth_sql = None
	if ground_truth_entry is not None:
		ground_truth_sql = ground_truth_entry.get("ground_truth_sql")
		if isinstance(ground_truth_sql, str) and ground_truth_sql.strip():
			ground_truth_cluster_analysis = analyze_ground_truth_embedding(
				ground_truth_sql=ground_truth_sql,
				vectors=selection["vectors"],
				biggest_cluster_indices=selection["biggest_cluster_indices"],
				selected_index=selected_index,
				similarity_threshold=selection["similarity_threshold"],
				verbose=embedding_verbose,
			)

	result_row = {
		"general_id": int(general_id),
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
		"ground_truth_sql_available": bool(ground_truth_sql),
		"ground_truth_cluster_analysis": ground_truth_cluster_analysis,
	}

	return result_row


def build_selection_results(
	candidates_by_general_id: dict[int, list[dict[str, Any]]],
	ground_truth_by_general_id: dict[int, dict[str, Any]],
	metrics_lookup: dict[tuple[str, int, str], dict[str, Any]],
	output_path: Path,
	existing_results: Optional[list[dict[str, Any]]] = None,
	save_every: int = 1,
	max_workers: int = DEFAULT_MAX_WORKERS,
	precomputed_index: Optional[dict[str, Any]] = None,
	verbose: int = 1,
) -> list[dict[str, Any]]:
	output_by_general_id: dict[int, dict[str, Any]] = {}
	for row in list(existing_results or []):
		if not isinstance(row, dict):
			continue
		gid = to_int_or_none(row.get("general_id"))
		if gid is None:
			continue
		output_by_general_id[int(gid)] = row

	processed_ids = {
		gid
		for gid, row in output_by_general_id.items()
		if isinstance(row.get("ground_truth_cluster_analysis"), dict)
	}
	new_rows_since_checkpoint = 0

	log_message(verbose, 1, f"Resume mode: {len(processed_ids)} general_id values already completed")
	general_ids_to_process = [
		general_id
		for general_id in sorted(candidates_by_general_id.keys())
		if general_id not in processed_ids
	]
	log_message(verbose, 1, f"Processing {len(general_ids_to_process)} general_id values with {max_workers} worker threads")

	if general_ids_to_process:
		embedding_verbose = 0 if verbose == 0 else max(0, verbose - 1)
		log_message(verbose, 1, "Warming up embedder before starting worker threads...")
		load_embedder(verbose=embedding_verbose)

	def checkpoint_save(last_general_id: Optional[int] = None) -> None:
		sorted_output = sorted(output_by_general_id.values(), key=lambda r: int(r.get("general_id", -1)))
		save_json(output_path, sorted_output)
		if last_general_id is not None:
			log_message(verbose, 1, f"Checkpoint saved at general_id={last_general_id} ({len(sorted_output)} total rows)")
		else:
			log_message(verbose, 1, f"Final checkpoint saved ({len(sorted_output)} total rows)")

	completed = 0
	with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
		futures = {
			executor.submit(
				process_single_general_id,
				general_id,
				candidates_by_general_id[general_id],
				ground_truth_by_general_id.get(general_id),
				metrics_lookup,
				precomputed_index,
				verbose,
			): general_id
			for general_id in general_ids_to_process
		}

		for future in as_completed(futures):
			general_id = futures[future]
			completed += 1
			try:
				result_row = future.result()
			except Exception as exc:
				log_message(verbose, 1, f"[WARN] Failed to process general_id={general_id}: {exc}")
				continue

			if result_row is None:
				continue

			output_by_general_id[int(general_id)] = result_row
			new_rows_since_checkpoint += 1

			if verbose >= 2:
				stats = result_row.get("selection_statistics", {})
				log_message(
					verbose,
					2,
					f"[DEBUG] Selected model={result_row.get('selected_model')} "
					f"index={stats.get('selected_index')} "
					f"mean={float(stats.get('similarity_mean', 0.0)):.4f}",
				)

			if new_rows_since_checkpoint >= max(1, save_every):
				checkpoint_save(last_general_id=general_id)
				new_rows_since_checkpoint = 0

			if verbose >= 1 and completed % 100 == 0:
				log_message(verbose, 1, f"Completed {completed}/{len(general_ids_to_process)} pending rows")

	if new_rows_since_checkpoint > 0:
		checkpoint_save(last_general_id=None)

	return sorted(output_by_general_id.values(), key=lambda r: int(r.get("general_id", -1)))


def main(verbose: int = 1) -> None:
	if not SQLS_DIR.exists():
		raise FileNotFoundError(f"Missing candidates directory: {SQLS_DIR}")
	if not GROUND_TRUTH_PATH.exists():
		raise FileNotFoundError(f"Missing ground truth file: {GROUND_TRUTH_PATH}")
	if not METRICS_RESULTS_DIR.exists():
		raise FileNotFoundError(f"Missing metrics results directory: {METRICS_RESULTS_DIR}")

	log_message(verbose, 1, "Loading SQL candidates...")
	candidates_by_general_id = load_sql_candidates(SQLS_DIR, verbose=verbose)
	log_message(verbose, 1, f"Loaded candidates for {len(candidates_by_general_id)} general_id values")

	log_message(verbose, 1, "Loading ground truth SQL from dev.json...")
	ground_truth_by_general_id = load_ground_truth_sql_by_general_id(
		GROUND_TRUTH_PATH,
		candidates_by_general_id,
		verbose=verbose,
	)

	log_message(verbose, 1, "Loading existing output for resume...")
	existing_results = load_existing_results(OUTPUT_PATH, verbose=verbose)

	log_message(verbose, 1, "Loading ground-truth metrics lookup...")
	metrics_lookup = load_ground_truth_metrics_lookup(METRICS_RESULTS_DIR, verbose=verbose)

	log_message(verbose, 1, "Loading precomputed similarity index...")
	precomputed_index = load_precomputed_similarity_index(PRECOMPUTED_SIMILARITY_INDEX, verbose=verbose)

	log_message(verbose, 1, "Running embedding-based selection pipeline...")
	output_rows = build_selection_results(
		candidates_by_general_id,
		ground_truth_by_general_id,
		metrics_lookup,
		output_path=OUTPUT_PATH,
		existing_results=existing_results,
		save_every=DEFAULT_SAVE_EVERY,
		max_workers=DEFAULT_MAX_WORKERS,
		precomputed_index=precomputed_index,
		verbose=verbose,
	)
	log_message(verbose, 1, f"Selection completed for {len(output_rows)} rows")

	aggregate = build_ground_truth_cluster_aggregate(output_rows)
	save_json(GROUND_TRUTH_AGGREGATE_PATH, aggregate)
	log_message(verbose, 1, f"Ground truth aggregate stats saved to {GROUND_TRUTH_AGGREGATE_PATH}")

	log_message(verbose, 1, f"Output kept up-to-date incrementally at {OUTPUT_PATH}")


if __name__ == "__main__":
	# 0 = silent, 1 = step information, 2 = detailed debug output
	main(verbose=1)

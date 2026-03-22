import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from common_utils import load_json


METRIC_FIELDS = [
	"schema_precision",
	"schema_recall",
	"cell_value_accuracy",
	"row_set_jaccard",
	"execution_accuracy",
	"f1_score",
]


def _safe_float(value: Any) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return 0.0


def _load_json(path: Path) -> Any:
	return load_json(path)


def _extract_metrics(record: Dict[str, Any]) -> Dict[str, float]:
	return {field: _safe_float(record.get(field, 0.0)) for field in METRIC_FIELDS}


def _get_ground_truth_comparison(
	comparisons: Dict[str, Dict[str, Any]], selected_model: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
	# Prefer the canonical key shape first.
	canonical_key = f"ground_truth_vs_{selected_model}"
	if canonical_key in comparisons:
		return canonical_key, comparisons[canonical_key]

	# Fall back to scanning comparisons for a matching pairing.
	for key, value in comparisons.items():
		system1 = value.get("system1")
		system2 = value.get("system2")
		if system1 == "ground_truth" and system2 == selected_model:
			return key, value

	return None, None


def compute_selector_performance(
	selector_data: Dict[str, Any], pairwise_data: Dict[str, Any]
) -> Dict[str, Any]:
	selector_results = selector_data.get("results", [])
	per_query_results: List[Dict[str, Any]] = []

	for row in selector_results:
		question_id = row.get("question_id")
		selected_model = row.get("selected_candidate_model")

		default_metrics = {field: 0.0 for field in METRIC_FIELDS}
		resolved_metrics = default_metrics.copy()
		defaulted_to_zero = True
		reason = "missing_selected_candidate"
		pairwise_key = None

		if selected_model is not None and question_id is not None:
			question_entry = pairwise_data.get(str(question_id), {})
			comparisons = question_entry.get("comparisons", {})
			pairwise_key, comparison = _get_ground_truth_comparison(
				comparisons, selected_model
			)

			if comparison is not None:
				resolved_metrics = _extract_metrics(comparison)
				defaulted_to_zero = False
				reason = "matched_ground_truth_comparison"
			else:
				reason = "ground_truth_comparison_not_found"

		per_query_results.append(
			{
				"question_id": question_id,
				"db_id": row.get("db_id"),
				"selected_candidate_model": selected_model,
				"defaulted_to_zero": defaulted_to_zero,
				"reason": reason,
				"source_comparison_key": pairwise_key,
				**resolved_metrics,
			}
		)

	aggregates: Dict[str, float] = {}
	for field in METRIC_FIELDS:
		values = [_safe_float(x.get(field, 0.0)) for x in per_query_results]
		aggregates[field] = mean(values) if values else 0.0

	return {
		"selector_model": selector_data.get("selector_model"),
		"database": selector_data.get("database"),
		"total_queries": len(per_query_results),
		"defaulted_to_zero_queries": sum(
			1 for x in per_query_results if x.get("defaulted_to_zero")
		),
		"metric_averages": aggregates,
		"per_query": per_query_results,
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compute selector performance against ground truth by taking, for each "
			"query, the ground_truth-vs-selected_model metrics from pairwise comparisons."
		)
	)
	parser.add_argument(
		"--selector-file",
		default="selectors/deepseek-chat_single_selector_choices.json",
		help="Path to selector choices JSON file.",
	)
	parser.add_argument(
		"--pairwise-file",
		default="all_pairwise_comparisons.json",
		help="Path to all pairwise comparisons JSON file.",
	)
	parser.add_argument(
		"--output-file",
		default="selectors/deepseek-chat_selector_performance_vs_ground_truth.json",
		help="Path to output JSON file.",
	)
	args = parser.parse_args()

	selector_path = Path(args.selector_file)
	pairwise_path = Path(args.pairwise_file)
	output_path = Path(args.output_file)

	selector_data = _load_json(selector_path)
	pairwise_data = _load_json(pairwise_path)

	result = compute_selector_performance(selector_data, pairwise_data)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(result, f, indent=2)

	print(f"Saved selector performance to: {output_path}")


if __name__ == "__main__":
	main()

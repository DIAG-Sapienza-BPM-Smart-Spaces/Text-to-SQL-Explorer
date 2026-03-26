from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common_utils import CANONICAL_METRICS, load_json, metric_to_percentage


def _pick_model_from_leaderboard(result_row: dict[str, Any], selector_model: str, random_tie_break: bool = True) -> str | None:
    leaderboard = result_row.get("leaderboard")
    if not isinstance(leaderboard, list) or not leaderboard:
        return result_row.get("selected_candidate_model")

    top_wins = max(int(x.get("wins", 0)) for x in leaderboard if isinstance(x, dict))
    top_models = [x.get("model") for x in leaderboard if isinstance(x, dict) and int(x.get("wins", 0)) == top_wins and x.get("model")]
    if not top_models:
        return result_row.get("selected_candidate_model")

    if len(top_models) == 1 or not random_tie_break:
        return sorted(top_models)[0]

    seed = f"{selector_model}|{result_row.get('db_id')}|{result_row.get('question_id')}"
    return random.Random(seed).choice(sorted(top_models))


def build_selector_metrics_from_pairwise(
    pairwise_selector_payload: dict[str, Any],
    metrics_lookup_payload: dict[str, Any],
    random_tie_break: bool = True,
) -> dict[str, Any]:
    selector_model = str(pairwise_selector_payload.get("selector_model", "deepseek-chat"))
    results = pairwise_selector_payload.get("results", [])

    per_query = []
    for row in results:
        if not isinstance(row, dict):
            continue
        selected_model = _pick_model_from_leaderboard(row, selector_model, random_tie_break=random_tie_break)
        if not selected_model:
            continue

        key = f"{row.get('db_id')}|{row.get('question_id')}|{selected_model}"
        metrics = metrics_lookup_payload.get(key, {})

        out = {
            "question_id": row.get("question_id"),
            "db_id": row.get("db_id"),
            "selected_candidate_model": selected_model,
        }
        for metric in CANONICAL_METRICS:
            out[metric] = metric_to_percentage(metrics.get(metric)) or 0.0
        per_query.append(out)

    aggregates = {}
    for metric in CANONICAL_METRICS:
        vals = [float(x.get(metric, 0.0)) for x in per_query]
        aggregates[metric] = (sum(vals) / len(vals)) if vals else 0.0

    return {
        "selector_model": selector_model,
        "mode": "pairwise_runtime_ranking",
        "total_queries": len(per_query),
        "metric_averages": aggregates,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selector metrics from pairwise-judge outputs and canonical metrics lookup.")
    parser.add_argument("--pairwise-selector-file", default="pairwise_results/deepseek-chat_pairwise_selector_results.json")
    parser.add_argument("--metrics-lookup-file", default="precomputed/metrics/bird_dev_metrics_lookup.json")
    parser.add_argument("--output-file", default="selectors/deepseek-chat_selector_runtime_metrics.json")
    args = parser.parse_args()

    pairwise_selector_payload = load_json(Path(args.pairwise_selector_file))
    metrics_lookup_payload = load_json(Path(args.metrics_lookup_file))
    out = build_selector_metrics_from_pairwise(pairwise_selector_payload, metrics_lookup_payload)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved runtime selector metrics to: {output_path}")


if __name__ == "__main__":
    main()

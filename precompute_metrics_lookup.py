from __future__ import annotations

import argparse
import json
from pathlib import Path

from common_utils import CANONICAL_METRICS, metric_to_percentage


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "metrics_results"
DEFAULT_OUTPUT_PATH = ROOT / "precomputed" / "metrics" / "bird_dev_metrics_lookup.json"


def run(input_dir: Path, output_path: Path) -> dict:
    lookup = {}
    for path in sorted(input_dir.glob("evaluation_sql_metrics_*_vs_ground_truth.json")):
        model = path.stem[len("evaluation_sql_metrics_"):-len("_vs_ground_truth")]
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            continue

        for row in payload:
            if not isinstance(row, dict):
                continue
            db_id = row.get("db_id")
            question_id = row.get("question_id")
            if db_id is None or question_id is None:
                continue

            key = f"{db_id}|{int(question_id)}|{model}"
            values = {}
            for metric in CANONICAL_METRICS:
                pct = metric_to_percentage(row.get(metric))
                if pct is not None:
                    values[metric] = pct
            if values:
                lookup[key] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)

    return {
        "entries": len(lookup),
        "output_file": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute compact canonical metrics lookup for runtime selector ranking.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    summary = run(Path(args.input_dir), Path(args.output_file))
    print(summary)


if __name__ == "__main__":
    main()

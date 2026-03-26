import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from common_utils import collect_models_from_metric_files, fast_hash_hex


ROOT = Path(__file__).resolve().parent

METRICS = [
    "execution_accuracy",
    "exact_match",
    "sql_f1_score",
    "response_schema_f1_score",
    "cell_f1_score",
]

DISCOVERED_MODELS = [m for m in collect_models_from_metric_files(ROOT / "metrics_results") if m != "ground_truth"]

CANDIDATE_MODELS = DISCOVERED_MODELS or [
    "cogito_70b",
    "deepseek-chat",
    "qwen2.5-coder_32b",
    "qwen3-coder_30b",
    "codellama_70b",
    "codestral_22b",
    "sqlcoder_15b",
]

SELECTOR_MODELS = list(CANDIDATE_MODELS)

JUDGE_MODELS = list(CANDIDATE_MODELS)

DATASET_SOURCES = [
    ("BIRD Training", "datasets_files/BIRD/train.json", "SQL"),
    ("BIRD Developer", "datasets_files/BIRD/dev.json", "SQL"),
    ("SPIDER Training", "datasets_files/SPIDER/train_spider.json", "query"),
    ("SPIDER Training", "datasets_files/SPIDER/train_others.json", "query"),
    ("SPIDER Dev", "datasets_files/SPIDER/dev.json", "query"),
    ("SPIDER Test", "datasets_files/SPIDER/test.json", "query"),
]

# True data is only available for BIRD Developer in current visualization pipeline.
FAKE_EXECUTION_EXCLUDED_DATASETS = {"BIRD Developer"}
FAKE_DEEPSEEK_SELECTOR_EXCLUDED_DATASETS = {"BIRD Developer"}
FAKE_PAIRWISE_SELECTOR_EXCLUDED_DATASETS = {"BIRD Developer"}


@dataclass
class QueryRow:
    dataset: str
    question_id: int
    db_id: str
    question: str
    sql_text: str


def stable_unit(*parts: object) -> float:
    raw = "|".join(str(p) for p in parts)
    digest = fast_hash_hex(raw, digest_size=16)
    return int(digest[:16], 16) / float(16**16 - 1)


def stable_between(low: float, high: float, *parts: object) -> float:
    return low + (high - low) * stable_unit(*parts)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def slugify(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_all_queries() -> List[QueryRow]:
    rows: List[QueryRow] = []

    for dataset_name, rel_path, sql_key in DATASET_SOURCES:
        payload = load_json(ROOT / rel_path)
        if not isinstance(payload, list):
            continue

        for idx, row in enumerate(payload):
            if not isinstance(row, dict):
                continue

            qid_raw = row.get("question_id", row.get("id", idx))
            try:
                question_id = int(qid_raw)
            except (TypeError, ValueError):
                question_id = idx

            db_id = str(row.get("db_id", row.get("database", "unknown_db")))
            question = str(row.get("question", row.get("query", "")))
            sql_text = str(row.get(sql_key, row.get("SQL", row.get("query", ""))))

            rows.append(
                QueryRow(
                    dataset=dataset_name,
                    question_id=question_id,
                    db_id=db_id,
                    question=question,
                    sql_text=sql_text,
                )
            )

    return rows


def sql_feature_score(sql_text: str) -> float:
    token_count = len(sql_text.split())
    joins = sql_text.lower().count(" join ")
    nesting = sql_text.count("(")

    # Convert complexity signals into a compact 0..1 difficulty proxy.
    normalized = min(1.0, (token_count / 120.0) + (joins * 0.12) + (nesting * 0.03))
    return clamp01(normalized)


def compute_fake_metrics(dataset: str, model: str, query: QueryRow) -> Dict[str, float]:
    model_base = {
        "deepseek-chat": 0.74,
        "qwen3-coder_30b": 0.71,
        "qwen2.5-coder_32b": 0.69,
        "cogito_70b": 0.67,
        "codestral_22b": 0.68,
        "sqlcoder_15b": 0.63,
        "codellama_70b": 0.64,
    }.get(model, 0.65)

    dataset_adjust = {
        "BIRD Training": -0.02,
        "SPIDER Training": -0.04,
        "SPIDER Dev": -0.03,
        "SPIDER Test": -0.05,
        "BIRD Developer": 0.00,
    }.get(dataset, -0.03)

    difficulty = sql_feature_score(query.sql_text)
    difficulty_penalty = 0.18 * difficulty

    base = model_base + dataset_adjust - difficulty_penalty

    metrics: Dict[str, float] = {}
    for metric in METRICS:
        metric_jitter = stable_between(-0.10, 0.10, dataset, model, query.db_id, query.question_id, metric)
        metric_bias = {
            "execution_accuracy": -0.05,
            "exact_match": -0.04,
            "sql_f1_score": 0.01,
            "response_schema_f1_score": 0.02,
            "cell_f1_score": -0.03,
        }[metric]
        metrics[metric] = round(clamp01(base + metric_bias + metric_jitter), 4)

    return metrics


def generate_fake_execution_data(queries: List[QueryRow]) -> List[dict]:
    out: List[dict] = []
    for q in queries:
        if q.dataset in FAKE_EXECUTION_EXCLUDED_DATASETS:
            continue
        for model in CANDIDATE_MODELS:
            metrics = compute_fake_metrics(q.dataset, model, q)
            out.append(
                {
                    "dataset": q.dataset,
                    "question_id": q.question_id,
                    "db_id": q.db_id,
                    "model": model,
                    "metrics": metrics,
                }
            )
    return out


def pick_model_for_selector(selector_model: str, dataset: str, query: QueryRow) -> str:
    scored = []
    for candidate in CANDIDATE_MODELS:
        metrics = compute_fake_metrics(dataset, candidate, query)
        quality = mean(metrics.values())

        # Mild selector-specific preferences to avoid identical selector outputs.
        selector_affinity = stable_between(-0.05, 0.05, "selector", selector_model, candidate)
        query_noise = stable_between(-0.07, 0.07, "selector_q", selector_model, dataset, query.db_id, query.question_id, candidate)

        score = quality + selector_affinity + query_noise
        scored.append((score, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def generate_fake_selector_data(queries: List[QueryRow]) -> List[dict]:
    rows: List[dict] = []

    for selector_model in SELECTOR_MODELS:
        for q in queries:
            if selector_model == "deepseek-chat" and q.dataset in FAKE_DEEPSEEK_SELECTOR_EXCLUDED_DATASETS:
                continue

            selected = pick_model_for_selector(selector_model, q.dataset, q)
            selected_metrics = compute_fake_metrics(q.dataset, selected, q)

            rows.append(
                {
                    "selector_model": selector_model,
                    "dataset": q.dataset,
                    "question_id": q.question_id,
                    "db_id": q.db_id,
                    "selected_candidate_model": selected,
                    **selected_metrics,
                }
            )

    return rows


def generate_fake_embedding_data(queries: List[QueryRow]) -> List[dict]:
    rows: List[dict] = []

    for q in queries:
        selected = pick_model_for_selector("embedding-selector", q.dataset, q)
        metrics = compute_fake_metrics(q.dataset, selected, q)

        rows.append(
            {
                "dataset": q.dataset,
                "question_id": q.question_id,
                "db_id": q.db_id,
                "selected_model": selected,
                "num_model_candidates": len(CANDIDATE_MODELS),
                "candidate_models": CANDIDATE_MODELS,
                "selection_statistics": {
                    "selected_index": CANDIDATE_MODELS.index(selected),
                    "num_candidates": len(CANDIDATE_MODELS),
                    "num_clusters": int(stable_between(2, 4.9999, "emb_clusters", q.dataset, q.db_id, q.question_id)),
                },
                "ground_truth_comparison_metrics": {
                    "comparison_key": f"ground_truth_vs_{selected}",
                    "system1": "ground_truth",
                    "system2": selected,
                    **metrics,
                    "comparison_performed": True,
                },
            }
        )

    return rows


def _pairwise_winner(metrics_a: Dict[str, float], metrics_b: Dict[str, float], tie_break_key: str) -> str:
    score_a = mean(metrics_a.values()) + stable_between(-0.02, 0.02, tie_break_key, "a")
    score_b = mean(metrics_b.values()) + stable_between(-0.02, 0.02, tie_break_key, "b")
    if abs(score_a - score_b) <= 0.005:
        return "tie"
    return "model_a" if score_a > score_b else "model_b"


def generate_fake_selector_pairwise_data(queries: List[QueryRow]) -> List[dict]:
    rows: List[dict] = []

    for judge_model in JUDGE_MODELS:
        for q in queries:
            if q.dataset in FAKE_PAIRWISE_SELECTOR_EXCLUDED_DATASETS:
                continue

            pairwise_rows = []
            for model_a, model_b in combinations(CANDIDATE_MODELS, 2):
                metrics_a = compute_fake_metrics(q.dataset, model_a, q)
                metrics_b = compute_fake_metrics(q.dataset, model_b, q)
                winner_token = _pairwise_winner(
                    metrics_a,
                    metrics_b,
                    tie_break_key=f"{judge_model}|{q.dataset}|{q.db_id}|{q.question_id}|{model_a}|{model_b}",
                )
                winner = "tie"
                if winner_token == "model_a":
                    winner = model_a
                elif winner_token == "model_b":
                    winner = model_b

                pairwise_rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "winner": winner,
                        "judge_model": judge_model,
                        "reasoning": f"Synthetic pairwise judgment for {model_a} vs {model_b}.",
                        "metrics_a": metrics_a,
                        "metrics_b": metrics_b,
                    }
                )

            rows.append(
                {
                    "dataset": q.dataset,
                    "question_id": q.question_id,
                    "db_id": q.db_id,
                    "judge_model": judge_model,
                    "candidate_models": CANDIDATE_MODELS,
                    "pairwise_judgments": pairwise_rows,
                }
            )

    return rows


def generate_fake_binary_data(queries: List[QueryRow]) -> List[dict]:
    rows: List[dict] = []

    for judge_model in JUDGE_MODELS:
        for candidate_model in CANDIDATE_MODELS:
            for q in queries:
                if q.dataset in FAKE_EXECUTION_EXCLUDED_DATASETS:
                    continue

                metrics = compute_fake_metrics(q.dataset, candidate_model, q)
                accept_score = 0.60 * metrics["execution_accuracy"] + 0.40 * metrics["sql_f1_score"]
                threshold = stable_between(0.45, 0.72, "binary_threshold", judge_model, q.dataset, q.db_id, q.question_id)
                choice = "ACCEPT" if accept_score >= threshold else "REJECT"

                rows.append(
                    {
                        "judge_model": judge_model,
                        "candidate_model": candidate_model,
                        "dataset": q.dataset,
                        "question_id": q.question_id,
                        "db_id": q.db_id,
                        "choice": choice,
                        "execution_vs_ground_truth": {
                            "system1": "ground_truth",
                            "system2": candidate_model,
                            **metrics,
                            "comparison_performed": True,
                        },
                    }
                )

    return rows


def write_outputs(
    output_root: Path,
    execution_rows: List[dict],
    selector_rows: List[dict],
    embedding_rows: List[dict],
    binary_rows: List[dict],
    selector_pairwise_rows: List[dict],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = {
        "meta": {
            "description": "Deterministic fake data for missing visualization scenarios.",
            "notes": [
                "Execution fake data excludes BIRD Developer because true data already exists.",
                "DeepSeek single-selector fake data excludes BIRD Developer for the same reason.",
                "Embedding fake data covers all datasets.",
            ],
        },
        "execution_fake": execution_rows,
        "selector_fake": selector_rows,
        "selector_pairwise_fake": selector_pairwise_rows,
        "embedding_fake": embedding_rows,
        "binary_fake": binary_rows,
    }

    dump_json(output_root / "fake_generation_bundle.json", bundle)
    dump_json(output_root / "fake_execution_metrics.json", execution_rows)
    dump_json(output_root / "fake_selector_pairwise_results.json", selector_pairwise_rows)
    dump_json(output_root / "fake_embedding_selection.json", embedding_rows)
    dump_json(output_root / "fake_binary_choices.json", binary_rows)

    for dataset in {r["dataset"] for r in embedding_rows}:
        ds_rows = [r for r in embedding_rows if r["dataset"] == dataset]
        dump_json(output_root / "embedding" / f"embedding_selector_{slugify(dataset)}_fake.json", ds_rows)


def generate_all_fake_data(output_dir: str = "fake_data") -> dict:
    queries = load_all_queries()

    execution_rows = generate_fake_execution_data(queries)
    selector_rows = generate_fake_selector_data(queries)
    selector_pairwise_rows = generate_fake_selector_pairwise_data(queries)
    embedding_rows = generate_fake_embedding_data(queries)
    binary_rows = generate_fake_binary_data(queries)

    output_root = ROOT / output_dir
    write_outputs(
        output_root,
        execution_rows,
        selector_rows,
        embedding_rows,
        binary_rows,
        selector_pairwise_rows,
    )

    return {
        "queries_loaded": len(queries),
        "execution_rows": len(execution_rows),
        "selector_rows": len(selector_rows),
        "selector_pairwise_rows": len(selector_pairwise_rows),
        "embedding_rows": len(embedding_rows),
        "binary_rows": len(binary_rows),
        "output_dir": str(output_root),
    }


def main() -> None:
    summary = generate_all_fake_data()
    print("Fake data generation complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

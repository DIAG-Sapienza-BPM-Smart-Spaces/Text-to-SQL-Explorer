import json
import os
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import numpy as np

from common_utils import collect_models_from_metric_files, fast_hash_hex
from embedding import (
    compute_average_and_std_of_similarities,
    save_json_artifact,
    save_similarity_matrix_artifact,
)


ROOT = Path(__file__).resolve().parent
_CHECKPOINT_LOCK = threading.Lock()

# Canonical metric keys used by visualization and fake generators.
METRICS = [
    "execution_accuracy",
    "exact_match",
    "sql_f1_score",
    "response_schema_f1_score",
    "cell_f1_score",
]

DISCOVERED_MODELS = [m for m in collect_models_from_metric_files(ROOT / "metrics_results") if m != "ground_truth"]

# Prefer auto-discovered models; fallback keeps local runs robust.
CANDIDATE_MODELS = DISCOVERED_MODELS or [
    "cogito_70b",
    "deepseek-chat",
    "qwen2.5-coder_32b",
    "qwen3-coder_30b",
    "codestral_22b",
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
    # Deterministic pseudo-random unit value in [0, 1], stable across runs.
    raw = "|".join(str(p) for p in parts)
    digest = fast_hash_hex(raw, digest_size=16)
    return int(digest[:16], 16) / float(16**16 - 1)


def stable_between(low: float, high: float, *parts: object) -> float:
    # Deterministic value in [low, high] keyed by input parts.
    return low + (high - low) * stable_unit(*parts)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def slugify(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")


def _dataset_to_similarity_token(dataset_name: str) -> str:
    mapping = {
        "BIRD Developer": "bird_dev",
        "BIRD Training": "bird_training",
        "SPIDER Dev": "spider_dev",
        "SPIDER Training": "spider_training",
        "SPIDER Test": "spider_test",
    }
    return mapping.get(dataset_name, slugify(str(dataset_name or "")))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def log_message(verbose: int, level: int, message: str) -> None:
    if verbose >= level:
        print(message)


def load_all_queries() -> List[QueryRow]:
    # Flatten all configured dataset splits into one uniform query list.
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
    # Base quality prior per model, then adjust by split and query difficulty.
    model_base = {
        "deepseek-chat": 0.74,
        "qwen3-coder_30b": 0.71,
        "qwen2.5-coder_32b": 0.69,
        "cogito_70b": 0.67,
        "codestral_22b": 0.68,
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
    # Generate per-model metric rows for non-BIRD-Developer datasets.
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
    # Pick top candidate by synthetic quality + deterministic selector/query noise.
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
    # Produce one selected model per (selector_model, query).
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
    # Simulate embedding selector outputs with selection metadata payload.
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


def _build_fake_similarity_matrix(dataset: str, query: QueryRow, models: List[str]) -> np.ndarray:
    """Build a deterministic symmetric similarity matrix for one query."""
    n = len(models)
    matrix = np.eye(n, dtype=np.float32)

    # Reuse synthetic metric quality as a signal that drives pairwise similarity.
    quality_by_model = {
        m: mean(compute_fake_metrics(dataset, m, query).values())
        for m in models
    }

    for i in range(n):
        for j in range(i + 1, n):
            m_i = models[i]
            m_j = models[j]
            quality_delta = abs(quality_by_model[m_i] - quality_by_model[m_j])

            # Higher quality agreement -> higher similarity, plus bounded deterministic jitter.
            base = 0.80 - 0.55 * quality_delta
            jitter = stable_between(
                -0.08,
                0.08,
                "fake_similarity",
                dataset,
                query.db_id,
                query.question_id,
                m_i,
                m_j,
            )
            sim = clamp01(base + jitter)
            matrix[i, j] = np.float32(sim)
            matrix[j, i] = np.float32(sim)

    return matrix


def generate_fake_similarity_assets(
    queries: List[QueryRow],
    output_root: Path,
    max_workers: Optional[int] = None,
    checkpoint_every: int = 100,
    verbose: int = 1,
) -> dict:
    """Generate fake per-query similarity matrices and a true-schema index."""
    similarity_dir = output_root / "embedding" / "similarity"
    similarity_dir.mkdir(parents=True, exist_ok=True)

    ordered_models = sorted(CANDIDATE_MODELS)
    index_payload: Dict[str, object] = {"dataset": "all_fake", "queries": {}}
    queries_index: Dict[str, dict] = index_payload["queries"]  # type: ignore[assignment]

    workers = max(1, int(max_workers or min(8, (os.cpu_count() or 4))))
    checkpoint_stride = max(1, int(checkpoint_every or 1))
    log_message(
        verbose,
        1,
        f"[fake-similarity] Starting generation for {len(queries)} queries with workers={workers}, checkpoint_every={checkpoint_stride}",
    )

    def _build_one(q: QueryRow) -> tuple[str, dict]:
        dataset_token = _dataset_to_similarity_token(q.dataset)
        query_key = f"{dataset_token}|{q.db_id}|{q.question_id}"

        matrix = _build_fake_similarity_matrix(q.dataset, q, ordered_models)
        avg, std = compute_average_and_std_of_similarities(matrix)

        matrix_file = similarity_dir / f"similarity_{dataset_token}_{slugify(q.db_id)}_{q.question_id}.npz"
        save_similarity_matrix_artifact(matrix_file, matrix)

        entry = {
            "dataset": dataset_token,
            "db_id": q.db_id,
            "question_id": int(q.question_id),
            "models": ordered_models,
            # Fake sql ids are positional placeholders; visualization only needs alignment with `models`.
            "sql_ids": list(range(len(ordered_models))),
            "matrix_file": str(matrix_file.relative_to(ROOT)),
            "similarity_mean": float(avg),
            "similarity_std": float(std),
            "default_threshold": float(avg),
        }
        return query_key, entry

    generated = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_one, q) for q in queries]
        for future in as_completed(futures):
            query_key, entry = future.result()
            queries_index[query_key] = entry
            generated += 1

            if verbose >= 2 and (generated % 50 == 0 or generated == len(queries)):
                log_message(verbose, 2, f"[fake-similarity] Completed {generated}/{len(queries)} matrices")

            if generated % checkpoint_stride == 0:
                # Checkpoint index so interruptions don't lose all progress.
                with _CHECKPOINT_LOCK:
                    save_json_artifact(similarity_dir / "similarity_index.json", index_payload)
                    save_json_artifact(similarity_dir / "bird_dev_similarity_index.json", index_payload)
                log_message(verbose, 1, f"[fake-similarity] Checkpoint saved at {generated} matrices")

    # Provide both generic and bird_dev index file names for compatibility with loaders.
    save_json_artifact(similarity_dir / "similarity_index.json", index_payload)
    save_json_artifact(similarity_dir / "bird_dev_similarity_index.json", index_payload)
    log_message(verbose, 1, f"[fake-similarity] Final index saved with {generated} entries")

    return {
        "num_similarity_matrices": generated,
        "similarity_index_file": str((similarity_dir / "similarity_index.json").relative_to(ROOT)),
        "bird_dev_similarity_index_file": str((similarity_dir / "bird_dev_similarity_index.json").relative_to(ROOT)),
    }


def _pairwise_winner(metrics_a: Dict[str, float], metrics_b: Dict[str, float], tie_break_key: str) -> str:
    # Decide synthetic pairwise winner using average metric score + tiny tie-break noise.
    score_a = mean(metrics_a.values()) + stable_between(-0.02, 0.02, tie_break_key, "a")
    score_b = mean(metrics_b.values()) + stable_between(-0.02, 0.02, tie_break_key, "b")
    if abs(score_a - score_b) <= 0.005:
        return "tie"
    return "model_a" if score_a > score_b else "model_b"


def generate_fake_selector_pairwise_data(queries: List[QueryRow]) -> List[dict]:
    # Build all unique model-vs-model judgments for each (judge_model, query).
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
    # Generate ACCEPT/REJECT style rows to mirror binary judge output format.
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
    similarity_summary: dict,
    verbose: int = 1,
) -> None:
    # Write one bundle plus per-purpose files consumed by visualization loaders.
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
        "similarity_fake": similarity_summary,
    }

    dump_json(output_root / "fake_generation_bundle.json", bundle)
    dump_json(output_root / "fake_execution_metrics.json", execution_rows)
    dump_json(output_root / "fake_selector_pairwise_results.json", selector_pairwise_rows)
    dump_json(output_root / "fake_embedding_selection.json", embedding_rows)
    dump_json(output_root / "fake_binary_choices.json", binary_rows)
    log_message(verbose, 1, f"[fake-data] Wrote bundle + base fake files to {output_root}")

    for dataset in {r["dataset"] for r in embedding_rows}:
        ds_rows = [r for r in embedding_rows if r["dataset"] == dataset]
        dump_json(output_root / "embedding" / f"embedding_selector_{slugify(dataset)}_fake.json", ds_rows)

    log_message(verbose, 1, f"[fake-data] Wrote {len({r['dataset'] for r in embedding_rows})} dataset embedding selector files")


def generate_all_fake_data(
    output_dir: str = "fake_data",
    max_workers: Optional[int] = None,
    checkpoint_every: int = 100,
    verbose: int = 1,
) -> dict:
    # End-to-end orchestration: load queries, synthesize payloads, persist outputs.
    log_message(verbose, 1, "[fake-data] Loading input queries...")
    queries = load_all_queries()
    log_message(verbose, 1, f"[fake-data] Loaded {len(queries)} queries")

    log_message(verbose, 1, "[fake-data] Generating execution fake rows...")
    execution_rows = generate_fake_execution_data(queries)
    log_message(verbose, 1, "[fake-data] Generating selector fake rows...")
    selector_rows = generate_fake_selector_data(queries)
    log_message(verbose, 1, "[fake-data] Generating selector pairwise fake rows...")
    selector_pairwise_rows = generate_fake_selector_pairwise_data(queries)
    log_message(verbose, 1, "[fake-data] Generating embedding fake rows...")
    embedding_rows = generate_fake_embedding_data(queries)
    log_message(verbose, 1, "[fake-data] Generating binary fake rows...")
    binary_rows = generate_fake_binary_data(queries)

    output_root = ROOT / output_dir
    similarity_summary = generate_fake_similarity_assets(
        queries,
        output_root,
        max_workers=max_workers,
        checkpoint_every=checkpoint_every,
        verbose=verbose,
    )
    write_outputs(
        output_root,
        execution_rows,
        selector_rows,
        embedding_rows,
        binary_rows,
        selector_pairwise_rows,
        similarity_summary,
        verbose=verbose,
    )

    log_message(
        verbose,
        1,
        (
            "[fake-data] Generation complete "
            f"(execution={len(execution_rows)}, selector={len(selector_rows)}, "
            f"selector_pairwise={len(selector_pairwise_rows)}, embedding={len(embedding_rows)}, "
            f"binary={len(binary_rows)}, similarity={int(similarity_summary.get('num_similarity_matrices', 0))})"
        ),
    )

    return {
        "queries_loaded": len(queries),
        "execution_rows": len(execution_rows),
        "selector_rows": len(selector_rows),
        "selector_pairwise_rows": len(selector_pairwise_rows),
        "embedding_rows": len(embedding_rows),
        "binary_rows": len(binary_rows),
        "similarity_matrices": int(similarity_summary.get("num_similarity_matrices", 0)),
        "similarity_index_file": similarity_summary.get("similarity_index_file"),
        "output_dir": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic fake visualization data artifacts.")
    parser.add_argument("--output-dir", default="fake_data", help="Output directory for generated fake artifacts")
    parser.add_argument("--max-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))), help="Worker threads for fake similarity matrix generation")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save similarity index checkpoint every N completed matrices")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    summary = generate_all_fake_data(
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        checkpoint_every=args.checkpoint_every,
        verbose=args.verbose,
    )
    print("Fake data generation complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

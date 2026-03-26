from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from embedding import (
    compute_average_and_std_of_similarities,
    load_embeddings_artifact,
    save_json_artifact,
    save_similarity_matrix_artifact,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_DIR = ROOT / "precomputed" / "embeddings"
DEFAULT_OUT_DIR = ROOT / "precomputed" / "similarity"


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0).astype(np.float32)


def run_precompute(
    embedding_dir: Path,
    out_dir: Path,
    dataset_filter: str = "BIRD Developer",
    verbose: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings_artifact(embedding_dir / "sql_embeddings.npz")
    vectors = embeddings["vectors"]

    with (embedding_dir / "sql_embeddings_meta.json").open("r", encoding="utf-8") as f:
        meta = __import__("json").load(f)

    query_candidates = meta.get("query_candidates", {})

    grouped = {}
    for key, model_to_sql_id in query_candidates.items():
        if not isinstance(model_to_sql_id, dict):
            continue

        db_id, qid = key.split("|", 1)
        row = {
            "db_id": db_id,
            "question_id": int(qid),
            "model_to_sql_id": model_to_sql_id,
        }
        # Kept simple: BIRD Developer is inferred from BIRD dev DB ids available in sqls artifacts.
        # If additional dataset markers become available, this filter can be upgraded.
        if dataset_filter == "BIRD Developer":
            grouped[key] = row
        else:
            grouped[key] = row

    matrix_meta = {
        "dataset": dataset_filter,
        "queries": {},
    }

    for key, row in grouped.items():
        model_to_sql_id = row["model_to_sql_id"]
        ordered_models = sorted(model_to_sql_id.keys())
        sql_ids = [int(model_to_sql_id[m]) for m in ordered_models]
        sub_vectors = vectors[sql_ids]
        sim = cosine_matrix(sub_vectors)
        avg, std = compute_average_and_std_of_similarities(sim)

        matrix_file = out_dir / f"similarity_{row['db_id']}_{row['question_id']}.npz"
        save_similarity_matrix_artifact(matrix_file, sim)

        matrix_meta["queries"][key] = {
            "db_id": row["db_id"],
            "question_id": row["question_id"],
            "models": ordered_models,
            "sql_ids": sql_ids,
            "matrix_file": str(matrix_file.relative_to(ROOT)),
            "similarity_mean": float(avg),
            "similarity_std": float(std),
            "default_threshold": float(avg),
        }

    save_json_artifact(out_dir / "bird_dev_similarity_index.json", matrix_meta)

    if verbose >= 1:
        print(f"[similarity] Saved {len(matrix_meta['queries'])} similarity matrices to {out_dir}")

    return {
        "dataset": dataset_filter,
        "num_queries": len(matrix_meta["queries"]),
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute per-query similarity matrices from precomputed embeddings.")
    parser.add_argument("--embedding-dir", default=str(DEFAULT_EMBEDDING_DIR), help="Directory produced by precompute_embeddings.py")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for per-query similarity matrices")
    parser.add_argument("--dataset", default="BIRD Developer", help="Dataset label used in metadata")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    summary = run_precompute(
        embedding_dir=Path(args.embedding_dir),
        out_dir=Path(args.out_dir),
        dataset_filter=args.dataset,
        verbose=args.verbose,
    )
    print(summary)


if __name__ == "__main__":
    main()

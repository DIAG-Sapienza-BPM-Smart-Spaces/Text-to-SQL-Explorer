from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from embedding import (
    compute_average_and_std_of_similarities,
    load_embeddings_artifact,
    load_json_artifact,
    save_json_artifact,
    save_similarity_matrix_artifact,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_DIR = ROOT / "precomputed" / "embeddings"
DEFAULT_OUT_DIR = ROOT / "precomputed" / "similarity"


def _normalized_dataset_token(value: str) -> str:
    """Normalize dataset names so filtering is robust across naming variants."""
    return str(value).strip().lower().replace(" ", "_")


def _dataset_matches(dataset_value: str, dataset_filter: str) -> bool:
    """Return True when a row dataset satisfies the user filter."""
    token = _normalized_dataset_token(dataset_filter)
    if token in ("", "all", "*"):
        return True
    return _normalized_dataset_token(dataset_value) == token


def _safe_filename_token(value: str) -> str:
    """Sanitize arbitrary text so it is safe in file names."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token or "unknown"


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix with zero-norm safety."""
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0).astype(np.float32)


def _build_and_save_query_matrix(
    *,
    key: str,
    row: dict[str, Any],
    vectors: np.ndarray,
    out_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Build one per-query similarity matrix, save it, and return index metadata."""
    model_to_sql_id = row["model_to_sql_id"]
    ordered_models = sorted(model_to_sql_id.keys())
    sql_ids = [int(model_to_sql_id[m]) for m in ordered_models]
    sub_vectors = vectors[sql_ids]
    sim = cosine_matrix(sub_vectors)
    avg, std = compute_average_and_std_of_similarities(sim)

    matrix_file = out_dir / (
        f"similarity_{_safe_filename_token(row['dataset'])}_"
        f"{_safe_filename_token(row['db_id'])}_{row['question_id']}.npz"
    )
    save_similarity_matrix_artifact(matrix_file, sim)

    entry = {
        "dataset": row["dataset"],
        "db_id": row["db_id"],
        "question_id": row["question_id"],
        "models": ordered_models,
        "sql_ids": sql_ids,
        "matrix_file": str(matrix_file.relative_to(ROOT)),
        "similarity_mean": float(avg),
        "similarity_std": float(std),
        "default_threshold": float(avg),
    }
    return key, entry


def run_precompute(
    embedding_dir: Path,
    out_dir: Path,
    dataset_filter: str = "bird_dev",
    verbose: int = 1,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Build per-query similarity matrices from precomputed embeddings + lookup."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose >= 1:
        print(
            "[similarity] Starting precompute "
            f"(dataset={dataset_filter}, embedding_dir={embedding_dir}, out_dir={out_dir}, workers={max_workers})"
        )

    # Intermediate input 1: full embeddings matrix produced by precompute_embeddings.py
    embeddings = load_embeddings_artifact(embedding_dir / "sql_embeddings.npz")
    vectors = embeddings["vectors"]
    if verbose >= 2:
        print(f"[similarity] Loaded embeddings matrix with shape={tuple(vectors.shape)}")

    # Intermediate input 2: key->embedding-index mapping
    lookup = load_json_artifact(embedding_dir / "sql_embeddings_lookup.json")
    if verbose >= 2:
        print(f"[similarity] Loaded lookup entries={len(lookup)}")

    # Group rows by query identity (dataset, db_id, question_id).
    grouped: dict[str, dict[str, Any]] = {}
    # Track skipped rows to make input quality issues visible.
    skipped_parse = 0
    skipped_dataset = 0
    skipped_bounds = 0

    for composite_key, idx_raw in lookup.items():
        if not isinstance(composite_key, str):
            skipped_parse += 1
            continue
        try:
            model, dataset, db_id, qid_raw = composite_key.split("|", 3)
            qid = int(qid_raw)
            sql_idx = int(idx_raw)
        except (ValueError, TypeError):
            skipped_parse += 1
            continue

        if not _dataset_matches(dataset, dataset_filter):
            skipped_dataset += 1
            continue
        if sql_idx < 0 or sql_idx >= len(vectors):
            skipped_bounds += 1
            continue

        grouped_key = f"{dataset}|{db_id}|{qid}"
        if grouped_key not in grouped:
            grouped[grouped_key] = {
                "dataset": dataset,
                "db_id": db_id,
                "question_id": qid,
                "model_to_sql_id": {},
            }
        grouped[grouped_key]["model_to_sql_id"][model] = sql_idx

    if verbose >= 1:
        print(
            "[similarity] Grouped queries="
            f"{len(grouped)} (skipped_parse={skipped_parse}, skipped_dataset={skipped_dataset}, skipped_bounds={skipped_bounds})"
        )

    matrix_meta = {
        "dataset": dataset_filter,
        "queries": {},
    }

    # Multithread by query: each task builds/saves one matrix and returns its metadata entry.
    workers = max(1, int(max_workers or 1))
    if grouped:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _build_and_save_query_matrix,
                    key=key,
                    row=row,
                    vectors=vectors,
                    out_dir=out_dir,
                )
                for key, row in grouped.items()
            ]

            completed = 0
            total = len(futures)
            for future in concurrent.futures.as_completed(futures):
                key, entry = future.result()
                matrix_meta["queries"][key] = entry
                completed += 1
                if verbose >= 2 and (completed % 50 == 0 or completed == total):
                    print(f"[similarity] Completed {completed}/{total} query matrices")

    elif verbose >= 1:
        print("[similarity] No grouped queries matched the requested dataset filter")

    # Final output index that points to all generated matrix files.
    dataset_token = _safe_filename_token(_normalized_dataset_token(dataset_filter))
    index_name = "similarity_index.json" if dataset_token in ("", "all") else f"{dataset_token}_similarity_index.json"
    save_json_artifact(out_dir / index_name, matrix_meta)
    if verbose >= 1:
        print(f"[similarity] Saved similarity index: {out_dir / index_name}")

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
    parser.add_argument("--dataset", default="bird_dev", help="Dataset filter used in grouping and metadata")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--max-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))), help="Worker threads for per-query matrix generation")
    args = parser.parse_args()

    summary = run_precompute(
        embedding_dir=Path(args.embedding_dir),
        out_dir=Path(args.out_dir),
        dataset_filter=args.dataset,
        verbose=args.verbose,
        max_workers=args.max_workers,
    )
    print(summary)


if __name__ == "__main__":
    main()

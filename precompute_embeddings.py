from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from common_utils import atomic_dump_json, fast_hash_hex, load_json
from embedding import (
    load_embedder,
    load_embeddings_artifact,
    load_json_artifact,
    parse_and_normalize_sql,
    save_embeddings_artifact,
    save_json_artifact,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SQL_DIR = ROOT / "candidates"
DEFAULT_OUT_DIR = ROOT / "precomputed" / "embeddings"
_CHECKPOINT_LOCK = threading.RLock()


def _model_from_file(path: Path) -> str:
    stem = path.stem
    if stem.startswith("evaluation_sql_metrics_") and stem.endswith("_vs_ground_truth"):
        return stem[len("evaluation_sql_metrics_") : -len("_vs_ground_truth")]
    if stem.endswith("_query_results"):
        return stem[: -len("_query_results")]
    return stem


def _extract_model_from_row(row: dict[str, Any]) -> str | None:
    for key in ("model", "model_id", "system", "system_id", "source_model"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_dataset_from_row(row: dict[str, Any]) -> str | None:
    for key in ("dataset", "dataset_id", "dataset_name", "split"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_sql(row: dict[str, Any]) -> str:
    for key in ("clean_sql", "extracted_sql", "generated_sql"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _iter_candidate_files(sql_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sql_dir.glob("*.json")
        if path.is_file()
    )


def _checkpoint_paths(out_dir: Path) -> tuple[Path, Path]:
    return (
        out_dir / "sql_embeddings_checkpoint.npz",
        out_dir / "sql_embeddings_checkpoint_meta.json",
    )


def _rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    tokens = [
        f"{row['model']}|{row['dataset']}|{row['db_id']}|{row['question_id']}|{row['sql']}"
        for row in rows
    ]
    return fast_hash_hex("\n".join(tokens), digest_size=16)


def _save_checkpoint(
    out_dir: Path,
    *,
    embedder_id: str,
    rows: list[dict[str, Any]],
    vectors: list[np.ndarray | None],
) -> int:
    checkpoint_npz, checkpoint_meta = _checkpoint_paths(out_dir)

    completed_indices = [i for i, vec in enumerate(vectors) if vec is not None]
    if not completed_indices:
        return 0

    completed_ids = np.array(completed_indices, dtype=np.int64)
    completed_vectors = np.vstack([vectors[i] for i in completed_indices]).astype(np.float32)

    with _CHECKPOINT_LOCK:
        checkpoint_npz.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = checkpoint_npz.with_suffix(checkpoint_npz.suffix + ".tmp")
        with tmp_npz.open("wb") as f:
            np.savez_compressed(f, sql_ids=completed_ids, vectors=completed_vectors)
        tmp_npz.replace(checkpoint_npz)

        atomic_dump_json(
            checkpoint_meta,
            {
                "embedder_id": embedder_id,
                "num_rows": len(rows),
                "rows_fingerprint": _rows_fingerprint(rows),
                "num_completed": len(completed_indices),
            },
            lock=_CHECKPOINT_LOCK,
        )

    return len(completed_indices)


def _load_checkpoint(
    out_dir: Path,
    *,
    embedder_id: str,
    rows: list[dict[str, Any]],
    vectors: list[np.ndarray | None],
) -> int:
    checkpoint_npz, checkpoint_meta = _checkpoint_paths(out_dir)
    if not checkpoint_npz.exists() or not checkpoint_meta.exists():
        return 0

    try:
        meta = load_json_artifact(checkpoint_meta)
        if meta.get("embedder_id") != embedder_id:
            return 0
        if int(meta.get("num_rows", -1)) != len(rows):
            return 0
        if meta.get("rows_fingerprint") != _rows_fingerprint(rows):
            return 0

        payload = load_embeddings_artifact(checkpoint_npz)
        sql_ids = payload.get("sql_ids", np.array([], dtype=np.int64))
        vecs = payload.get("vectors", np.array([], dtype=np.float32))
        if len(sql_ids) != len(vecs):
            return 0

        restored = 0
        for pos, idx_raw in enumerate(sql_ids):
            idx = int(idx_raw)
            if idx < 0 or idx >= len(rows):
                continue
            vectors[idx] = np.array(vecs[pos], dtype=np.float32)
            restored += 1
        return restored
    except Exception:
        return 0


def collect_sql_rows(sql_dir: Path, default_dataset: str) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    files = _iter_candidate_files(sql_dir)
    for path in files:
        payload = load_json(path)
        if not isinstance(payload, list):
            continue

        default_model = _model_from_file(path)
        if default_model == "ground_truth":
            continue

        for row in payload:
            if not isinstance(row, dict):
                continue

            model = _extract_model_from_row(row) or default_model
            if model == "ground_truth":
                continue

            db_id = row.get("db_id")
            qid = row.get("question_id")
            if db_id is None or qid is None:
                continue
            try:
                qid_int = int(qid)
            except (TypeError, ValueError):
                continue

            sql = _extract_sql(row)
            if not sql:
                continue

            dataset = _extract_dataset_from_row(row) or default_dataset
            rows_out.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "db_id": str(db_id),
                    "question_id": qid_int,
                    "sql": sql,
                }
            )

    return rows_out


def _encode_single_sql(sql: str, embedder) -> np.ndarray:
    normalized = parse_and_normalize_sql(sql, verbose=0)
    return np.array(embedder.encode(normalized), dtype=np.float32)


def run_precompute(
    sql_dir: Path,
    out_dir: Path,
    embedder_id: str,
    verbose: int = 1,
    max_workers: int = 4,
    default_dataset: str = "bird_dev",
    checkpoint_every: int = 250,
    resume: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_sql_rows(sql_dir, default_dataset=default_dataset)
    sql_queries = [row["sql"] for row in rows]
    if verbose >= 1:
        print(f"[precompute] SQL rows count: {len(sql_queries)}")

    embedder = load_embedder(embedder_id=embedder_id, verbose=verbose)

    workers = max(1, int(max_workers or 1))
    vectors: list[np.ndarray | None] = [None] * len(sql_queries)
    restored = 0

    if resume and rows:
        restored = _load_checkpoint(
            out_dir,
            embedder_id=embedder_id,
            rows=rows,
            vectors=vectors,
        )
        if verbose >= 1 and restored > 0:
            print(f"[precompute] Restored {restored}/{len(sql_queries)} embeddings from checkpoint")

    pending_indices = [idx for idx, vec in enumerate(vectors) if vec is None]
    if pending_indices:
        completed_since_checkpoint = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_encode_single_sql, sql_queries[idx], embedder): idx
                for idx in pending_indices
            }
            for done_count, future in enumerate(concurrent.futures.as_completed(future_to_idx), start=1):
                idx = future_to_idx[future]
                try:
                    vectors[idx] = future.result()
                except Exception as exc:
                    if checkpoint_every > 0:
                        _save_checkpoint(
                            out_dir,
                            embedder_id=embedder_id,
                            rows=rows,
                            vectors=vectors,
                        )
                    raise RuntimeError(f"Embedding failed at row index {idx}") from exc

                completed_since_checkpoint += 1
                if verbose >= 2 and done_count % 100 == 0:
                    print(f"[precompute] Encoded {restored + done_count}/{len(sql_queries)}")

                if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
                    saved = _save_checkpoint(
                        out_dir,
                        embedder_id=embedder_id,
                        rows=rows,
                        vectors=vectors,
                    )
                    completed_since_checkpoint = 0
                    if verbose >= 1:
                        print(f"[precompute] Checkpoint saved: {saved}/{len(sql_queries)}")

    if checkpoint_every > 0 and rows:
        saved = _save_checkpoint(
            out_dir,
            embedder_id=embedder_id,
            rows=rows,
            vectors=vectors,
        )
        if verbose >= 1:
            print(f"[precompute] Final checkpoint saved: {saved}/{len(sql_queries)}")

    if any(vec is None for vec in vectors):
        raise RuntimeError("Embedding precompute did not finish all rows.")

    finalized_vectors = [vec for vec in vectors if vec is not None]

    if finalized_vectors:
        stacked = np.vstack(finalized_vectors).astype(np.float32)
    else:
        stacked = np.zeros((0, 0), dtype=np.float32)

    save_embeddings_artifact(
        out_dir / "sql_embeddings.npz",
        {
            "sql_ids": np.arange(len(rows), dtype=np.int64),
            "vectors": stacked,
        },
    )

    # Key format: model|dataset|db_id|question_id -> embedding matrix row index.
    lookup: dict[str, int] = {}
    for idx, row in enumerate(rows):
        key = f"{row['model']}|{row['dataset']}|{row['db_id']}|{row['question_id']}"
        lookup[key] = idx

    save_json_artifact(out_dir / "sql_embeddings_lookup.json", lookup)

    metadata = {
        "embedder_id": embedder_id,
        "num_sql": len(rows),
        "vector_dim": int(stacked.shape[1]) if stacked.size else 0,
        "num_lookup_keys": len(lookup),
    }
    save_json_artifact(out_dir / "sql_embeddings_meta.json", metadata)

    return {
        "num_sql": len(rows),
        "vector_dim": metadata["vector_dim"],
        "num_lookup_keys": len(lookup),
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reusable SQL embeddings for selector/visualization strategies.")
    parser.add_argument("--sql-dir", default=str(DEFAULT_SQL_DIR), help="Directory with candidate JSON files containing db_id/question_id and SQL fields")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for precomputed embedding artifacts")
    parser.add_argument("--embedder-id", default="s2593817/sft-sql-embedding", help="SentenceTransformer model id")
    parser.add_argument("--dataset", default="bird_dev", help="Fallback dataset label used when dataset field is missing")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--max-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))), help="Thread workers for embedding")
    parser.add_argument("--checkpoint-every", type=int, default=250, help="Save checkpoint every N newly encoded rows (<=0 disables)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint artifacts and recompute from scratch")
    args = parser.parse_args()

    summary = run_precompute(
        sql_dir=Path(args.sql_dir),
        out_dir=Path(args.out_dir),
        embedder_id=args.embedder_id,
        verbose=args.verbose,
        max_workers=args.max_workers,
        default_dataset=args.dataset,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(summary)


if __name__ == "__main__":
    main()

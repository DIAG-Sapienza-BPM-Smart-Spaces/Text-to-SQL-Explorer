from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from common_utils import load_json
from embedding import (
    load_embedder,
    parse_and_normalize_sql,
    save_embeddings_artifact,
    save_json_artifact,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SQL_DIR = ROOT / "candidates"
DEFAULT_OUT_DIR = ROOT / "precomputed" / "embeddings"


def _model_from_file(path: Path) -> str:
    suffix = "_query_results.json"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def _extract_sql(row: dict[str, Any]) -> str:
    for key in ("clean_sql", "extracted_sql", "generated_sql"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def collect_unique_sql_candidates(sql_dir: Path) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    sql_to_id: dict[str, int] = {}
    id_to_sql: list[str] = []
    query_candidates: dict[str, Any] = {}

    files = sorted(sql_dir.glob("*_query_results.json"))
    for path in files:
        model = _model_from_file(path)
        if model == "ground_truth":
            continue

        payload = load_json(path)
        if not isinstance(payload, list):
            continue

        for row in payload:
            if not isinstance(row, dict):
                continue
            db_id = row.get("db_id")
            qid = row.get("question_id")
            if db_id is None or qid is None:
                continue
            sql = _extract_sql(row)
            if not sql:
                continue

            if sql not in sql_to_id:
                sql_to_id[sql] = len(id_to_sql)
                id_to_sql.append(sql)

            key = f"{db_id}|{int(qid)}"
            if key not in query_candidates:
                query_candidates[key] = {}
            query_candidates[key][model] = sql_to_id[sql]

    return sql_to_id, id_to_sql, query_candidates


def run_precompute(sql_dir: Path, out_dir: Path, embedder_id: str, verbose: int = 1) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    _, id_to_sql, query_candidates = collect_unique_sql_candidates(sql_dir)
    if verbose >= 1:
        print(f"[precompute] Unique SQL count: {len(id_to_sql)}")

    embedder = load_embedder(embedder_id=embedder_id, verbose=verbose)

    vectors: list[np.ndarray] = []
    sql_hashes: list[str] = []
    for idx, sql in enumerate(id_to_sql):
        normalized = parse_and_normalize_sql(sql, verbose=0)
        vec = np.array(embedder.encode(normalized), dtype=np.float32)
        vectors.append(vec)
        sql_hashes.append(hashlib.sha256(sql.encode("utf-8")).hexdigest())
        if verbose >= 2 and idx % 500 == 0:
            print(f"[precompute] Encoded {idx}/{len(id_to_sql)}")

    if vectors:
        stacked = np.vstack(vectors)
    else:
        stacked = np.zeros((0, 0), dtype=np.float32)

    save_embeddings_artifact(
        out_dir / "sql_embeddings.npz",
        {
            "sql_ids": np.arange(len(id_to_sql), dtype=np.int64),
            "vectors": stacked,
        },
    )

    metadata = {
        "embedder_id": embedder_id,
        "num_sql": len(id_to_sql),
        "vector_dim": int(stacked.shape[1]) if stacked.size else 0,
        "sql_records": [
            {"sql_id": idx, "sql": sql, "sql_sha256": sql_hashes[idx]}
            for idx, sql in enumerate(id_to_sql)
        ],
        "query_candidates": query_candidates,
    }
    save_json_artifact(out_dir / "sql_embeddings_meta.json", metadata)

    return {
        "num_sql": len(id_to_sql),
        "vector_dim": metadata["vector_dim"],
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute reusable SQL embeddings for selector/visualization strategies.")
    parser.add_argument("--sql-dir", default=str(DEFAULT_SQL_DIR), help="Directory with *_query_results.json files")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for precomputed embedding artifacts")
    parser.add_argument("--embedder-id", default="s2593817/sft-sql-embedding", help="SentenceTransformer model id")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    summary = run_precompute(
        sql_dir=Path(args.sql_dir),
        out_dir=Path(args.out_dir),
        embedder_id=args.embedder_id,
        verbose=args.verbose,
    )
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import ctypes
import json
import os
import threading
from typing import Optional, Any, TYPE_CHECKING

import numpy as np
import sqlglot
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EMBEDDER_ID = "s2593817/sft-sql-embedding"
CACHE_DIR = Path(__file__).resolve().parent / ".hf_cache"
_EMBEDDER: Optional[Any] = None
_EMBEDDER_LOCK = threading.Lock()


def _preload_openmp_runtime(verbose: int = 0) -> None:
    """Preload libgomp to avoid static TLS allocation errors on some systems."""
    candidates = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libgomp.so.1")

    candidates.extend(
        [
            Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1"),
            Path("/usr/lib/aarch64-linux-gnu/libgomp.so.1"),
            Path("/lib/x86_64-linux-gnu/libgomp.so.1"),
            Path("/lib/aarch64-linux-gnu/libgomp.so.1"),
        ]
    )

    for candidate in candidates:
        try:
            if candidate.exists():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                log_message(verbose, 2, f"[DEBUG] Preloaded OpenMP runtime: {candidate}")
                return
        except OSError as e:
            log_message(verbose, 2, f"[DEBUG] Failed to preload {candidate}: {e}")

    # Fallback: rely on dynamic loader search path.
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
        log_message(verbose, 2, "[DEBUG] Preloaded OpenMP runtime via linker path")
    except OSError as e:
        log_message(verbose, 2, f"[DEBUG] Could not preload libgomp.so.1: {e}")


def log_message(verbose: int, level: int, message: str) -> None:
    """Print a message only when the active verbose level allows it."""
    if verbose >= level:
        print(message)


def load_embedder(embedder_id: str = EMBEDDER_ID, verbose: int = 1) -> "SentenceTransformer":
    """Load the embedder once and reuse it from memory."""
    global _EMBEDDER

    # Double-checked locking avoids concurrent initialization races
    # when multiple worker threads call into the embedder at startup.
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _preload_openmp_runtime(verbose=verbose)
                from sentence_transformers import SentenceTransformer

                log_message(verbose, 1, f"[INFO] Loading embedder: {embedder_id}")
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                _EMBEDDER = SentenceTransformer(embedder_id, cache_folder=str(CACHE_DIR))
                log_message(verbose, 1, f"[INFO] Embedder loaded and cached at {CACHE_DIR}")
    else:
        log_message(verbose, 2, "[DEBUG] Using cached embedder")
    return _EMBEDDER

def parse_and_normalize_sql(sql: str, verbose: int = 0) -> str:
    """Parse and normalize SQL using sqlglot."""
    try:
        parsed = sqlglot.parse_one(sql)
        normalized = normalize_identifiers(parsed)
        return repr(normalized)
    except Exception as e:
        log_message(verbose, 2, f"[DEBUG] Error parsing SQL, using original text: {e}")
        return sql


def save_embeddings_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist embedding payload as compressed npz for fast reuse."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    sql_ids = np.array(payload.get("sql_ids", []), dtype=np.int64)
    vectors = np.array(payload.get("vectors", []), dtype=np.float32)
    np.savez_compressed(resolved, sql_ids=sql_ids, vectors=vectors)


def load_embeddings_artifact(path: str | Path) -> dict[str, Any]:
    """Load embeddings artifact saved by save_embeddings_artifact."""
    resolved = Path(path)
    data = np.load(resolved)
    return {
        "sql_ids": data["sql_ids"],
        "vectors": data["vectors"],
    }


def save_similarity_matrix_artifact(path: str | Path, matrix: np.ndarray) -> None:
    """Persist similarity matrix as compressed npz."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(resolved, similarity_matrix=np.array(matrix, dtype=np.float32))


def load_similarity_matrix_artifact(path: str | Path) -> np.ndarray:
    """Load similarity matrix saved with save_similarity_matrix_artifact."""
    resolved = Path(path)
    data = np.load(resolved)
    return data["similarity_matrix"]


def save_json_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist metadata json used by precomputed embedding artifacts."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load metadata json produced by save_json_artifact."""
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_similarity_matrix(vectors):
    """Compute a similarity matrix for a list of vectors."""
    num_vectors = len(vectors)
    similarity_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(i, num_vectors):
            similarity = compute_cosine_similarity(vectors[i], vectors[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Symmetric matrix
    return similarity_matrix


def compute_average_and_std_of_similarities(similarity_matrix):
    """Compute the average and standard deviation of the upper triangle of the similarity matrix."""
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    if upper_triangle.size == 0:
        return 0.0, 0.0
    average_similarity = np.mean(upper_triangle)
    std_similarity = np.std(upper_triangle)
    return average_similarity, std_similarity


def compute_similarity_groups(vectors, similarity_matrix, verbose: int = 1):
    """Group vectors based on a similarity threshold."""
    threshold = compute_average_and_std_of_similarities(similarity_matrix)[0]
    log_message(verbose, 1, f"[INFO] Grouping vectors with threshold: {threshold:.4f}")

    num_vectors = len(vectors)
    groups = []
    visited = set()

    for i in range(num_vectors):
        if i in visited:
            continue

        group = [(i, vectors[i])]
        visited.add(i)

        for j in range(i + 1, num_vectors):
            if j not in visited and similarity_matrix[i][j] >= threshold:
                group.append((j, vectors[j]))
                visited.add(j)

        groups.append(group)

    if verbose >= 2:
        print(f"[DEBUG] Created {len(groups)} groups")
        for idx, g in enumerate(groups):
            print(f"[DEBUG]   Group {idx}: {len(g)} vectors")

    return groups


def compute_similarity_groups_pairwise(similarity_matrix, verbose: int = 1, threshold: Optional[float] = None):
    """Group vectors using pairwise connectivity above a similarity threshold.

    This is a single-link strategy over the thresholded similarity graph:
    two vectors belong to the same group if they are connected by a chain
    of pairwise similarities >= threshold.
    """
    if threshold is None:
        threshold = compute_average_and_std_of_similarities(similarity_matrix)[0]
    log_message(verbose, 1, f"[INFO] Grouping vectors (pairwise) with threshold: {threshold:.4f}")

    num_vectors = len(similarity_matrix)
    groups = []
    visited = set()

    for start_idx in range(num_vectors):
        if start_idx in visited:
            continue

        # Expand a connected component using BFS over thresholded edges.
        queue = [start_idx]
        visited.add(start_idx)
        component_indices = []

        while queue:
            current = queue.pop(0)
            component_indices.append(current)

            for neighbor in range(num_vectors):
                if neighbor in visited:
                    continue
                if similarity_matrix[current][neighbor] >= threshold:
                    visited.add(neighbor)
                    queue.append(neighbor)

        group = [(idx, similarity_matrix[idx]) for idx in component_indices]
        groups.append(group)

    if verbose >= 2:
        print(f"[DEBUG] Created {len(groups)} groups (pairwise)")
        for idx, g in enumerate(groups):
            print(f"[DEBUG]   Group {idx}: {len(g)} vectors")

    return groups


def get_vector_closest_to_centroid(group):
    """Get the vector closest to the centroid of a group."""
    centroid = np.mean([v for _, v in group], axis=0)
    closest_vector = None
    closest_distance = float('inf')
    
    for index, vector in group:
        distance = np.linalg.norm(np.array(vector) - centroid)
        if distance < closest_distance:
            closest_distance = distance
            closest_vector = index

    return closest_vector


def sql_selection_trough_embedding_similarities(sql_queries, verbose: int = 1):
    """Select SQL queries based on embedding similarities and return stats."""
    if not sql_queries:
        raise ValueError("sql_queries must contain at least one query")

    log_message(verbose, 1, f"[INFO] Processing {len(sql_queries)} SQL queries")

    embedder = load_embedder(verbose=verbose)

    if verbose >= 2:
        print("[DEBUG] Parsing and encoding SQL queries...")

    vectors = [embedder.encode(parse_and_normalize_sql(sql, verbose=verbose)) for sql in sql_queries]
    if verbose >= 2:
        print(f"[DEBUG] Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")

    if verbose >= 2:
        print("[DEBUG] Computing similarity matrix...")

    similarity_matrix = compute_similarity_matrix(vectors)
    avg_sim, std_sim = compute_average_and_std_of_similarities(similarity_matrix)
    if verbose >= 2:
        print(f"[DEBUG] Similarity stats - mean: {avg_sim:.4f}, std: {std_sim:.4f}")

    groups = compute_similarity_groups(vectors, similarity_matrix, verbose=verbose)
    biggest_group = max(groups, key=len)
    if verbose >= 2:
        print(f"[DEBUG] Biggest group has {len(biggest_group)} vectors")

    selected_vector = get_vector_closest_to_centroid(biggest_group)
    selected_sql = sql_queries[selected_vector]

    selection_stats = {
        "selected_index": int(selected_vector),
        "num_candidates": len(sql_queries),
        "num_clusters": len(groups),
        "cluster_sizes": [len(group) for group in groups],
        "biggest_cluster_size": len(biggest_group),
        "similarity_mean": float(avg_sim),
        "similarity_std": float(std_sim),
    }

    log_message(verbose, 1, f"[INFO] Selected SQL (index {selected_vector}): {selected_sql}")
    if verbose >= 2:
        print(f"[DEBUG] Selection stats: {selection_stats}")

    return selected_sql, selection_stats


if __name__ == "__main__":
    sql_queries = [
        "SELECT name FROM users WHERE age > 30",
        "SELECT name FROM users WHERE age > 25",
        "SELECT name FROM customers WHERE age > 30",
        "SELECT name FROM users WHERE age > 35",
        "SELECT name FROM customers WHERE age > 25"
    ]

    # Set verbose level: 0=silent, 1=steps (default), 2=all outputs
    selected_sql, selection_stats = sql_selection_trough_embedding_similarities(sql_queries, verbose=2)
    print(f"Selection statistics: {selection_stats}")

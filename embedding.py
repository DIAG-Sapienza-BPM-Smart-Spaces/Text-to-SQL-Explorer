from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import sqlglot
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

EMBEDDER_ID = "s2593817/sft-sql-embedding"
CACHE_DIR = Path(__file__).resolve().parent / ".hf_cache"
_EMBEDDER: Optional[SentenceTransformer] = None


def log_message(verbose: int, level: int, message: str) -> None:
    """Print a message only when the active verbose level allows it."""
    if verbose >= level:
        print(message)


def load_embedder(embedder_id: str = EMBEDDER_ID, verbose: int = 1) -> SentenceTransformer:
    """Load the embedder once and reuse it from memory."""
    global _EMBEDDER
    if _EMBEDDER is None:
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

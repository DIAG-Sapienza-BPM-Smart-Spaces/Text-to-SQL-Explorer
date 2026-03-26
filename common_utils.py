from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Optional


CANONICAL_METRICS = [
    "execution_accuracy",
    "exact_match",
    "sql_f1_score",
    "response_schema_f1_score",
    "cell_f1_score",
]


def fast_hash_hex(text: str, digest_size: int = 16) -> str:
    """Return a deterministic lightweight hex digest for non-security use cases."""
    size = max(4, min(int(digest_size), 32))
    return hashlib.blake2b(str(text).encode("utf-8"), digest_size=size).hexdigest()


def fast_hash_int(text: str, digest_size: int = 16) -> int:
    """Return a deterministic integer hash for IDs/cache keys (non-cryptographic)."""
    size = max(4, min(int(digest_size), 32))
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=size).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path, optionally relative to a provided base directory."""
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = base_dir / resolved
    return resolved


def load_json(path: str | Path, base_dir: Optional[Path] = None) -> Any:
    """Load JSON content from disk with optional base-dir resolution."""
    resolved = resolve_path(path, base_dir=base_dir)
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_dump_json(
    path: str | Path,
    payload: Any,
    *,
    base_dir: Optional[Path] = None,
    lock: Optional[Lock] = None,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """Write JSON atomically to reduce risk of partial files on interruptions."""
    resolved = resolve_path(path, base_dir=base_dir)

    def _write() -> None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent)
        tmp_path.replace(resolved)

    if lock is not None:
        with lock:
            _write()
    else:
        _write()

def metric_to_percentage(value: Any) -> Optional[float]:
    """Convert a scalar metric from [0,1] to [0,100] when needed."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num * 100.0 if num <= 1.0 else num


def readable_model_label(system_id: str) -> str:
    """Convert a system id into a display label used by visualizations."""
    if not isinstance(system_id, str):
        return "Unknown"
    token = system_id.strip()
    explicit = {
        "deepseek-chat": "DeepSeek Chat",
        "qwen2.5-coder_32b": "Qwen2.5 Coder 32B",
        "qwen3-coder_30b": "Qwen3 Coder 30B",
        "cogito_70b": "Cogito 70B",
        "codellama_70b": "CodeLlama 70B",
        "codestral_22b": "Codestral 22B",
        "sqlcoder_15b": "SQLCoder 15B",
        "ground_truth": "Ground Truth",
    }
    if token in explicit:
        return explicit[token]
    words = re.split(r"[_\-]+", token)
    return " ".join(w.capitalize() for w in words if w)


def collect_models_from_metric_files(metrics_dir: str | Path) -> list[str]:
    """Collect system model ids from metrics_results file names."""
    root = resolve_path(metrics_dir)
    if not root.exists():
        return []

    models = set()
    for path in root.glob("evaluation_sql_metrics_*_vs_ground_truth.json"):
        stem = path.stem
        marker = "evaluation_sql_metrics_"
        suffix = "_vs_ground_truth"
        if not stem.startswith(marker) or not stem.endswith(suffix):
            continue
        model_id = stem[len(marker): -len(suffix)]
        if model_id:
            models.add(model_id)

    return sorted(models)


def build_pairwise_index(pairwise_payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Index pairwise payload rows by (db_id, question_id)."""
    index: dict[tuple[str, int], dict[str, Any]] = {}

    for row in pairwise_payload.values():
        if not isinstance(row, dict):
            continue
        db_id = row.get("db_id")
        question_id = row.get("question_id")
        if db_id is None or question_id is None:
            continue
        index[(str(db_id), int(question_id))] = row

    return index


def _pair_key(model_a: str, model_b: str) -> tuple[str, str]:
    a = str(model_a)
    b = str(model_b)
    return (a, b) if a <= b else (b, a)


def extract_pairwise_comparison(
    pairwise_entry: Optional[dict[str, Any]],
    model_a: str,
    model_b: str,
) -> Optional[dict[str, Any]]:
    """Extract comparison block for any model pair, regardless of ordering."""
    if not pairwise_entry:
        return None
    comparisons = pairwise_entry.get("comparisons", {})
    if not isinstance(comparisons, dict):
        return None

    expected = _pair_key(model_a, model_b)
    for comp in comparisons.values():
        if not isinstance(comp, dict):
            continue
        s1 = comp.get("system1")
        s2 = comp.get("system2")
        if s1 is None or s2 is None:
            continue
        if _pair_key(str(s1), str(s2)) == expected:
            return comp

    return None


def extract_ground_truth_comparison(
    pairwise_entry: Optional[dict[str, Any]],
    candidate_model: str,
) -> Optional[dict[str, Any]]:
    """Extract the ground_truth-vs-candidate_model comparison block, if present."""
    if not pairwise_entry:
        return None

    comparisons = pairwise_entry.get("comparisons", {})
    if not isinstance(comparisons, dict):
        return None

    expected_key = f"ground_truth_vs_{candidate_model}"
    comparison = comparisons.get(expected_key)
    if isinstance(comparison, dict):
        return comparison

    for comp in comparisons.values():
        if not isinstance(comp, dict):
            continue
        if comp.get("system1") == "ground_truth" and comp.get("system2") == candidate_model:
            return comp

    return None

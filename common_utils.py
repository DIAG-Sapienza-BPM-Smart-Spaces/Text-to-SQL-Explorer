from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Optional


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

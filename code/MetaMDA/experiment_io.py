from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def default_similarity_graph_path(data_dir: Union[str, Path]) -> str:
    """Prefer MDAD run_similarity_graph_0.7.txt (best auroc_mean in param_sim_alpha sweep); else demo 0.65."""
    d = Path(data_dir)
    tuned = d / "run_similarity_graph_0.7.txt"
    if tuned.is_file():
        return str(tuned)
    return str(d / "demo_similarity_graph_0_65.txt")


def append_result_csv(
    row: Dict[str, Any],
    out_csv: str,
    fieldnames: Optional[Iterable[str]] = None,
) -> None:
    """
    Append one experiment result as a row into CSV.

    - Creates parent dirs automatically.
    - If CSV doesn't exist, writes header first.
    - Serializes dict/list values to JSON strings for stability.
    """
    out_csv = str(out_csv)
    _ensure_parent_dir(out_csv)

    normalized: Dict[str, Any] = dict(row)
    normalized.setdefault("timestamp_utc", _utc_now_iso())
    for k, v in list(normalized.items()):
        if isinstance(v, (dict, list, tuple)):
            normalized[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)

    if fieldnames is None:
        # stable ordering: common fields first if present, then the rest
        preferred = [
            "timestamp_utc",
            "exp_name",
            "dataset",
            "cv",
            "seed",
            "model",
            "embedding_file",
            "pair_file",
            "acc",
            "auroc",
            "aupr",
            "f1",
            "acc_mean",
            "acc_std",
            "auroc_mean",
            "auroc_std",
            "aupr_mean",
            "aupr_std",
            "f1_mean",
            "f1_std",
            "config_json",
            "notes",
        ]
        keys = list(normalized.keys())
        ordered = [k for k in preferred if k in normalized] + [k for k in keys if k not in preferred]
        fieldnames = ordered

    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({k: normalized.get(k, "") for k in writer.fieldnames})


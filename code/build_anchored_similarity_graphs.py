#!/usr/bin/env python3
from __future__ import annotations

"""
Anchor-based similarity graph builder for MDAD.

Goal:
- Generate lower-threshold similarity graph files (e.g. 0.50/0.55/0.60)
- Keep demo 0.65 graph as a hard anchor:
  filtering generated graph with threshold >= 0.65 must exactly match
  data/MDAD/demo_similarity_graph_0_65.txt.

Approach:
1) Read anchor edges from demo_similarity_graph_0_65.txt.
2) Compute additional candidate similarities from mda.xlsx:
   - microbe-microbe cosine similarity on row profiles
   - drug-drug cosine similarity on column profiles
3) Only add candidates with score in [min_target_threshold, 0.65).
4) Export thresholded graph files for requested thresholds.
5) Verify strict anchor consistency.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


EdgeKey = Tuple[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_pair(a: str, b: str) -> EdgeKey:
    return (a, b) if a <= b else (b, a)


def _read_nodes(path: Path) -> set[str]:
    nodes: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                nodes.add(s.split("\t")[0])
    return nodes


def _read_graph(path: Path) -> Dict[EdgeKey, float]:
    edges: Dict[EdgeKey, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 4:
                continue
            u, v = parts[0], parts[1]
            score = float(parts[3])
            edges[_canonical_pair(u, v)] = score
    return edges


def _cosine_sim(mat: np.ndarray) -> np.ndarray:
    # mat: (n, d)
    eps = 1e-12
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    x = mat / denom
    return x @ x.T


def _iter_pairs(names: Sequence[str], sim: np.ndarray) -> Iterable[Tuple[EdgeKey, float]]:
    n = len(names)
    for i in range(n):
        ni = names[i]
        row = sim[i]
        for j in range(i + 1, n):
            nj = names[j]
            yield _canonical_pair(ni, nj), float(row[j])


def _build_candidates_from_xlsx(
    xlsx_path: Path,
    allowed_nodes: set[str],
    low_thr: float,
    high_thr: float,
    blocked_pairs: set[EdgeKey],
) -> Dict[EdgeKey, float]:
    df = pd.read_excel(xlsx_path)
    row_nodes = [str(x) for x in df.iloc[:, 0].tolist()]
    col_nodes = [str(c) for c in df.columns[1:].tolist()]
    mat = df.iloc[:, 1:].to_numpy(dtype=float)

    out: Dict[EdgeKey, float] = {}

    # Row-row (microbe-microbe)
    rr = _cosine_sim(mat)
    for key, score in _iter_pairs(row_nodes, rr):
        if key in blocked_pairs:
            continue
        if key[0] not in allowed_nodes or key[1] not in allowed_nodes:
            continue
        if low_thr <= score < high_thr:
            out[key] = score

    # Col-col (drug-drug)
    cc = _cosine_sim(mat.T)
    for key, score in _iter_pairs(col_nodes, cc):
        if key in blocked_pairs:
            continue
        if key[0] not in allowed_nodes or key[1] not in allowed_nodes:
            continue
        if low_thr <= score < high_thr:
            out[key] = score

    return out


def _write_graph(path: Path, edges: Dict[EdgeKey, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(edges.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    with path.open("w", encoding="utf-8") as f:
        for i, ((u, v), s) in enumerate(ordered):
            f.write(f"{u}\t{v}\t1\t{s}\t{i}\n")


def _write_nodes(path: Path, edges: Dict[EdgeKey, float]) -> None:
    nodes = set()
    for (u, v) in edges.keys():
        nodes.add(u)
        nodes.add(v)
    with path.open("w", encoding="utf-8") as f:
        for n in sorted(nodes):
            f.write(f"{n}\n")


def _filter_by_threshold(edges: Dict[EdgeKey, float], thr: float) -> Dict[EdgeKey, float]:
    return {k: v for k, v in edges.items() if v >= thr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anchored MDAD similarity graphs.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_repo_root() / "data" / "MDAD",
        help="MDAD data directory",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60],
        help="Target thresholds to export",
    )
    parser.add_argument(
        "--anchor_threshold",
        type=float,
        default=0.65,
        help="Anchor threshold used by demo graph",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="run_similarity_graph",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    thresholds = sorted(args.thresholds)
    min_thr = min(thresholds)
    anchor_thr = float(args.anchor_threshold)

    anchor_graph = data_dir / "demo_similarity_graph_0_65.txt"
    node_file = data_dir / "demo_graph_node.txt"
    xlsx_file = data_dir / "mda.xlsx"

    if not anchor_graph.exists():
        raise FileNotFoundError(f"Anchor graph not found: {anchor_graph}")
    if not node_file.exists():
        raise FileNotFoundError(f"Node file not found: {node_file}")
    if not xlsx_file.exists():
        raise FileNotFoundError(f"mda.xlsx not found: {xlsx_file}")

    allowed_nodes = _read_nodes(node_file)
    anchor_edges = _read_graph(anchor_graph)

    candidates = _build_candidates_from_xlsx(
        xlsx_path=xlsx_file,
        allowed_nodes=allowed_nodes,
        low_thr=min_thr,
        high_thr=anchor_thr - 1e-12,
        blocked_pairs=set(anchor_edges.keys()),
    )

    union_edges: Dict[EdgeKey, float] = dict(anchor_edges)
    union_edges.update(candidates)

    # Strict consistency check:
    # filtering union edges at anchor threshold must exactly recover anchor graph.
    recovered = _filter_by_threshold(union_edges, anchor_thr)
    if recovered != anchor_edges:
        extra = len(set(recovered.keys()) - set(anchor_edges.keys()))
        miss = len(set(anchor_edges.keys()) - set(recovered.keys()))
        raise RuntimeError(
            f"Anchor consistency failed: extra={extra}, missing={miss}. "
            f"Expected exact match with {anchor_graph.name} at >= {anchor_thr}."
        )

    for thr in thresholds:
        thr_tag = f"{thr:.2f}".rstrip("0").rstrip(".")
        out_graph = data_dir / f"{args.prefix}_{thr_tag}.txt"
        out_nodes = data_dir / f"{args.prefix}_{thr_tag}_nodes.txt"
        e = _filter_by_threshold(union_edges, thr)
        _write_graph(out_graph, e)
        _write_nodes(out_nodes, e)
        print(f"[OK] {out_graph.name} edges={len(e)} nodes={len(_read_nodes(out_nodes))}")

    print(
        f"[DONE] anchor_edges={len(anchor_edges)} added_candidates={len(candidates)} "
        f"union={len(union_edges)}"
    )


if __name__ == "__main__":
    main()

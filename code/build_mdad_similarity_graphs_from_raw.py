#!/usr/bin/env python3
from __future__ import annotations

"""
从 MDAD 数据构建药物–药物 / 微生物–微生物相似图边表（与 run_similarity_alpha_sensitivity.py 兼容）。

支持 3 种相似度来源（择一）：

1) 默认（仓库预计算）：使用 `data/MDAD/mdad_from_raw.npz` 的 drug_sim / microbe_sim。
   相似度在 [0,1]，可对阈值 τ（论文中的 α）做扫描导出 `run_similarity_graph_<τ>.txt`。

2) 旧“高斯×融合”（--from_txt_gaussian）：
   读取距离矩阵 Sr/Sm 并按高斯核生成相似度，再与先验（结构/功能相似度）线性融合：
     G_ij = exp( - d_ij^2 / (2 σ^2) )，S_ij = λ·G_ij + (1-λ)·S_ij^(0)

3) 论文一致的 GIP 核方式（--paper_gip，对应 btaf649）：
   - 由交互 profile 计算 Gaussian interaction profile kernel:
       K(i,j) = exp( -η ||p_i - p_j||^2 ),
       η = 1 / ( (1/N) * Σ ||p_i||^2 ) （论文中 η' 固定为 1）
   - 再与功能/结构相似度取平均：sim = (GIP + FUNC)/2
   - 最后按阈值 α 过滤生成同类型相似边（微生物-微生物、药物-药物）

注意：run_similarity_alpha_sensitivity.py 中的「sim_alpha」就是这里导出时使用的阈值（论文 α）。
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EdgeKey = Tuple[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _patch_numpy_pickle_compat() -> None:
    """
    Compatibility shim for loading object arrays saved in .npz with pickle.

    Some NumPy versions pickle objects with module paths like `numpy._core.*`.
    Older environments may only expose `numpy.core.*`, causing:
      ModuleNotFoundError: No module named 'numpy._core'
    """
    import sys

    try:
        import numpy.core as _core  # type: ignore
    except Exception:
        return

    sys.modules.setdefault("numpy._core", _core)

    # Best-effort alias common submodules used in pickles.
    for sub in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        src = f"numpy.core.{sub}"
        dst = f"numpy._core.{sub}"
        if dst in sys.modules:
            continue
        try:
            mod = __import__(src, fromlist=["*"])
        except Exception:
            continue
        sys.modules.setdefault(dst, mod)


def _canonical_pair(a: str, b: str) -> EdgeKey:
    return (a, b) if a <= b else (b, a)


def _read_nodetypes(path: Path) -> Dict[str, str]:
    """Read `demo_nodetypes.tsv` mapping: node -> type."""
    m: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                continue
            node, typ = parts[0], parts[1]
            if node and typ:
                m[node] = typ
    return m


def _load_I_microbe_drug(
    pairf: Path, drug_index: Dict[str, int], microbe_index: Dict[str, int]
) -> np.ndarray:
    """
    Load microbe–drug association matrix I from `combined_mda.tsv`.
    Format: drug<TAB>disease(microbe)<TAB>label
    """
    I = np.zeros((len(microbe_index), len(drug_index)), dtype=np.uint8)
    with pairf.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            drug, microbe, label = parts[0], parts[1], parts[2]
            if label != "1":
                continue
            di = drug_index.get(drug)
            mi = microbe_index.get(microbe)
            if di is None or mi is None:
                continue
            I[mi, di] = 1
    return I


def _load_SM_DM_from_demo_graph(
    graphf: Path,
    nodetypes: Dict[str, str],
    drug_index: Dict[str, int],
    microbe_index: Dict[str, int],
    metabolite_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load microbe–metabolite (SM) and drug–metabolite (DM) binary matrices."""
    SM = np.zeros((len(microbe_index), len(metabolite_index)), dtype=np.uint8)
    DM = np.zeros((len(drug_index), len(metabolite_index)), dtype=np.uint8)

    # Repo convention:
    # - drugs: "drug"
    # - microbes: sometimes "microbe", sometimes reused label "disease"
    # - metabolites: sometimes "metabolite", sometimes reused label "gene"
    drug_labels = {"drug"}
    microbe_labels = {"microbe", "disease"}
    metabolite_labels = {"metabolite", "gene"}

    with graphf.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            tu = nodetypes.get(u)
            tv = nodetypes.get(v)
            if tu is None or tv is None:
                continue

            if (tu in drug_labels and tv in metabolite_labels) or (tu in metabolite_labels and tv in drug_labels):
                drug = u if tu in drug_labels else v
                met = v if tu in drug_labels else u
                di = drug_index.get(drug)
                mi = metabolite_index.get(met)
                if di is not None and mi is not None:
                    DM[di, mi] = 1
                continue

            if (tu in microbe_labels and tv in metabolite_labels) or (tu in metabolite_labels and tv in microbe_labels):
                mic = u if tu in microbe_labels else v
                met = v if tu in microbe_labels else u
                si = microbe_index.get(mic)
                mi = metabolite_index.get(met)
                if si is not None and mi is not None:
                    SM[si, mi] = 1
                continue

    return SM, DM


def _gip_kernel(profile: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Gaussian interaction profile kernel (paper):
      K(i,j) = exp( -η ||p_i - p_j||^2 )
      η = 1 / ( (1/N) * Σ ||p_i||^2 )
    """
    x = np.asarray(profile, dtype=np.float64)
    n2 = np.sum(x * x, axis=1)
    denom = float(np.mean(n2)) if n2.size else 0.0
    eta = 1.0 / max(denom, eps)
    gram = x @ x.T
    dist2 = (n2[:, None] + n2[None, :] - 2.0 * gram)
    dist2 = np.maximum(dist2, 0.0)
    K = np.exp(-eta * dist2)
    np.fill_diagonal(K, 1.0)
    return K


def _try_load_square_matrix(path: Optional[Path], n: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    m = np.loadtxt(p)
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (n, n):
        raise ValueError(f"相似度矩阵形状不匹配: {p} shape={m.shape}, expected=({n},{n})")
    return m


def _default_sigma(distance: np.ndarray) -> float:
    """由非零距离估计 σ（鲁棒：中位数；全零则回退）。"""
    d = distance[np.triu_indices_from(distance, k=1)]
    d = d[d > 0]
    if d.size == 0:
        return 0.1
    med = float(np.median(d))
    return max(med * np.sqrt(2.0), 1e-6)


def _gaussian_from_distance(distance: np.ndarray, sigma: float) -> np.ndarray:
    s = max(float(sigma), 1e-12)
    return np.exp(-(distance**2) / (2.0 * s * s))


def _fuse(gauss: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    return lam * gauss + (1.0 - lam) * prior


def _iter_pairs(names: Sequence[str], sim: np.ndarray) -> Iterable[Tuple[EdgeKey, float]]:
    n = len(names)
    for i in range(n):
        ni = names[i]
        row = sim[i]
        for j in range(i + 1, n):
            nj = names[j]
            yield _canonical_pair(ni, nj), float(row[j])


def _write_graph(path: Path, edges: Dict[EdgeKey, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(edges.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    with path.open("w", encoding="utf-8") as f:
        for i, ((u, v), s) in enumerate(ordered):
            f.write(f"{u}\t{v}\t1\t{s}\t{i}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="从原始 MDAD 矩阵生成相似图边表")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/Users/chenziyang/Desktop/mdad原始数据"),
        help="含 Sr_dis_matrix.txt、Sm_dis_matrix.txt、drug_structure_sim.txt、microbe_function_sim.txt",
    )
    parser.add_argument(
        "--name_npz",
        type=Path,
        default=_repo_root() / "data" / "MDAD" / "mdad_from_raw.npz",
        help="含 drug_names、microbe_names（与矩阵行序一致）",
    )
    parser.add_argument(
        "--paper_gip",
        action="store_true",
        help="按论文 btaf649：用 GIP 核从交互矩阵计算相似度，并与功能/结构相似度取平均",
    )
    parser.add_argument(
        "--pairf",
        type=Path,
        default=_repo_root() / "data" / "MDAD" / "combined_mda.tsv",
        help="microbe–drug 交互边表（用于构建 I 矩阵）",
    )
    parser.add_argument(
        "--graphf",
        type=Path,
        default=_repo_root() / "data" / "MDAD" / "demo_graph.txt",
        help="异构图边表（用于构建 SM/DM 矩阵）",
    )
    parser.add_argument(
        "--nodetypef",
        type=Path,
        default=_repo_root() / "data" / "MDAD" / "demo_nodetypes.tsv",
        help="节点类型文件（用于解析 demo_graph 中的 drug/microbe/metabolite）",
    )
    parser.add_argument(
        "--use_metabolite_profiles",
        action="store_true",
        help="在 GIP 中额外纳入 microbe–metabolite 与 drug–metabolite profile（更贴近论文）",
    )
    parser.add_argument(
        "--drug_func_sim",
        type=Path,
        default=None,
        help="药物结构/功能相似度矩阵（Nd×Nd），用于与 GIP 取平均；不传则只用 GIP",
    )
    parser.add_argument(
        "--microbe_func_sim",
        type=Path,
        default=None,
        help="微生物功能相似度矩阵（Ns×Ns），用于与 GIP 取平均；不传则只用 GIP",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_repo_root() / "data" / "MDAD",
        help="输出 run_similarity_graph_<τ>.txt 的目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="run_similarity_graph",
        help="输出文件名前缀",
    )
    parser.add_argument(
        "--fusion-weight",
        type=float,
        default=0.5,
        help="λ：S = λ·G(d) + (1-λ)·S^(0)，与阈值扫描中的 τ（sim_alpha）不同",
    )
    parser.add_argument(
        "--sigma-drug",
        type=float,
        default=None,
        help="药物距离高斯核 σ；默认由 Sr 非零距离自动估计",
    )
    parser.add_argument(
        "--sigma-microbe",
        type=float,
        default=None,
        help="微生物距离高斯核 σ；默认由 Sm 非零距离自动估计",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65, 0.70],
        help="仅输出相似度 >= τ 的边（对应敏感性实验中的 sim_alpha）",
    )
    parser.add_argument(
        "--from_txt_gaussian",
        action="store_true",
        help="从 raw_dir 读 Sr/Sm 与结构/功能矩阵做高斯×融合（不推荐，易与先验量纲不匹配）",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    z = None
    drug_names: List[str]
    microbe_names: List[str]
    drug_index: Dict[str, int]
    microbe_index: Dict[str, int]

    if args.paper_gip:
        pairf = Path(args.pairf)
        if not pairf.exists():
            raise FileNotFoundError(f"缺少 pair 文件: {pairf}")

        # IMPORTANT: build node universe from pair file to avoid name-order mismatch
        # with `mdad_from_raw.npz` (the repo contains multiple naming conventions).
        pair_drugs: set[str] = set()
        pair_microbes: set[str] = set()
        with pairf.open("r", encoding="utf-8") as f:
            _ = f.readline()
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split("\t")
                if len(parts) < 3:
                    continue
                d, m, y = parts[0], parts[1], parts[2]
                if y != "1":
                    continue
                pair_drugs.add(d)
                pair_microbes.add(m)

        drug_names = sorted(pair_drugs)
        microbe_names = sorted(pair_microbes)
        drug_index = {n: i for i, n in enumerate(drug_names)}
        microbe_index = {n: i for i, n in enumerate(microbe_names)}

        I = _load_I_microbe_drug(pairf, drug_index=drug_index, microbe_index=microbe_index)

        # GIP derived from microbe-drug matrix I (rows: microbe, cols: drug)
        gip_microbe = _gip_kernel(I)
        gip_drug = _gip_kernel(I.T)

        if args.use_metabolite_profiles:
            nodetypef = Path(args.nodetypef)
            graphf = Path(args.graphf)
            if not nodetypef.exists():
                raise FileNotFoundError(f"缺少 nodetype 文件: {nodetypef}")
            if not graphf.exists():
                raise FileNotFoundError(f"缺少 graph 文件: {graphf}")

            nodetypes = _read_nodetypes(nodetypef)
            metabolites = [n for n, t in nodetypes.items() if t in ("metabolite", "gene")]
            metabolite_index = {n: i for i, n in enumerate(sorted(metabolites))}
            SM, DM = _load_SM_DM_from_demo_graph(
                graphf=graphf,
                nodetypes=nodetypes,
                drug_index=drug_index,
                microbe_index=microbe_index,
                metabolite_index=metabolite_index,
            )
            if SM.shape[1] > 0:
                gip_microbe = 0.5 * (gip_microbe + _gip_kernel(SM))
            if DM.shape[1] > 0:
                gip_drug = 0.5 * (gip_drug + _gip_kernel(DM))

        # Optional functional similarity: only accept well-aligned square matrices.
        mfs = None
        dss = None
        if args.microbe_func_sim is not None:
            try:
                mfs = _try_load_square_matrix(args.microbe_func_sim, n=len(microbe_names))
            except Exception as e:
                print(f"[WARN] microbe_func_sim 忽略（与当前节点集合不匹配）: {e}")
        if args.drug_func_sim is not None:
            try:
                dss = _try_load_square_matrix(args.drug_func_sim, n=len(drug_names))
            except Exception as e:
                print(f"[WARN] drug_func_sim 忽略（与当前节点集合不匹配）: {e}")

        s_microbe = 0.5 * (gip_microbe + mfs) if mfs is not None else gip_microbe
        s_drug = 0.5 * (gip_drug + dss) if dss is not None else gip_drug
        print(
            "[sim] paper_gip: 使用 GIP 核"
            + (" + 功能/结构相似度取平均" if (mfs is not None or dss is not None) else "（未提供 FUNC，仅用 GIP）")
        )

    elif not args.from_txt_gaussian:
        name_npz = Path(args.name_npz)
        if not name_npz.exists():
            raise FileNotFoundError(f"缺少名称文件: {name_npz}")

        _patch_numpy_pickle_compat()
        z = np.load(name_npz, allow_pickle=True)
        drug_names = [str(x) for x in z["drug_names"].tolist()]
        microbe_names = [str(x) for x in z["microbe_names"].tolist()]
        drug_index = {n: i for i, n in enumerate(drug_names)}
        microbe_index = {n: i for i, n in enumerate(microbe_names)}

        s_drug = np.asarray(z["drug_sim"], dtype=np.float64)
        s_microbe = np.asarray(z["microbe_sim"], dtype=np.float64)
        print("[sim] 使用 mdad_from_raw.npz 中的 drug_sim / microbe_sim")
    else:
        name_npz = Path(args.name_npz)
        if not name_npz.exists():
            raise FileNotFoundError(f"缺少名称文件: {name_npz}")

        _patch_numpy_pickle_compat()
        z = np.load(name_npz, allow_pickle=True)
        drug_names = [str(x) for x in z["drug_names"].tolist()]
        microbe_names = [str(x) for x in z["microbe_names"].tolist()]
        drug_index = {n: i for i, n in enumerate(drug_names)}
        microbe_index = {n: i for i, n in enumerate(microbe_names)}

        Sr = np.loadtxt(raw_dir / "Sr_dis_matrix.txt")
        Sm = np.loadtxt(raw_dir / "Sm_dis_matrix.txt")
        dss = np.loadtxt(raw_dir / "drug_structure_sim.txt")
        mfs = np.loadtxt(raw_dir / "microbe_function_sim.txt")
        if Sr.shape[0] != len(drug_names) or dss.shape[0] != len(drug_names):
            raise ValueError(
                f"药物矩阵形状 {Sr.shape} / {dss.shape} 与 drug_names {len(drug_names)} 不一致"
            )
        if Sm.shape[0] != len(microbe_names):
            raise ValueError(f"微生物矩阵形状 {Sm.shape} 与 microbe_names {len(microbe_names)} 不一致")

        sig_d = float(args.sigma_drug) if args.sigma_drug is not None else _default_sigma(Sr)
        sig_m = float(args.sigma_microbe) if args.sigma_microbe is not None else _default_sigma(Sm)
        Gd = _gaussian_from_distance(Sr, sig_d)
        Gm = _gaussian_from_distance(Sm, sig_m)
        lam = float(args.fusion_weight)
        s_drug = _fuse(Gd, dss, lam)
        s_microbe = _fuse(Gm, mfs, lam)
        print(
            f"[fuse] λ={lam:g}  σ_drug={sig_d:.6g}  σ_microbe={sig_m:.6g}  "
            f"(若需与论文完全一致，请用 --sigma-drug / --sigma-microbe 显式指定)"
        )

    union_edges: Dict[EdgeKey, float] = {}
    for key, score in _iter_pairs(drug_names, s_drug):
        union_edges[key] = score
    for key, score in _iter_pairs(microbe_names, s_microbe):
        union_edges[key] = score

    for thr in sorted(float(x) for x in args.thresholds):
        thr_tag = f"{thr:.2f}".rstrip("0").rstrip(".")
        out = args.out_dir / f"{args.prefix}_{thr_tag}.txt"
        filtered = {k: v for k, v in union_edges.items() if v >= thr - 1e-15}
        _write_graph(out, filtered)
        print(f"[OK] {out.name}  edges={len(filtered)}  (>= {thr:g})")

    print(f"[DONE] total_pairs_union={len(union_edges)}")


if __name__ == "__main__":
    main()

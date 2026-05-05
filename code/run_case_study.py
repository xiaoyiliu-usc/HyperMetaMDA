from __future__ import annotations

"""
通用 case study 脚本：支持 tetracycline / vancomycin / acarbose，以及任意自定义药物。

功能：
1. 从 pair 文件中删除目标药物的所有已知关联（drug-specific cold-start / case study）
2. 用给定 embedding + XGBoost 重新训练
3. 对目标药物 × 全图微生物（来自 nodetypes）打分并排序
4. 导出 Top-K 结果到 CSV，并额外保存 summary.json

推荐用法（仓库根目录下）：

# 1) 先列出和 tetracycline 相关的真实药物名，避免字符串不一致
python code/run_case_study.py --dataset MDAD --find_drug tetracycline

# 2) 跑论文前两个 case（默认超图 main_hgat；若要复现原论文无超图 ablation，加 --embedding_file .../w_o_hypergraph...）
python code/run_case_study.py --dataset MDAD --cases tetracycline vancomycin --top_k 20

# 3) 自定义目标药物（精确名字）
python code/run_case_study.py --dataset MDAD --target_drug "Tetracycline" --top_k 20

# 4) 自定义关键词自动解析（忽略大小写，支持包含匹配）
python code/run_case_study.py --dataset MDAD --target_query tetracycline --top_k 20

# 5) 固定微生物 → Top-K 药物（microbe cold-start：删掉该微生物全部标注）
python code/run_case_study.py --dataset MDAD --target_microbe Staphylococcus_aureus --top_k 20
"""

import argparse
import json
import pickle
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from MetaMDA.utils import set_seed
from MetaMDA.predict_final import convert_dataset


# 与 run_all_exps / run_coldstart_case 对齐
XGB_PARAMS = dict(
    gamma=0,
    learning_rate=0.049,
    max_depth=12,
    n_estimators=564,
    min_child_weight=2,
    subsample=0.81,
    colsample_bytree=0.54,
)


# 预设 case：别名用于在 MDAD 等数据集中解析（pair 里多为分子式）
CASE_PRESETS = {
    "tetracycline": ["tetracycline", "C22H24N2O8"],
    "vancomycin": ["vancomycin", "C66H75Cl2N9O24"],
    "acarbose": ["acarbose", "c25h43no18", "C25H43NO18"],
    # MDAD 中节点为 Hill 分子式；与部分文献/其它库写法不一致时见 MDAD_DRUG_FORMULA_ALIASES
    "rifampicin": ["rifampicin", "rifampin", "C43H58N4O12"],
    "sulfamethoxazole": ["sulfamethoxazole", "C10H11N3O3S"],
}

# 常见误写 -> MDAD combined_mda / 图中实际使用的 drug 节点名（分子式）
MDAD_DRUG_FORMULA_ALIASES: Dict[str, str] = {
    "c43h49n9o7": "C43H58N4O12",  # 利福平：MDAD 为 C43H58N4O12（非 C43H49N9O7）
    "c18h18n4o5s": "C10H11N3O3S",  # 磺胺甲噁唑：MDAD 为 C10H11N3O3S（非 C18H18N4O5S）
}


def _maybe_apply_mdad_formula_alias(requested: str) -> str:
    key = _normalize_text(requested)
    if key in MDAD_DRUG_FORMULA_ALIASES:
        fixed = MDAD_DRUG_FORMULA_ALIASES[key]
        print(
            f"[Hint] 将药物节点 '{requested}' 映射为 MDAD 图中的 '{fixed}' "
            f"（原写法在 pair/embedding 中不存在）"
        )
        return fixed
    return requested


@dataclass
class CaseResult:
    dataset: str
    requested_case: str
    resolved_target_drug: str
    embedding_file: str
    pair_file: str
    nodetype_file: str
    candidates_source: str
    train_rows_before: int
    train_rows_after: int
    removed_rows: int
    candidate_microbe_count: int
    scored_microbe_count: int
    output_csv: str
    output_json: str
    top_k: int


@dataclass
class MicrobeCaseResult:
    dataset: str
    requested_case: str
    resolved_target_microbe: str
    embedding_file: str
    pair_file: str
    nodetype_file: str
    candidates_source: str
    train_rows_before: int
    train_rows_after: int
    removed_rows: int
    candidate_drug_count: int
    scored_drug_count: int
    output_csv: str
    output_json: str
    top_k: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_embedding_file(root: Path, emb_dir: Path, embedding_file: str | None) -> str:
    """默认 main_hgat.cv5.seed42.pkl；若不存在则尝试 data/<dataset>/embedding_file.pkl。"""
    if embedding_file is not None:
        p = Path(embedding_file)
        return str(p if p.is_absolute() else (root / p))
    preferred = emb_dir / "main_hgat.cv5.seed42.pkl"
    if preferred.is_file():
        return str(preferred)
    fallback = emb_dir.parent / "embedding_file.pkl"
    if fallback.is_file():
        try:
            rel = fallback.relative_to(root)
        except ValueError:
            rel = fallback
        print(f"[Hint] 未找到 {preferred.name}，使用 {rel}")
        return str(fallback)
    return str(preferred)


def _default_paths(dataset: str) -> dict:
    d = _repo_root() / "data" / dataset
    return {
        "pairf": str(d / "combined_mda.tsv"),
        "nodetypef": str(d / "demo_nodetypes.tsv"),
        "emb_dir": d / "embeddings",
    }


def _read_pairs(pairf: str) -> tuple:
    df = pd.read_csv(pairf, sep="\t")
    drug_col = "drug" if "drug" in df.columns else df.columns[0]
    microbe_col = "disease" if "disease" in df.columns else df.columns[1]
    label_col = "label" if "label" in df.columns else df.columns[2]
    return df, drug_col, microbe_col, label_col


def _normalize_text(x: object) -> str:
    return str(x).strip().lower()


def _unique_drugs(df: pd.DataFrame, drug_col: str) -> List[str]:
    vals = df[drug_col].astype(str).map(str.strip)
    return sorted({v for v in vals if v})


def find_matching_drugs(pairf: str, query: str) -> List[str]:
    df, drug_col, _, _ = _read_pairs(pairf)
    q = _normalize_text(query)
    matches = []
    for drug in _unique_drugs(df, drug_col):
        dn = _normalize_text(drug)
        if q == dn or q in dn:
            matches.append(drug)
    return matches


def _unique_microbes(df: pd.DataFrame, microbe_col: str) -> List[str]:
    vals = df[microbe_col].astype(str).map(str.strip)
    return sorted({v for v in vals if v})


def find_matching_microbes(pairf: str, query: str) -> List[str]:
    df, _, microbe_col, _ = _read_pairs(pairf)
    q = _normalize_text(query)
    matches = []
    for m in _unique_microbes(df, microbe_col):
        mn = _normalize_text(m)
        if q == mn or q in mn:
            matches.append(m)
    return matches


def _resolve_target_drug_one(pairf: str, requested: str) -> str:
    """按以下顺序解析：精确匹配 -> 忽略大小写精确匹配 -> 包含匹配（唯一）"""
    df, drug_col, _, _ = _read_pairs(pairf)
    drugs = _unique_drugs(df, drug_col)
    if requested in drugs:
        return requested

    req_n = _normalize_text(requested)
    exact_ci = [d for d in drugs if _normalize_text(d) == req_n]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(
            f"药物 '{requested}' 存在多个仅大小写不同的候选: {exact_ci}，请显式指定 --target_drug。"
        )

    contains = [d for d in drugs if req_n in _normalize_text(d)]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        preview = ", ".join(contains[:20])
        raise ValueError(
            f"药物关键词 '{requested}' 命中多个候选，请改用 --target_drug 精确指定。候选: {preview}"
        )

    raise ValueError(
        f"在 pair 文件中找不到药物 '{requested}'。"
        " 可先用 --find_drug 查询真实节点名。"
    )


def resolve_target_drug(
    pairf: str, requested: str, aliases: List[str] | None = None
) -> str:
    """解析目标药物名；若提供 aliases，则依次尝试 requested 与各别名直到命中。"""
    to_try = [requested]
    if aliases:
        to_try = [requested] + [a for a in aliases if a != requested]
    last_err = None
    for name in to_try:
        try:
            return _resolve_target_drug_one(pairf, name)
        except ValueError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise ValueError(
        f"在 pair 文件中找不到药物 '{requested}'（已尝试: {to_try}）。"
        " 可先用 --find_drug 查询真实节点名。"
    )


def _load_embeddings(embedding_file: str) -> dict:
    with open(embedding_file, "rb") as fin:
        emb = pickle.load(fin)
    if not isinstance(emb, dict) or len(emb) == 0:
        raise ValueError(f"embedding 文件无效或为空: {embedding_file}")
    return emb


def resolve_target_drug_from_embedding(
    emb: dict,
    requested: str,
    *,
    aliases: List[str] | None = None,
) -> str:
    """当目标药物不在 labeled pair 中时，尝试从 embedding 的节点名解析目标药物。"""
    keys = list(emb.keys())
    if requested in emb:
        return requested

    req_n = _normalize_text(requested)
    exact_ci = [k for k in keys if _normalize_text(k) == req_n]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(f"embedding 中存在多个仅大小写不同的候选: {exact_ci}，请显式指定 --target_drug。")

    contains = [k for k in keys if req_n and req_n in _normalize_text(k)]
    if len(contains) == 1:
        return contains[0]

    if aliases:
        for a in aliases:
            if a in emb:
                return a
        for a in aliases:
            an = _normalize_text(a)
            exact_ci = [k for k in keys if _normalize_text(k) == an]
            if len(exact_ci) == 1:
                return exact_ci[0]

    raise ValueError(
        f"目标药物 '{requested}' 不在 labeled pair，也不在 embedding 中，无法进行 case study。"
    )


def _resolve_target_microbe_one(pairf: str, requested: str) -> str:
    """解析微生物节点名：精确匹配 -> 忽略大小写精确 -> 包含匹配（唯一）。"""
    df, _, microbe_col, _ = _read_pairs(pairf)
    microbes = _unique_microbes(df, microbe_col)
    if requested in microbes:
        return requested

    req_n = _normalize_text(requested)
    exact_ci = [m for m in microbes if _normalize_text(m) == req_n]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(
            f"微生物 '{requested}' 存在多个仅大小写不同的候选: {exact_ci}，请显式指定 --target_microbe。"
        )

    contains = [m for m in microbes if req_n in _normalize_text(m)]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        preview = ", ".join(contains[:20])
        raise ValueError(
            f"微生物关键词 '{requested}' 命中多个候选，请改用 --target_microbe 精确指定。候选: {preview}"
        )

    raise ValueError(
        f"在 pair 文件中找不到微生物 '{requested}'。"
        " 可先用 --find_microbe 查询真实节点名。"
    )


def resolve_target_microbe(
    pairf: str, requested: str, aliases: List[str] | None = None
) -> str:
    to_try = [requested]
    if aliases:
        to_try = [requested] + [a for a in aliases if a != requested]
    last_err = None
    for name in to_try:
        try:
            return _resolve_target_microbe_one(pairf, name)
        except ValueError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise ValueError(
        f"在 pair 文件中找不到微生物 '{requested}'（已尝试: {to_try}）。"
        " 可先用 --find_microbe 查询真实节点名。"
    )


def resolve_target_microbe_from_embedding(
    emb: dict,
    requested: str,
    *,
    aliases: List[str] | None = None,
) -> str:
    keys = list(emb.keys())
    if requested in emb:
        return requested

    req_n = _normalize_text(requested)
    exact_ci = [k for k in keys if _normalize_text(k) == req_n]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(
            f"embedding 中存在多个仅大小写不同的微生物候选: {exact_ci}，请显式指定 --target_microbe。"
        )

    contains = [k for k in keys if req_n and req_n in _normalize_text(k)]
    if len(contains) == 1:
        return contains[0]

    if aliases:
        for a in aliases:
            if a in emb:
                return a
        for a in aliases:
            an = _normalize_text(a)
            exact_ci = [k for k in keys if _normalize_text(k) == an]
            if len(exact_ci) == 1:
                return exact_ci[0]

    raise ValueError(
        f"目标微生物 '{requested}' 不在 labeled pair，也不在 embedding 中，无法进行 case study。"
    )


def _candidate_microbes(
    nodetypef: str,
) -> List[str]:
    """nodetype 中 microbe/disease 类型节点作为候选（MDAD 等数据集中微生物标为 disease）。"""
    node_df = pd.read_csv(nodetypef, sep="\t")
    node_col = "node" if "node" in node_df.columns else node_df.columns[0]
    type_col = "type" if "type" in node_df.columns else (node_df.columns[1] if len(node_df.columns) > 1 else "type")
    microbe_types = {"microbe", "disease"}
    microbes = (
        node_df[node_df[type_col].astype(str).str.strip().str.lower().isin(microbe_types)][node_col]
        .astype(str)
        .map(str.strip)
        .tolist()
    )
    return sorted({m for m in microbes if m})


def _candidate_microbes_from_pair(pairf: str) -> List[str]:
    df, _, microbe_col, _ = _read_pairs(pairf)
    vals = df[microbe_col].astype(str).map(str.strip).tolist()
    return sorted({v for v in vals if v})


def _candidate_drugs(nodetypef: str) -> List[str]:
    """nodetype 中 type==drug 的节点作为候选药物。"""
    node_df = pd.read_csv(nodetypef, sep="\t")
    node_col = "node" if "node" in node_df.columns else node_df.columns[0]
    type_col = "type" if "type" in node_df.columns else (node_df.columns[1] if len(node_df.columns) > 1 else "type")
    drugs = (
        node_df[node_df[type_col].astype(str).str.strip().str.lower() == "drug"][node_col]
        .astype(str)
        .map(str.strip)
        .tolist()
    )
    return sorted({d for d in drugs if d})


def _candidate_drugs_from_pair(pairf: str) -> List[str]:
    df, drug_col, _, _ = _read_pairs(pairf)
    vals = df[drug_col].astype(str).map(str.strip).tolist()
    return sorted({v for v in vals if v})


def run_case_study(
    dataset: str,
    target_drug: str,
    *,
    embedding_file: str | None = None,
    pair_file: str | None = None,
    nodetype_file: str | None = None,
    out_dir: str | None = None,
    top_k: int = 20,
    seed: int = 42,
    requested_case: str | None = None,
    aliases: List[str] | None = None,
    candidate_scope: str = "nodetype_all",
) -> CaseResult:
    root = _repo_root()
    paths = _default_paths(dataset)
    pairf = pair_file or paths["pairf"]
    nodetypef = nodetype_file or paths["nodetypef"]
    emb_dir = paths["emb_dir"]

    embedding_file = _resolve_embedding_file(root, emb_dir, embedding_file)

    out_root = Path(out_dir or str(root / "runs" / "case_study"))
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    target_drug = _maybe_apply_mdad_formula_alias(target_drug)

    df, drug_col, microbe_col, label_col = _read_pairs(pairf)
    n_before = len(df)
    if n_before == 0:
        raise ValueError(f"pair 文件为空，无法训练: {pairf}")

    # 先加载 embedding：无论目标药物是否在 labeled pairs 中，都必须存在对应 drug 节点 embedding
    emb = _load_embeddings(embedding_file)

    # 1) 优先在 labeled pairs 中解析目标药物（tetracycline/vancomycin：删掉该药物的已知标注行）
    resolved_in_pair: str | None = None
    df_train = df
    removed_rows = 0
    try:
        resolved_in_pair = resolve_target_drug(pairf, target_drug, aliases=aliases)
        keep_mask = df[drug_col].astype(str).map(str.strip) != resolved_in_pair.strip()
        df_train = df[keep_mask].copy()
        removed_rows = n_before - len(df_train)
        if removed_rows > 0:
            print(f"[Cold-start] 删除目标药物在 labeled pairs 中的已知标注: {n_before} -> {len(df_train)}")
        else:
            # 在 pair 中能解析到名字但没有任何行（极少见），按 absent-from-labeled 处理
            resolved_in_pair = None
            df_train = df
            removed_rows = 0
    except ValueError:
        # 2) 目标药物不在 labeled pairs 中（acarbose：absent from labeled MDA dataset），不删行
        resolved_in_pair = None
        df_train = df
        removed_rows = 0

    # 最终用于预测阶段的目标药物名：从 embedding 解析（必要），若在 pair 中成功解析则优先使用该名字
    resolved_target = resolved_in_pair or resolve_target_drug_from_embedding(
        emb, target_drug, aliases=aliases
    )

    # 若目标药物在 pair 中的写法与 embedding 中不一致，优先使用 embedding 中的键（否则后续取 emb 会失败）
    if resolved_target not in emb:
        resolved_target = resolve_target_drug_from_embedding(emb, resolved_target, aliases=aliases)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        df_train.to_csv(f.name, sep="\t", index=False)
        train_pairf = f.name

    try:
        X_list, y_list = convert_dataset(train_pairf, embedding_file)
        X = np.asarray(X_list)
        y = np.asarray(y_list)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("训练特征为空，请检查 pair 文件与 embedding 是否匹配。")
        dim = X.shape[1]

        clf = XGBClassifier(
            base_score=0.5,
            booster="gbtree",
            eval_metric="error",
            objective="binary:logistic",
            tree_method="auto",
            scale_pos_weight=1,
            max_delta_step=1,
            seed=seed,
            **XGB_PARAMS,
        )
        clf.fit(X, y)
        print(f"[OK] XGBoost 训练完成: {resolved_target}")

        if str(candidate_scope).lower() == "pair_only":
            microbes = _candidate_microbes_from_pair(pairf)
            cand_src = "pair_only"
            print(f"[Case] 候选微生物来源=pair(训练口径), 数量={len(microbes)}")
        else:
            microbes = _candidate_microbes(nodetypef=nodetypef)
            cand_src = "nodetype_all"
            print(f"[Case] 候选微生物来源=nodetype(全图), 数量={len(microbes)}")

        results = []
        missing_microbes = 0
        for m in microbes:
            if m not in emb:
                missing_microbes += 1
                continue
            x = np.asarray(emb[resolved_target]) * np.asarray(emb[m])
            prob = float(clf.predict_proba(x.reshape(1, dim))[:, 1][0])
            known_rows = df[
                (df[drug_col].astype(str).map(str.strip) == resolved_target.strip())
                & (df[microbe_col].astype(str).map(str.strip) == m.strip())
            ]
            known_label = None
            if len(known_rows) > 0:
                try:
                    known_label = int(known_rows.iloc[0][label_col])
                except Exception:
                    known_label = None
            results.append((m, prob, known_label))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        case_tag = (requested_case or resolved_target).replace("/", "_").replace(" ", "_")
        emb_tag = Path(embedding_file).stem
        out_csv = out_root / f"case_{dataset}_{case_tag}_{emb_tag}_{cand_src}_top{top_k}.csv"
        out_json = out_root / f"case_{dataset}_{case_tag}_{emb_tag}_{cand_src}_summary.json"

        out_df = pd.DataFrame(top, columns=["microbe", "score", "known_label_in_original_pair"])
        out_df.insert(0, "rank", range(1, len(out_df) + 1))
        out_df.insert(1, "target_drug", resolved_target)
        out_df.to_csv(out_csv, index=False)

        summary = CaseResult(
            dataset=dataset,
            requested_case=requested_case or target_drug,
            resolved_target_drug=resolved_target,
            embedding_file=str(embedding_file),
            pair_file=str(pairf),
            nodetype_file=str(nodetypef),
            candidates_source=cand_src,
            train_rows_before=n_before,
            train_rows_after=len(df_train),
            removed_rows=removed_rows,
            candidate_microbe_count=len(microbes),
            scored_microbe_count=len(results),
            output_csv=str(out_csv),
            output_json=str(out_json),
            top_k=top_k,
        )
        with open(out_json, "w", encoding="utf-8") as fout:
            json.dump(asdict(summary), fout, ensure_ascii=False, indent=2)

        print(f"[OK] 结果已写入: {out_csv}")
        print(f"[OK] 摘要已写入: {out_json}")
        if missing_microbes > 0:
            print(f"[Warn] 有 {missing_microbes} 个候选微生物不在 embedding 中，已跳过。")

        print(f"\n--- {requested_case or resolved_target} Top {top_k} ---")
        for idx, (m, s, lbl) in enumerate(top, 1):
            lbl_txt = "" if lbl is None else f"  known_label={lbl}"
            print(f"{idx:2d}. {m}  {s:.6f}{lbl_txt}")

        return summary
    finally:
        Path(train_pairf).unlink(missing_ok=True)


def run_microbe_case_study(
    dataset: str,
    target_microbe: str,
    *,
    embedding_file: str | None = None,
    pair_file: str | None = None,
    nodetype_file: str | None = None,
    out_dir: str | None = None,
    top_k: int = 20,
    seed: int = 42,
    requested_case: str | None = None,
    aliases: List[str] | None = None,
    candidate_scope: str = "nodetype_all",
) -> MicrobeCaseResult:
    """固定微生物：删掉该微生物在 pair 中全部标注（microbe cold-start），对候选药物排序 Top-K。"""
    root = _repo_root()
    paths = _default_paths(dataset)
    pairf = pair_file or paths["pairf"]
    nodetypef = nodetype_file or paths["nodetypef"]
    emb_dir = paths["emb_dir"]

    embedding_file = _resolve_embedding_file(root, emb_dir, embedding_file)

    out_root = Path(out_dir or str(root / "runs" / "case_study"))
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    df, drug_col, microbe_col, label_col = _read_pairs(pairf)
    n_before = len(df)
    if n_before == 0:
        raise ValueError(f"pair 文件为空，无法训练: {pairf}")

    emb = _load_embeddings(embedding_file)

    resolved_in_pair: str | None = None
    df_train = df
    removed_rows = 0
    try:
        resolved_in_pair = resolve_target_microbe(pairf, target_microbe, aliases=aliases)
        keep_mask = df[microbe_col].astype(str).map(str.strip) != resolved_in_pair.strip()
        df_train = df[keep_mask].copy()
        removed_rows = n_before - len(df_train)
        if removed_rows > 0:
            print(
                f"[Cold-start] 删除目标微生物在 labeled pairs 中的已知标注: "
                f"{n_before} -> {len(df_train)}"
            )
        else:
            resolved_in_pair = None
            df_train = df
            removed_rows = 0
    except ValueError:
        resolved_in_pair = None
        df_train = df
        removed_rows = 0

    resolved_target = resolved_in_pair or resolve_target_microbe_from_embedding(
        emb, target_microbe, aliases=aliases
    )
    if resolved_target not in emb:
        resolved_target = resolve_target_microbe_from_embedding(
            emb, resolved_target, aliases=aliases
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        df_train.to_csv(f.name, sep="\t", index=False)
        train_pairf = f.name

    try:
        X_list, y_list = convert_dataset(train_pairf, embedding_file)
        X = np.asarray(X_list)
        y = np.asarray(y_list)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("训练特征为空，请检查 pair 文件与 embedding 是否匹配。")
        dim = X.shape[1]

        clf = XGBClassifier(
            base_score=0.5,
            booster="gbtree",
            eval_metric="error",
            objective="binary:logistic",
            tree_method="auto",
            scale_pos_weight=1,
            max_delta_step=1,
            seed=seed,
            **XGB_PARAMS,
        )
        clf.fit(X, y)
        print(f"[OK] XGBoost 训练完成 (microbe case): {resolved_target}")

        if str(candidate_scope).lower() == "pair_only":
            drugs = _candidate_drugs_from_pair(pairf)
            cand_src = "pair_only"
            print(f"[Case] 候选药物来源=pair(训练口径), 数量={len(drugs)}")
        else:
            drugs = _candidate_drugs(nodetypef=nodetypef)
            cand_src = "nodetype_all"
            print(f"[Case] 候选药物来源=nodetype(全图), 数量={len(drugs)}")

        results = []
        missing_drugs = 0
        for d in drugs:
            if d not in emb:
                missing_drugs += 1
                continue
            x = np.asarray(emb[d]) * np.asarray(emb[resolved_target])
            prob = float(clf.predict_proba(x.reshape(1, dim))[:, 1][0])
            known_rows = df[
                (df[drug_col].astype(str).map(str.strip) == d.strip())
                & (df[microbe_col].astype(str).map(str.strip) == resolved_target.strip())
            ]
            known_label = None
            if len(known_rows) > 0:
                try:
                    known_label = int(known_rows.iloc[0][label_col])
                except Exception:
                    known_label = None
            results.append((d, prob, known_label))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        case_tag = (requested_case or resolved_target).replace("/", "_").replace(" ", "_")
        emb_tag = Path(embedding_file).stem
        out_csv = out_root / f"case_{dataset}_mb_{case_tag}_{emb_tag}_{cand_src}_top{top_k}.csv"
        out_json = out_root / f"case_{dataset}_mb_{case_tag}_{emb_tag}_{cand_src}_summary.json"

        out_df = pd.DataFrame(top, columns=["drug", "score", "known_label_in_original_pair"])
        out_df.insert(0, "rank", range(1, len(out_df) + 1))
        out_df.insert(1, "target_microbe", resolved_target)
        out_df.to_csv(out_csv, index=False)

        summary = MicrobeCaseResult(
            dataset=dataset,
            requested_case=requested_case or target_microbe,
            resolved_target_microbe=resolved_target,
            embedding_file=str(embedding_file),
            pair_file=str(pairf),
            nodetype_file=str(nodetypef),
            candidates_source=cand_src,
            train_rows_before=n_before,
            train_rows_after=len(df_train),
            removed_rows=removed_rows,
            candidate_drug_count=len(drugs),
            scored_drug_count=len(results),
            output_csv=str(out_csv),
            output_json=str(out_json),
            top_k=top_k,
        )
        with open(out_json, "w", encoding="utf-8") as fout:
            json.dump(asdict(summary), fout, ensure_ascii=False, indent=2)

        print(f"[OK] 结果已写入: {out_csv}")
        print(f"[OK] 摘要已写入: {out_json}")
        if missing_drugs > 0:
            print(f"[Warn] 有 {missing_drugs} 个候选药物不在 embedding 中，已跳过。")

        print(f"\n--- microbe={requested_case or resolved_target} Top {top_k} drugs ---")
        for idx, (d, s, lbl) in enumerate(top, 1):
            lbl_txt = "" if lbl is None else f"  known_label={lbl}"
            print(f"{idx:2d}. {d}  {s:.6f}{lbl_txt}")

        return summary
    finally:
        Path(train_pairf).unlink(missing_ok=True)


def _expand_cases(cases: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for case in cases:
        key = case.strip().lower()
        if key == "paper12":
            expanded.extend(["tetracycline", "vancomycin"])
        elif key in CASE_PRESETS:
            expanded.append(key)
        else:
            expanded.append(case)
    return expanded


def _resolve_requested_target(case_name: str) -> tuple[str, List[str] | None]:
    """返回 (target_drug 首选关键词, 别名列表或 None)。"""
    key = case_name.strip().lower()
    if key in CASE_PRESETS:
        aliases = CASE_PRESETS[key]
        return aliases[0], aliases
    return case_name, None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用 microbe-drug case study 脚本")
    parser.add_argument("--dataset", type=str, default="MDAD")
    parser.add_argument("--embedding_file", type=str, default=None,
                        help="默认 main_hgat.cv5.seed42.pkl；若不存在则回退 data/<dataset>/embedding_file.pkl")
    parser.add_argument("--pair_file", type=str, default=None)
    parser.add_argument("--nodetype_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate_scope",
        choices=["nodetype_all", "pair_only"],
        default="nodetype_all",
        help="固定药物时：候选微生物集合；固定微生物时：候选药物集合（nodetype_all=全图，pair_only=仅 pair 出现）",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--target_drug", type=str, default=None, help="目标药物真实节点名（推荐精确指定）")
    group.add_argument("--target_query", type=str, default=None, help="目标药物关键词，自动解析真实节点名")
    group.add_argument("--cases", nargs="+", default=None,
                       help="预设 case 名。支持 tetracycline / vancomycin / acarbose / paper12")
    group.add_argument("--find_drug", type=str, default=None, help="只查找候选药物名，不训练")
    group.add_argument(
        "--target_microbe",
        type=str,
        default=None,
        help="固定微生物节点名，输出 Top-K 药物（microbe cold-start：删掉该微生物全部标注）",
    )
    group.add_argument("--find_microbe", type=str, default=None, help="只查找候选微生物名，不训练")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    paths = _default_paths(args.dataset)
    pairf = args.pair_file or paths["pairf"]

    if args.find_drug:
        matches = find_matching_drugs(pairf, args.find_drug)
        print(f"[Find] query={args.find_drug}")
        if matches:
            for i, drug in enumerate(matches, 1):
                print(f"{i:2d}. {drug}")
        else:
            print("未找到匹配药物。")
        return

    if args.find_microbe:
        matches = find_matching_microbes(pairf, args.find_microbe)
        print(f"[Find microbe] query={args.find_microbe}")
        if matches:
            for i, m in enumerate(matches, 1):
                print(f"{i:2d}. {m}")
        else:
            print("未找到匹配微生物。")
        return

    summaries: List[Union[CaseResult, MicrobeCaseResult]] = []

    if args.target_microbe:
        summary = run_microbe_case_study(
            dataset=args.dataset,
            target_microbe=args.target_microbe,
            embedding_file=args.embedding_file,
            pair_file=args.pair_file,
            nodetype_file=args.nodetype_file,
            out_dir=args.out_dir,
            top_k=args.top_k,
            seed=args.seed,
            requested_case=args.target_microbe,
            aliases=None,
            candidate_scope=args.candidate_scope,
        )
        summaries.append(summary)
    elif args.target_drug or args.target_query:
        requested = args.target_drug or args.target_query
        _, aliases = _resolve_requested_target(requested)
        summary = run_case_study(
            dataset=args.dataset,
            target_drug=requested,
            embedding_file=args.embedding_file,
            pair_file=args.pair_file,
            nodetype_file=args.nodetype_file,
            out_dir=args.out_dir,
            top_k=args.top_k,
            seed=args.seed,
            requested_case=requested,
            aliases=aliases,
            candidate_scope=args.candidate_scope,
        )
        summaries.append(summary)
    else:
        cases = _expand_cases(args.cases or ["paper12"])
        for case_name in cases:
            requested, aliases = _resolve_requested_target(case_name)
            print("=" * 80)
            print(f"[Run] case={case_name}")
            summary = run_case_study(
                dataset=args.dataset,
                target_drug=requested,
                embedding_file=args.embedding_file,
                pair_file=args.pair_file,
                nodetype_file=args.nodetype_file,
                out_dir=args.out_dir,
                top_k=args.top_k,
                seed=args.seed,
                requested_case=case_name,
                aliases=aliases,
                candidate_scope=args.candidate_scope,
            )
            summaries.append(summary)

    if len(summaries) > 1:
        print("\n" + "=" * 80)
        print("[Done] 汇总")
        for s in summaries:
            if isinstance(s, MicrobeCaseResult):
                print(
                    f"- microbe {s.requested_case}: {s.resolved_target_microbe}, csv={s.output_csv}"
                )
            else:
                print(f"- {s.requested_case}: target={s.resolved_target_drug}, csv={s.output_csv}")


if __name__ == "__main__":
    main()

"""
Figure 5 cold-start 案例：Acarbose (C25H43NO18) 预测 Top-K 微生物

用法（在仓库根目录）:
  python code/run_coldstart_case.py
  # 或指定参数
  python code/run_coldstart_case.py --target_drug C25H43NO18 --top_k 20
"""
from __future__ import annotations

import argparse
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from MetaMDA.utils import set_seed
from MetaMDA.predict_final import convert_dataset

XGB_PARAMS = dict(
    gamma=0,
    learning_rate=0.049,
    max_depth=12,
    n_estimators=564,
    min_child_weight=2,
    subsample=0.81,
    colsample_bytree=0.54,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths(dataset: str) -> dict:
    d = _repo_root() / "data" / dataset
    return {
        "pairf": str(d / "combined_mda.tsv"),
        "nodetypef": str(d / "demo_nodetypes.tsv"),
        "emb_dir": d / "embeddings",
    }


def run_coldstart(
    dataset: str,
    target_drug: str,
    embedding_file: str | None = None,
    pair_file: str | None = None,
    nodetype_file: str | None = None,
    out_dir: str | None = None,
    top_k: int = 20,
    seed: int = 42,
    candidates_from_pair: bool = False,
):
    root = _repo_root()
    paths = _default_paths(dataset)
    pairf = pair_file or paths["pairf"]
    nodetypef = nodetype_file or paths["nodetypef"]
    emb_dir = paths["emb_dir"]

    if embedding_file is None:
        embedding_file = str(emb_dir / "main_hgat.cv5.seed42.pkl")
    if not Path(embedding_file).is_absolute():
        embedding_file = str(root / embedding_file)

    out_dir = Path(out_dir or str(root / "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    df = pd.read_csv(pairf, sep="\t")
    drug_col = "drug" if "drug" in df.columns else df.columns[0]
    microbe_col = "disease" if "disease" in df.columns else df.columns[1]
    label_col = "label" if "label" in df.columns else df.columns[2]

    n_before = len(df)
    df_train = df[df[drug_col].astype(str).str.strip() != str(target_drug).strip()]
    n_after = len(df_train)
    if n_after == n_before and n_before > 0:
        print(f"[Cold-start] 目标药物 '{target_drug}' 不在标注中，使用全部 {n_before} 条训练。")
    else:
        print(f"[Cold-start] 去掉药物 '{target_drug}' 的标注后，训练样本: {n_before} -> {n_after}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        df_train.to_csv(f.name, sep="\t", index=False)
        train_pairf = f.name

    try:
        X_list, y_list = convert_dataset(train_pairf, embedding_file)
        X = np.array(X_list)
        y = np.array(y_list)
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
        print("[OK] XGBoost 训练完成")

        with open(embedding_file, "rb") as fin:
            emb = pickle.load(fin)
        if target_drug not in emb:
            sample = [k for k in list(emb.keys())[:15] if k.strip()]
            hint = ", ".join(sample[:5]) if sample else "(无)"
            raise KeyError(
                f"目标药物 '{target_drug}' 不在 embedding 中。\n"
                "请确认图中包含该药物节点（如通过 drug-metabolite 边），并重新生成 embedding。\n"
                f"当前 embedding 中部分节点示例: {hint}\n"
                "若仅做流程测试，可改用上述任一药物名作为 --target_drug。"
            )

        if candidates_from_pair:
            microbes = (
                df[microbe_col]
                .astype(str)
                .map(str.strip)
                .tolist()
            )
            microbes = sorted({m for m in microbes if m})
            print(f"[Cold-start] 候选微生物来自 pair 文件: {len(microbes)} 个（仅 MDAD 标注内）")
        else:
            node_df = pd.read_csv(nodetypef, sep="\t")
            microbe_type = "disease"  
            node_col = "node" if "node" in node_df.columns else node_df.columns[0]
            type_col = "type" if "type" in node_df.columns else (node_df.columns[1] if len(node_df.columns) > 1 else "type")
            microbes = (
                node_df[node_df[type_col].astype(str).str.strip() == microbe_type][node_col]
                .astype(str)
                .map(str.strip)
                .tolist()
            )
            microbes = [m for m in microbes if m]
            print(f"[Cold-start] 候选微生物来自 nodetypes 全图: {len(microbes)} 个（可能包含非 MDAD 标注微生物）")

        results = []
        for m in microbes:
            if m not in emb:
                continue
            x = emb[target_drug] * emb[m]
            prob = clf.predict_proba(x.reshape(1, dim))[:, 1][0]
            results.append((m, float(prob)))
        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        emb_tag = Path(embedding_file).stem
        cand_tag = "pair" if candidates_from_pair else "nodetype"
        out_csv = out_dir / f"coldstart_{dataset}_{target_drug.replace('/', '_')}_{emb_tag}_{cand_tag}_top{top_k}.csv"
        out_df = pd.DataFrame(top, columns=["microbe", "score"])
        out_df.insert(0, "rank", range(1, len(out_df) + 1))
        out_df.to_csv(out_csv, index=False)
        print(f"[OK] Top {top_k} 已写入: {out_csv}")

        print("\n--- Top {} 预测微生物（供文献验证 / iTOL 系统发育树）---".format(top_k))
        for r, (m, s) in enumerate(top, 1):
            print(f"  {r:2d}. {m}  {s:.4f}")
        return out_csv, top
    finally:
        Path(train_pairf).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Acarbose cold-start: 预测 Top-K 微生物，不做任何后处理")
    parser.add_argument("--dataset", type=str, default="MDAD")
    parser.add_argument("--target_drug", type=str, default="C25H43NO18", help="目标药物，默认 Acarbose")
    parser.add_argument("--embedding_file", type=str, default=None, help="默认 w_o_hypergraph（无超图，贴近论文）")
    parser.add_argument("--pair_file", type=str, default=None)
    parser.add_argument("--nodetype_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidates_from_pair",
        action="store_true",
        default=False,
        help="仅用 pair 文件内出现的微生物作为候选（pair-only）；默认 False=使用 nodetypes 全图候选",
    )
    args = parser.parse_args()

    run_coldstart(
        dataset=args.dataset,
        target_drug=args.target_drug,
        embedding_file=args.embedding_file,
        pair_file=args.pair_file,
        nodetype_file=args.nodetype_file,
        out_dir=args.out_dir,
        top_k=args.top_k,
        seed=args.seed,
        candidates_from_pair=args.candidates_from_pair,
    )


if __name__ == "__main__":
    main()

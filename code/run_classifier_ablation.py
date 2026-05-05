from __future__ import annotations

"""
Classifier ablation for MetaMDA (Figure 3 style in paper).

Default behavior:
- datasets: MDAD, aBiofilm
- model list: DNN, DT, LR, SVM, AdaBoost, GBDT, RF, XGBoost
- CV: 10-fold
- seed: 42

Output:
- CSV with per-(dataset, classifier) metrics
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

from MetaMDA.experiment_io import default_similarity_graph_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir(dataset: str) -> Path:
    return _repo_root() / "data" / dataset


def _default_paths(dataset: str) -> Dict[str, str]:
    d = _data_dir(dataset)
    return dict(
        networkf=str(d / "demo_graph.txt"),
        simf=default_similarity_graph_path(d),
        nodetypef=str(d / "demo_nodetypes.tsv"),
        nodef=str(d / "demo_graph_node.txt"),
        simnodef=str(d / "demo_similarity_graph_node.txt"),
        pairf=str(d / "combined_mda.tsv"),
    )


def _xgb_params() -> Dict[str, float | int]:
    # Keep consistent with the project default paper setting.
    return dict(
        gamma=0,
        rate=0.049,
        max_depth=12,
        n_estimators=564,
        min_child_weight=2,
        subsample=0.81,
        colsample_bytree=0.54,
    )


def _write_rows(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "cv",
        "seed",
        "classifier",
        "embedding_file",
        "pair_file",
        "acc",
        "auroc",
        "aupr",
        "f1",
        "average",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classifier ablation experiment for MetaMDA.")
    parser.add_argument("--datasets", nargs="+", default=["MDAD", "aBiofilm"])
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["DNN", "DT", "LR", "SVM", "AdaBoost", "GBDT", "RF", "XGBoost"],
        help="Supported names in MetaMDA.predict_final.predict_mda",
    )
    parser.add_argument("--cv", type=int, default=10, help="KFold splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_tag", type=str, default="main_hgat_cv10_seed42")
    parser.add_argument("--out_csv", type=str, default="runs/classifier_ablation_cv10_seed42.csv")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    from MetaMDA.generate_embeddings_mmd import save_embedding_files
    from MetaMDA.predict_final import predict_mda

    os.chdir(_repo_root())
    xgb = _xgb_params()
    rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        p = _default_paths(dataset)
        emb_dir = _data_dir(dataset) / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        embeddingf = str(emb_dir / f"{args.embedding_tag}.pkl")

        # Generate one embedding file, then compare different classifiers fairly.
        save_embedding_files(
            netf=p["networkf"],
            sim_netf=p["simf"],
            outputf=embeddingf,
            nodetypef=p["nodetypef"],
            nodef=p["nodef"],
            simnodef=p["simnodef"],
            dimension=1024,
            window_size=8,
            t=0.9,
            seed=int(args.seed),
            use_hypergraph=True,
            hg_eval_n_splits=int(args.cv),
            hg_k=20,
            hg_layers=2,
            hg_tau=0.6,
            hg_safe_eval=True,
            hg_alpha_candidates=[0.0, 0.5, 0.8, 0.9, 0.95, 1.0],
            hg_pairf=p["pairf"],
            hg_xgb_params=dict(
                gamma=xgb["gamma"],
                learning_rate=xgb["rate"],
                max_depth=xgb["max_depth"],
                n_estimators=xgb["n_estimators"],
                min_child_weight=xgb["min_child_weight"],
                subsample=xgb["subsample"],
                colsample_bytree=xgb["colsample_bytree"],
            ),
            hg_trainable_hidden_dim=256,
            hg_trainable_epochs=400,
            hg_trainable_patience=50,
            hg_trainable_gate_lambda=0.003,
            hg_trainable_gate_lambda_sweep=None,
            hg_trainable_use_node_gate=True,
            hg_trainable_use_multihead_fusion=True,
            hg_trainable_num_heads=4,
            hg_trainable_verbose=True,
        )

        for clf_name in args.classifiers:
            result = predict_mda(
                embeddingf=embeddingf,
                pairf=p["pairf"],
                modelf=clf_name,
                seed=int(args.seed),
                n_splits=int(args.cv),
                plot=not args.no_plot,
                gamma=float(xgb["gamma"]),
                rate=float(xgb["rate"]),
                max_depth=int(xgb["max_depth"]),
                n_estimators=int(xgb["n_estimators"]),
                min_child_weight=int(xgb["min_child_weight"]),
                subsample=float(xgb["subsample"]),
                colsample_bytree=float(xgb["colsample_bytree"]),
            )
            acc, auroc, aupr, f1 = [float(v) for v in result]
            avg = (acc + auroc + aupr + f1) / 4.0
            row = dict(
                dataset=dataset,
                cv=int(args.cv),
                seed=int(args.seed),
                classifier=clf_name,
                embedding_file=embeddingf,
                pair_file=p["pairf"],
                acc=acc,
                auroc=auroc,
                aupr=aupr,
                f1=f1,
                average=avg,
            )
            rows.append(row)
            print(
                f"[OK] dataset={dataset} clf={clf_name} "
                f"AUROC={auroc:.4f} AUPR={aupr:.4f} ACC={acc:.4f} F1={f1:.4f}"
            )

    _write_rows(rows, Path(args.out_csv) if Path(args.out_csv).is_absolute() else _repo_root() / args.out_csv)


if __name__ == "__main__":
    main()


"""
Batch runner for MetaMDA experiments.

This script keeps the training/embedding logic unchanged, and standardizes:
- CLI configuration
- structured result logging to CSV

Run example (from repo root):
  python code/run_all_exps.py --datasets MDAD aBiofilm MASI --cv 5 10 --seeds 42 43 --no_plot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

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


def _base_config() -> Dict[str, Any]:
    return dict(
        # MDAD 默认：各单变量扫描中取 auroc_mean 最大（runs/param_*_MDAD_cv*_seed42.csv）
        # t=0.9, window=8, dim=1024, hg_tau=0.6；相似图见 default_similarity_graph_path（α=0.7）
        t=0.9,
        dimension=1024,
        window_size=8,
        # hypergraph defaults
        use_hypergraph=True,
        # ===== 主配置：full hyperedges（把 KNN-only 作为对照/消融）=====
        hg_use_gene_hyperedges=True,
        hg_use_path_hyperedges=True,
        hg_use_similarity_hyperedges=True,
        hg_random_hyperedges=False,
        hg_k=20,
        hg_layers=2,
        hg_tau=0.6,
        hg_safe_eval=True,
        hg_alpha_candidates=[0.0, 0.5, 0.8, 0.9, 0.95, 1.0],
        hg_trainable_hidden_dim=256,
        hg_trainable_epochs=400,
        hg_trainable_patience=50,
        hg_trainable_gate_lambda=0.003,
        hg_trainable_gate_lambda_sweep=None,
        hg_trainable_use_node_gate=True,
        hg_trainable_use_multihead_fusion=True,
        hg_trainable_num_heads=4,
        hg_trainable_verbose=True,
        # XGBoost params used inside hypergraph safe-eval
        hg_xgb_params=dict(
            gamma=0,
            learning_rate=0.049,
            max_depth=12,
            n_estimators=564,
            min_child_weight=2,
            subsample=0.81,
            colsample_bytree=0.54,
        ),
    )


def build_experiments() -> List[Dict[str, Any]]:
    """
    Define experiment switches here.

    Notes:
    - Base embedding is DREAMwalk + HeterogeneousSG.
    - Hypergraph enhancement is optional via `use_hypergraph`.
    """
    base = _base_config()

    exps: List[Dict[str, Any]] = []

    exps.append(dict(exp_name="main_hgat", **base))
    exps.append(dict(exp_name="w_o_hypergraph", **{**base, "use_hypergraph": False}))

    # fusion ablation: no multi-head fusion (equal-weight)
    exps.append(
        dict(
            exp_name="avg_fusion",
            **{
                **base,
                "hg_trainable_use_multihead_fusion": False,
            },
        )
    )

    # hyperedge type ablation: remove path hyperedges only
    exps.append(
        dict(
            exp_name="w_o_path_hyperedge",
            **{
                **base,
                "hg_use_path_hyperedges": False,
            },
        )
    )

    # hyperedge construction ablation: KNN-only (disable gene/path/sim hyperedges)
    exps.append(
        dict(
            exp_name="knn_only_hyperedge",
            **{
                **base,
                "hg_use_gene_hyperedges": False,
                "hg_use_path_hyperedges": False,
                "hg_use_similarity_hyperedges": False,
                "hg_random_hyperedges": False,
            },
        )
    )

    return exps


def build_fusion_experiments() -> List[Dict[str, Any]]:
    """Compare fusion strategies only: new fusion vs original MHA vs equal-weight."""
    base = _base_config()

    exps: List[Dict[str, Any]] = []

    # Original fusion: MHA-based view fusion
    exps.append(
        dict(
            exp_name="fusion_mha",
            **{
                **base,
                "hg_trainable_use_multihead_fusion": True,
            },
        )
    )

    # Equal-weight fusion
    exps.append(
        dict(
            exp_name="fusion_avg",
            **{
                **base,
                "hg_trainable_use_multihead_fusion": False,
            },
        )
    )
    return exps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["MDAD", "aBiofilm", "MASI"])
    parser.add_argument("--cv", nargs="+", type=int, default=[5], help="KFold splits (e.g. 5 10)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--results_csv", type=str, default=str(_repo_root() / "runs" / "results.csv"))
    parser.add_argument("--no_plot", action="store_true", help="disable ROC plotting in predict_mda")
    parser.add_argument(
        "--suite",
        choices=["all", "fusion"],
        default="all",
        help="which experiment suite to run",
    )
    args = parser.parse_args()

    from MetaMDA.generate_embeddings_mmd import save_embedding_files
    from MetaMDA.predict_final import predict_mda
    from MetaMDA.experiment_io import append_result_csv

    experiments = build_fusion_experiments() if args.suite == "fusion" else build_experiments()

    for dataset in args.datasets:
        p = _default_paths(dataset)
        for cv in args.cv:
            for seed in args.seeds:
                for exp in experiments:
                    exp_name = exp["exp_name"]

                    emb_dir = _data_dir(dataset) / "embeddings"
                    emb_dir.mkdir(parents=True, exist_ok=True)
                    embeddingf = str(emb_dir / f"{exp_name}.cv{cv}.seed{seed}.pkl")

                    save_embedding_files(
                        netf=p["networkf"],
                        sim_netf=p["simf"],
                        outputf=embeddingf,
                        nodetypef=p["nodetypef"],
                        nodef=p["nodef"],
                        simnodef=p["simnodef"],
                        dimension=int(exp.get("dimension", 1024)),
                        window_size=int(exp.get("window_size", 8)),
                        t=float(exp.get("t", 0.9)),
                        seed=int(seed),
                        use_hypergraph=bool(exp.get("use_hypergraph", False)),
                        hg_eval_n_splits=int(cv),
                        hg_use_gene_hyperedges=bool(exp.get("hg_use_gene_hyperedges", True)),
                        hg_use_path_hyperedges=bool(exp.get("hg_use_path_hyperedges", True)),
                        hg_use_similarity_hyperedges=bool(exp.get("hg_use_similarity_hyperedges", True)),
                        hg_path_length=int(exp.get("hg_path_length", 2)),
                        hg_knn_from_embeddings=bool(exp.get("hg_knn_from_embeddings", True)),
                        hg_k=int(exp.get("hg_k", 20)),
                        hg_layers=int(exp.get("hg_layers", 2)),
                        hg_tau=float(exp.get("hg_tau", 0.6)),
                        hg_safe_eval=bool(exp.get("hg_safe_eval", True)),
                        hg_alpha_candidates=exp.get("hg_alpha_candidates", None),
                        hg_alpha_fixed=exp.get("hg_alpha_fixed", None),
                        hg_pairf=p["pairf"],
                        hg_xgb_params=exp.get("hg_xgb_params", None),
                        hg_trainable_hidden_dim=int(exp.get("hg_trainable_hidden_dim", 256)),
                        hg_trainable_epochs=int(exp.get("hg_trainable_epochs", 400)),
                        hg_trainable_patience=int(exp.get("hg_trainable_patience", 50)),
                        hg_trainable_gate_lambda=float(exp.get("hg_trainable_gate_lambda", 1e-3)),
                        hg_trainable_gate_lambda_sweep=exp.get("hg_trainable_gate_lambda_sweep", None),
                        hg_trainable_use_node_gate=bool(exp.get("hg_trainable_use_node_gate", True)),
                        hg_trainable_use_multihead_fusion=bool(exp.get("hg_trainable_use_multihead_fusion", True)),
                        hg_trainable_num_heads=int(exp.get("hg_trainable_num_heads", 4)),
                        hg_trainable_verbose=bool(exp.get("hg_trainable_verbose", True)),
                    )

                    scores = predict_mda(
                        embeddingf=embeddingf,
                        pairf=p["pairf"],
                        modelf="XGBoost",
                        seed=int(seed),
                        n_splits=int(cv),
                        plot=not args.no_plot,
                        return_details=True,
                    )

                    row = dict(
                        exp_name=exp_name,
                        dataset=dataset,
                        cv=int(cv),
                        seed=int(seed),
                        model=scores.get("model"),
                        embedding_file=embeddingf,
                        pair_file=p["pairf"],
                        acc=float(scores["acc"]),
                        auroc=float(scores["auroc"]),
                        aupr=float(scores["aupr"]),
                        f1=float(scores["f1"]),
                        acc_mean=float(scores.get("fold_acc_pct_mean", float("nan"))),
                        acc_std=float(scores.get("fold_acc_pct_std", float("nan"))),
                        auroc_mean=float(scores.get("fold_auroc_mean", float("nan"))),
                        auroc_std=float(scores.get("fold_auroc_std", float("nan"))),
                        aupr_mean=float(scores.get("fold_aupr_mean", float("nan"))),
                        aupr_std=float(scores.get("fold_aupr_std", float("nan"))),
                        f1_mean=float(scores.get("fold_f1_mean", float("nan"))),
                        f1_std=float(scores.get("fold_f1_std", float("nan"))),
                        config_json=json.dumps(exp, ensure_ascii=False, sort_keys=True),
                    )
                    append_result_csv(row, out_csv=args.results_csv)
                    print(
                        f"[OK] dataset={dataset} cv={cv} seed={seed} exp={exp_name} -> AUROC={row['auroc']:.4f} AUPR={row['aupr']:.4f}"
                    )


if __name__ == "__main__":
    os.chdir(_repo_root())
    main()


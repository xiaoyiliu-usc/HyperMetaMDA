from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir(dataset: str) -> Path:
    return _repo_root() / "data" / dataset


def _base_config() -> Dict[str, Any]:
    return dict(
        t=0.9,
        dimension=1024,
        window_size=8,
        use_hypergraph=True,
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


def _default_graph_paths(dataset: str, alpha: str) -> Dict[str, str]:
    d = _data_dir(dataset)
    return dict(
        networkf=str(d / "demo_graph.txt"),
        simf=str(d / f"run_similarity_graph_{alpha}.txt"),
        nodetypef=str(d / "demo_nodetypes.tsv"),
        nodef=str(d / "demo_graph_node.txt"),
        # Keep similarity-node universe fixed across alphas to avoid missing-node
        # crashes during random walk (some nodes may have zero sim edges at high alpha).
        simnodef=str(d / "demo_similarity_graph_node.txt"),
        pairf=str(d / "combined_mda.tsv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MDAD 相似度阈值 alpha 敏感性实验")
    parser.add_argument("--dataset", type=str, default="MDAD")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["0.5", "0.55", "0.6", "0.65", "0.70", "0.75", "0.8"],
        help="相似度阈值列表，对应 data/<dataset>/run_similarity_graph_<alpha>.txt",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="输出 CSV（默认 runs/param_sim_alpha_<dataset>_cv<k>_seed<s>.csv）",
    )
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    from MetaMDA.experiment_io import append_result_csv
    from MetaMDA.generate_embeddings_mmd import save_embedding_files
    from MetaMDA.predict_final import predict_mda

    os.chdir(_repo_root())
    dataset = args.dataset
    cv = int(args.cv)
    seed = int(args.seed)
    base = _base_config()

    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else str(_repo_root() / "runs" / f"param_sim_alpha_{dataset}_cv{cv}_seed{seed}.csv")
    )

    emb_dir = _data_dir(dataset) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    for alpha_raw in args.alphas:
        # Normalize alpha string to match filenames like:
        # - run_similarity_graph_0.5.txt
        # - run_similarity_graph_0.65.txt
        # - run_similarity_graph_0.7.txt (NOT 0.70)
        alpha_val = float(alpha_raw)
        alpha_tag = f"{alpha_val:.2f}".rstrip("0").rstrip(".")
        p = _default_graph_paths(dataset, alpha_tag)
        simf = Path(p["simf"])
        simnodef = Path(p["simnodef"])
        if not simf.exists():
            raise FileNotFoundError(f"缺少相似图文件: {simf}")
        if not simnodef.exists():
            raise FileNotFoundError(f"缺少相似图节点文件: {simnodef}")

        exp = dict(base)
        exp_name = f"main_hgat_sim_alpha_{alpha_tag.replace('.', '_')}"
        exp["exp_name"] = exp_name
        exp["sim_alpha"] = float(alpha_val)

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
            seed=seed,
            use_hypergraph=bool(exp.get("use_hypergraph", False)),
            hg_eval_n_splits=cv,
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
            seed=seed,
            n_splits=cv,
            plot=not args.no_plot,
            return_details=True,
        )

        row = dict(
            dataset=dataset,
            param="sim_alpha",
            param_value=float(alpha_val),
            cv=cv,
            seed=seed,
            exp_name=exp_name,
            model=scores.get("model"),
            sim_file=p["simf"],
            sim_node_file=p["simnodef"],
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
        append_result_csv(row, out_csv=out_csv)
        print(
            f"[OK] dataset={dataset} sim_alpha={alpha_tag} "
            f"-> AUROC={row['auroc']:.4f} AUPR={row['aupr']:.4f}"
        )


if __name__ == "__main__":
    main()

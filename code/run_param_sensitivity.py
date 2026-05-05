from __future__ import annotations

"""
超参数敏感性实验脚本：
- 原论文参数实验：t / window_size / dimension（相似度阈值 α 需要多份相似度图文件支持）
- 仓库扩展：支持超图相关参数敏感性（hg_tau / hg_k / hg_layers / hg_gate_lambda）
- 随机游走参数 t
- 窗口大小 window_size
- 嵌入维度 dimension

用法示例（在仓库根目录下）：

1) 对 t 做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param t \\
       --values 0.75 0.80 0.85 0.90 0.95 \\
       --cv 5 --seed 42 --out_csv runs/param_t_mdAD_cv5_seed42.csv

2) 对 window_size 做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param window_size \\
       --values 4 5 6 7 8 9 10 \\
       --cv 5 --seed 42 --out_csv runs/param_window_MDAD_cv5_seed42.csv

3) 对 embedding 维度做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param dimension \\
       --values 64 128 256 512 1024 \\
       --cv 5 --seed 42 --out_csv runs/param_dim_MDAD_cv5_seed42.csv

4) 对超图温度/平滑系数 hg_tau 做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param hg_tau \\
       --values 0.30 0.40 0.50 0.60 0.70 \\
       --cv 5 --seed 42 --out_csv runs/param_hg_tau_MDAD_cv5_seed42.csv

5) 对 KNN 超边邻居数 hg_k 做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param hg_k \\
       --values 10 15 20 25 30 \\
       --cv 5 --seed 42 --out_csv runs/param_hg_k_MDAD_cv5_seed42.csv

6) 对超图层数 hg_layers 做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param hg_layers \\
       --values 1 2 3 4 \\
       --cv 5 --seed 42 --out_csv runs/param_hg_layers_MDAD_cv5_seed42.csv

7) 对可训练超图门控正则系数 gate_lambda 做敏感性实验：
   python code/run_param_sensitivity.py --dataset MDAD --param hg_gate_lambda \\
       --values 0.0005 0.001 0.003 0.005 0.01 \\
       --cv 5 --seed 42 --out_csv runs/param_hg_gate_lambda_MDAD_cv5_seed42.csv

脚本会在指定 CSV 中记录每个参数取值下的 AUROC/AUPR/ACC/F1 等指标，
方便后续用单独的绘图脚本画出折线图。
"""

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
    # 与 code/run_all_exps.py 中保持一致，便于对比
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


def build_param_experiments(param: str, values: List[float | int]) -> List[Dict[str, Any]]:
    """
    根据要扫描的参数名和取值列表，构造一组实验配置。
    每个配置都只改一个参数，其余保持与主实验一致。
    """
    base = _base_config()
    exps: List[Dict[str, Any]] = []

    for v in values:
        cfg = dict(base)
        if param == "t":
            cfg["t"] = float(v)
            exp_name = f"main_hgat_t_{v}"
        elif param == "window_size":
            cfg["window_size"] = int(v)
            exp_name = f"main_hgat_win_{v}"
        elif param == "dimension":
            cfg["dimension"] = int(v)
            exp_name = f"main_hgat_dim_{v}"
        elif param == "hg_tau":
            cfg["hg_tau"] = float(v)
            exp_name = f"main_hgat_hg_tau_{v}"
        elif param == "hg_k":
            cfg["hg_k"] = int(v)
            exp_name = f"main_hgat_hg_k_{v}"
        elif param == "hg_layers":
            cfg["hg_layers"] = int(v)
            exp_name = f"main_hgat_hg_layers_{v}"
        elif param == "hg_gate_lambda":
            cfg["hg_trainable_gate_lambda"] = float(v)
            exp_name = f"main_hgat_hg_gate_lambda_{v}"
        else:
            raise ValueError(f"未知参数类型: {param}")

        cfg["exp_name"] = exp_name
        exps.append(cfg)

    return exps


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaMDA 超参数敏感性实验脚本")
    parser.add_argument("--dataset", type=str, default="MDAD")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--param",
        type=str,
        choices=[
            "t",
            "window_size",
            "dimension",
            # hypergraph params
            "hg_tau",
            "hg_k",
            "hg_layers",
            "hg_gate_lambda",
        ],
        required=True,
        help="要做敏感性分析的参数名",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="该参数的一组取值，例如: 0.75 0.8 0.85 0.9 0.95",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="结果输出 CSV 路径（默认放到 runs/ 下，文件名自动生成）",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="禁用 predict_mda 中的 ROC 绘图",
    )
    args = parser.parse_args()

    # 将值解析成 float / int，便于后续使用
    parsed_values: List[float | int] = []
    for v in args.values:
        try:
            if "." in v:
                parsed_values.append(float(v))
            else:
                parsed_values.append(int(v))
        except ValueError:
            parsed_values.append(float(v))

    from MetaMDA.generate_embeddings_mmd import save_embedding_files
    from MetaMDA.predict_final import predict_mda
    from MetaMDA.experiment_io import append_result_csv

    os.chdir(_repo_root())

    experiments = build_param_experiments(args.param, parsed_values)

    dataset = args.dataset
    paths = _default_paths(dataset)

    cv = int(args.cv)
    seed = int(args.seed)

    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else str(
            _repo_root()
            / "runs"
            / f"param_{args.param}_{dataset}_cv{cv}_seed{seed}.csv"
        )
    )

    emb_dir = _data_dir(dataset) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        exp_name = exp["exp_name"]

        embeddingf = str(emb_dir / f"{exp_name}.cv{cv}.seed{seed}.pkl")
        # 1) 生成/更新 embedding
        save_embedding_files(
            netf=paths["networkf"],
            sim_netf=paths["simf"],
            outputf=embeddingf,
            nodetypef=paths["nodetypef"],
            nodef=paths["nodef"],
            simnodef=paths["simnodef"],
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
            hg_pairf=paths["pairf"],
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

        # 2) 运行 MDA 预测并得到四个指标
        scores = predict_mda(
            embeddingf=embeddingf,
            pairf=paths["pairf"],
            modelf="XGBoost",
            seed=seed,
            n_splits=cv,
            plot=not args.no_plot,
            return_details=True,
        )

        row = dict(
            dataset=dataset,
            param=args.param,
            param_value=exp.get(args.param)
            if args.param in exp
            else exp.get("hg_trainable_gate_lambda"),
            cv=cv,
            seed=seed,
            exp_name=exp_name,
            model=scores.get("model"),
            embedding_file=embeddingf,
            pair_file=paths["pairf"],
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
            f"[OK] dataset={dataset} param={args.param} value={row['param_value']} "
            f"-> AUROC={row['auroc']:.4f} AUPR={row['aupr']:.4f}"
        )


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

EXP_NAMES = [
    "main_hgat",
    "w_o_hypergraph",
    "avg_fusion",
    "w_o_path_hyperedge",
    "knn_only_hyperedge",
]


def _root_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_dir(root: Path) -> Path:
    candidates = [
        root / "mda",
        root / "HyperMetamda",
        root / "copy(crash)_副本" / "HyperMetamda",
        root,
    ]
    for p in candidates:
        if (p / "MetaMDA").exists() and (p / "实验脚本" / "run_all_exps.py").exists():
            return p
        if (p / "code" / "MetaMDA").exists() and (p / "code" / "实验脚本" / "run_all_exps.py").exists():
            return p
        if (p / "code" / "MetaMDA").exists() and (p / "code" / "run_all_exps.py").exists():
            return p
    raise FileNotFoundError("未找到项目目录（需包含 MetaMDA 与 run_all_exps.py）。")


def _code_root(project: Path) -> Path:
    if (project / "MetaMDA").exists():
        return project
    if (project / "code" / "MetaMDA").exists():
        return project / "code"
    raise FileNotFoundError(f"未找到 code 根目录: {project}")


def _load_module(mod_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script_path(code_root: Path, script_name: str) -> Path:
    candidates = [
        code_root / "实验脚本" / script_name,
        code_root / script_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到脚本: {script_name}（已尝试: {candidates}）")


def _run_ablation(
    project: Path,
    out_csv: Path,
    datasets: list[str],
    cv: int,
    seed: int,
    use_existing_embeddings: bool = True,
) -> None:
    code_root = _code_root(project)
    sys.path.insert(0, str(code_root))
    runner = _load_module(_script_path(code_root, "run_all_exps.py"), "run_all_exps_mod")
    # 兼容 HyperMetamda/code 目录结构，确保 run_all_exps 使用 project/data
    runner._repo_root = lambda: project  # type: ignore[attr-defined]

    # 优先复用已有 embedding；若复用文件与当前 pair 不一致会在 predict 阶段 KeyError，此时自动删文件并重算一次。
    emb_mod = importlib.import_module("MetaMDA.generate_embeddings_mmd")
    pred_mod = importlib.import_module("MetaMDA.predict_final")
    original_save_embedding_files = emb_mod.save_embedding_files
    original_predict_mda = pred_mod.predict_mda
    last_emb_call_by_path: dict[str, tuple[tuple, dict]] = {}

    def _save_embedding_files_with_cache(*args, **kwargs):
        outputf = kwargs.get("outputf")
        if outputf is None and len(args) >= 3:
            outputf = args[2]
        output_path = Path(str(outputf)).resolve() if outputf is not None else None
        if output_path is not None:
            last_emb_call_by_path[str(output_path)] = (args, dict(kwargs))
        if use_existing_embeddings and output_path is not None and output_path.exists():
            print(f"[Reuse] 已存在 embedding，跳过生成: {output_path}")
            return None
        return original_save_embedding_files(*args, **kwargs)

    def _predict_mda_with_recover(*p_args, **p_kwargs):
        embeddingf = p_kwargs.get("embeddingf")
        if embeddingf is None and p_args:
            embeddingf = p_args[0]
        try:
            return original_predict_mda(*p_args, **p_kwargs)
        except KeyError as err:
            if not use_existing_embeddings:
                raise
            emb_key = str(Path(str(embeddingf)).resolve())
            stored = last_emb_call_by_path.get(emb_key)
            if stored is None:
                raise RuntimeError(
                    f"predict_mda KeyError {err!r}，且无缓存的生成参数，无法自动重算: {embeddingf}"
                ) from err
            print(f"[Recover] KeyError {err!r}，删除并重算 embedding: {embeddingf}")
            Path(str(embeddingf)).unlink(missing_ok=True)
            s_args, s_kw = stored
            original_save_embedding_files(*s_args, **s_kw)
            return original_predict_mda(*p_args, **p_kwargs)

    emb_mod.save_embedding_files = _save_embedding_files_with_cache
    pred_mod.predict_mda = _predict_mda_with_recover

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "run_all_exps.py",
            "--datasets",
            *datasets,
            "--cv",
            str(cv),
            "--seeds",
            str(seed),
            "--results_csv",
            str(out_csv),
            "--no_plot",
        ]
        runner.main()
    finally:
        emb_mod.save_embedding_files = original_save_embedding_files
        pred_mod.predict_mda = original_predict_mda
        sys.argv = old_argv


def _embedding_paths(project: Path, datasets: list[str], cv: int, seed: int) -> list[Path]:
    paths: list[Path] = []
    for ds in datasets:
        emb_dir = project / "data" / ds / "embeddings"
        for exp in EXP_NAMES:
            paths.append(emb_dir / f"{exp}.cv{cv}.seed{seed}.pkl")
    return paths


def _decide_reuse_mode(project: Path, datasets: list[str], cv: int, seed: int, prefer_reuse: bool) -> bool:
    if not prefer_reuse:
        return False
    required = _embedding_paths(project, datasets, cv, seed)
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[Warn] 你选择了复用 embedding，但以下文件缺失，流程可能在运行中报错：")
        for p in missing[:8]:
            print(f"  - 缺失: {p}")
        if len(missing) > 8:
            print(f"  - ... 共缺失 {len(missing)} 个")
   
    return True


def _validate_inputs(project: Path, datasets: list[str]) -> None:
    missing: list[Path] = []
    for ds in datasets:
        d = project / "data" / ds
        required = [
            d / "demo_graph.txt",
            d / "demo_nodetypes.tsv",
            d / "demo_graph_node.txt",
            d / "demo_similarity_graph_node.txt",
            d / "combined_mda.tsv",
        ]
        for p in required:
            if not p.exists():
                missing.append(p)
    if missing:
        joined = "\n".join(f"- {p}" for p in missing[:12])
        more = "" if len(missing) <= 12 else f"\n- ... 共缺失 {len(missing)} 个文件"
        raise FileNotFoundError(
            "数据文件不完整，无法开始消融实验。请先准备数据集文件：\n"
            f"{joined}{more}"
        )


def _split_dataset_csv(results_csv: Path, out_dir: Path, cv: int, seed: int) -> list[Path]:
    df = pd.read_csv(results_csv)
    dataset_order = ["MDAD", "aBiofilm", "MASI"]
    outputs: list[Path] = []

    for ds in dataset_order:
        sub = df[(df["dataset"] == ds) & (df["cv"] == cv) & (df["seed"] == seed)].copy()
        if len(sub) == 0:
            continue
        # 同一 exp_name 可能因中断重跑被追加多次；保留最新一条，避免画图出现重复柱子。
        if "exp_name" in sub.columns:
            if "timestamp_utc" in sub.columns:
                sub["timestamp_utc"] = pd.to_datetime(sub["timestamp_utc"], errors="coerce")
                sub = sub.sort_values(["exp_name", "timestamp_utc"], kind="stable")
            sub = sub.drop_duplicates(subset=["exp_name"], keep="last")
        ds_tag = "masi" if ds.upper() == "MASI" else ds
        out_path = out_dir / f"ablation_{ds_tag}_cv{cv}_seed{seed}.csv"
        sub.to_csv(out_path, index=False)
        outputs.append(out_path)
    return outputs


def _plot_ablation(project: Path, csvs: list[Path], out_dir: Path, out_name: str) -> Path:
    plot_mod = _load_module(_script_path(_code_root(project), "plot_ablation_bars.py"), "plot_ablation_mod")
    out_path = out_dir / out_name

    df_list = [plot_mod.load_ablation_csv(str(p)) for p in csvs]
    name_list = []
    for p in csvs:
        raw = p.stem.replace("ablation_", "").split("_cv")[0]
        name_list.append(plot_mod.DATASET_LABELS.get(raw, raw))
    plot_mod.plot_ablation_like_reference(df_list, name_list, str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行消融实验（含 embedding 生成）并输出数据与图。")
    parser.add_argument("--datasets", nargs="+", default=["MDAD", "aBiofilm", "MASI"])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results", help="拆分 CSV 与柱状图输出目录（相对脚本所在目录）。")
    parser.add_argument("--results_csv", type=str, default="results/ablation_results.csv", help="汇总结果 CSV 路径。")
    parser.add_argument("--plot_name", type=str, default="ablation_bars__.png")
    parser.add_argument(
        "--no_reuse_embedding",
        action="store_true",
        help="不复用已有 embedding，全部强制重新生成（且关闭 KeyError 时的自动重算）。",
    )
    args = parser.parse_args()

    root = _root_dir()
    project = _project_dir(root)
    if not args.datasets:
        raise RuntimeError(f"未发现可用数据集目录：{project / 'data'}")
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_csv) if Path(args.results_csv).is_absolute() else root / args.results_csv
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    print("开始运行实验1（消融）...")
    print(f"项目目录: {project}")
    print(f"输出 CSV: {results_csv}")
    _validate_inputs(project, args.datasets)

    _run_ablation(
        project=project,
        out_csv=results_csv,
        datasets=args.datasets,
        cv=args.cv,
        seed=args.seed,
        use_existing_embeddings=_decide_reuse_mode(
            project=project,
            datasets=args.datasets,
            cv=args.cv,
            seed=args.seed,
            prefer_reuse=not args.no_reuse_embedding,
        ),
    )

    split_csvs = _split_dataset_csv(results_csv, out_dir, cv=args.cv, seed=args.seed)
    if not split_csvs:
        raise RuntimeError("未生成任何按数据集拆分的 ablation CSV，请检查运行日志。")

    plot_path = _plot_ablation(project, split_csvs, out_dir, args.plot_name)

    print("完成")
    print(f"results_csv: {results_csv}")
    print("split_csvs:")
    for p in split_csvs:
        print(f"  - {p}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()

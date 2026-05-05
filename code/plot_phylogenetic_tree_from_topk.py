"""
根据 Top-K 微生物预测结果自动构建“分类学系统发育树”并绘图。

默认流程：
1) 读取 CSV（至少包含 `microbe` 列）
2) 调用 NCBI Taxonomy（Entrez）解析每个微生物对应的 taxid 与 lineage
3) 按 lineage 构建层级树（Newick）
4) 画出系统发育树 PNG（可选高亮指定叶子）

示例：
python code/plot_phylogenetic_tree_from_topk.py \
  --input_csv runs/case_study/case_MDAD_acarbose_main_hgat.cv5.seed42_nodetype_top20.csv \
  --out_prefix runs/case_study/acarbose_main_hgat_top20_phylo \
  --highlight "Pseudomonas_aeruginosa,Candida_albicans,Staphylococcus_aureus,Escherichia_coli"
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from Bio import Entrez, Phylo
from Bio.Phylo.Newick import Clade, Tree
import matplotlib.pyplot as plt


def _clean_name(name: str) -> str:
    return str(name).strip().replace("_", " ")


def _search_taxid(name: str, sleep_s: float = 0.35) -> Optional[str]:
    """优先按 Scientific Name 精确搜，失败后退化到 All Names。"""
    exact_term = f"\"{name}\"[Scientific Name]"
    fallback_term = f"\"{name}\"[All Names]"
    for term in (exact_term, fallback_term):
        try:
            with Entrez.esearch(db="taxonomy", term=term, retmax=1) as handle:
                rec = Entrez.read(handle)
            ids = rec.get("IdList", [])
            if ids:
                time.sleep(sleep_s)
                return ids[0]
        except Exception:
            pass
    return None


def _fetch_lineage(taxid: str, sleep_s: float = 0.35) -> Optional[List[str]]:
    """返回从高到低的 lineage（不含当前物种名）。"""
    try:
        with Entrez.efetch(db="taxonomy", id=taxid, retmode="xml") as handle:
            recs = Entrez.read(handle)
        if not recs:
            return None
        rec = recs[0]
        lineage = [x.strip() for x in rec.get("Lineage", "").split(";") if x.strip()]
        time.sleep(sleep_s)
        return lineage
    except Exception:
        return None


def _build_tree_from_paths(paths: List[List[str]], leaves: List[str]) -> Tree:
    root = Clade(name="root")
    for path, leaf in zip(paths, leaves):
        node = root
        for taxon in path:
            child = next((c for c in node.clades if c.name == taxon), None)
            if child is None:
                child = Clade(name=taxon)
                node.clades.append(child)
            node = child
        node.clades.append(Clade(name=leaf))
    return Tree(root=root)


def _get_display_config(profile: str) -> Dict[str, List[str]]:
    """
    不同展示风格下的 lineage 压缩与标签显示配置。
    """
    if profile == "paper_like":
        return {
            "ordered": [
                "Bacillati",
                "Bacillota",
                "Bacilli",
                "Lactobacillales",
                "Streptococcaceae",
                "Streptococcus",
                "Lactobacillaceae",
                "Bacillales",
                "Staphylococcus",
                "Actinomycetota",
                "Actinomycetes",
                "Bifidobacteriales",
                "Bifidobacterium",
                "Pseudomonadati",
                "Gammaproteobacteria",
                "Pseudomonadaceae",
                "Enterobacterales",
                "Clostridia",
                "Eubacteriales",
                "Eubacteriaceae",
                "Eubacterium",
            ],
            "canonical_prefixes": [
                "Bacillati",
                "Bacillota",
                "Bacilli",
                "Lactobacillales",
                "Lactobacillaceae",
                "Streptococcus",
                "Staphylococcus",
                "Actinomycetota",
                "Actinomycetes",
                "Bifidobacterium",
                "Gammaproteobacteria",
                "Pseudomonadaceae",
                "Enterobacterales",
                "Lentivirus",
                "Fungi",
                "root",
            ],
            "singleton_allowed": ["Fungi", "Lentivirus", "root"],
        }

    # 默认：当前“去冗余版”
    return {
        "ordered": [
            "Bacillati",
            "Bacillota",
            "Bacilli",
            "Lactobacillales",
            "Streptococcaceae",
            "Lactobacillaceae",
            "Streptococcus",
            "Bacillales",
            "Staphylococcus",
            "Actinomycetota",
            "Actinomycetes",
            "Bifidobacteriales",
            "Bifidobacterium",
            "Pseudomonadati",
            "Gammaproteobacteria",
            "Enterobacterales",
            "Pseudomonadaceae",
            "Clostridia",
            "Eubacteriales",
            "Eubacteriaceae",
            "Eubacterium",
        ],
        "canonical_prefixes": [
            "Bacillati",
            "Bacillota",
            "Bacilli",
            "Lactobacillales",
            "Streptococcaceae",
            "Bacillales",
            "Actinomycetota",
            "Actinomycetes",
            "Bifidobacteriales",
            "Gammaproteobacteria",
            "Enterobacterales",
            "Lentivirus",
            "root",
        ],
        "singleton_allowed": ["Lentivirus", "root"],
    }


def _compress_lineage_for_display(lineage: List[str], profile: str = "noredundant") -> List[str]:
    """
    将完整 lineage 压缩为“图展示层级”，让不同分支按固定前缀层级对齐。
    """
    sset = {x.strip() for x in lineage if str(x).strip()}

    # 病毒 / 真菌单独处理
    if "Viruses" in sset:
        out = ["Viruses"]
        if any("Lentivirus" in x for x in sset):
            out.append("Lentivirus")
        return out
    if "Fungi" in sset:
        return ["Fungi"]

    # 细菌展示层级（按从高到低的固定顺序）
    ordered = _get_display_config(profile)["ordered"]
    out = [x for x in ordered if x in sset]
    return out


def _count_terminals(clade: Clade) -> int:
    if clade.is_terminal():
        return 1
    return sum(_count_terminals(c) for c in clade.clades)


def _layout_rectangular_tree(tree: Tree):
    """返回 x_map, y_map, max_depth，用于自定义绘制。"""
    x_map: Dict[int, float] = {}
    y_map: Dict[int, float] = {}

    def assign_x(clade: Clade, depth: int):
        x_map[id(clade)] = float(depth)
        for c in clade.clades:
            assign_x(c, depth + 1)

    leaves = tree.get_terminals()
    # 顶部到下方依次布局，扩大行距避免重叠
    leaf_spacing = 1.35
    y0 = (len(leaves) - 1) * leaf_spacing
    for i, leaf in enumerate(leaves):
        y_map[id(leaf)] = y0 - i * leaf_spacing

    def assign_y(clade: Clade):
        if clade.is_terminal():
            return y_map[id(clade)]
        ys = [assign_y(c) for c in clade.clades]
        y_map[id(clade)] = sum(ys) / len(ys)
        return y_map[id(clade)]

    assign_x(tree.root, 0)
    assign_y(tree.root)
    max_depth = max(x_map.values()) if x_map else 1.0
    return x_map, y_map, max_depth


def _norm_name(s: str) -> str:
    s = str(s).strip().replace("_", " ").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_parenthesized_aliases(name: str) -> List[str]:
    m = re.findall(r"\((.*?)\)", name)
    return [x.strip() for x in m if x.strip()]


def _load_evidence_map_from_table_s6_text(txt_path: Path) -> Dict[str, str]:
    """
    从 tmp_docx_text.txt 的 Supplementary Table S6 提取:
    key=规范化微生物名, value=Evidence 字段
    """
    if not txt_path.exists():
        return {}

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        if "Supplementary Table S6" in line:
            start = i
        if start is not None and i > start and "Supplementary Table S7" in line:
            end = i
            break
    if start is None:
        return {}
    if end is None:
        end = min(len(lines), start + 260)

    seg = [x.strip() for x in lines[start:end] if x.strip()]

    # 更稳健：只在“证据行”出现时回看上一条名称
    pairs: List[tuple[str, str]] = []
    prev = ""
    for t in seg:
        low = t.lower()
        is_evidence = low.startswith("pmid:") or ("unconfirmed" in low) or ("to be verified" in low)
        if is_evidence and prev:
            name = prev.strip()
            if name.lower() not in {"microbe", "evidence", "m", "icrobe", "e", "vidence"}:
                pairs.append((name, t))
        prev = t

    emap: Dict[str, str] = {}
    for n, e in pairs:
        key = _norm_name(n)
        if len(key) < 3:
            continue
        emap[key] = e
        for alias in _extract_parenthesized_aliases(n):
            emap[_norm_name(alias)] = e
    return emap


def _build_confirmed_unconfirmed_sets(evidence_map: Dict[str, str]) -> tuple[set[str], set[str]]:
    confirmed = set()
    unconfirmed = set()
    for k, v in evidence_map.items():
        low = str(v).lower()
        if ("unconfirmed" in low) or ("to be verified" in low):
            unconfirmed.add(k)
        elif v.strip():
            confirmed.add(k)
    return confirmed, unconfirmed


def _is_confirmed_with_prefix_fallback(
    leaf_norm: str,
    confirmed_exact: set[str],
    unconfirmed_exact: set[str],
) -> bool:
    """
    规则：
    1) 精确命中 unconfirmed -> False（最高优先级）
    2) 精确命中 confirmed -> True
    3) 兜底：同 species epithet / 同 genus 前缀视作 confirmed（用于改名或拼写变体）
    """
    if leaf_norm in unconfirmed_exact:
        return False
    if leaf_norm in confirmed_exact:
        return True

    leaf_tokens = leaf_norm.split()
    if not leaf_tokens:
        return False
    leaf_genus = leaf_tokens[0]
    leaf_species = leaf_tokens[1] if len(leaf_tokens) >= 2 else ""

    # 同种名兜底（如 Limosilactobacillus reuteri vs Lactobacillus reuteri）
    if leaf_species:
        for c in confirmed_exact:
            ct = c.split()
            if len(ct) >= 2 and ct[1] == leaf_species:
                return True

    # 同属兜底（用户提到“同前缀”）
    for c in confirmed_exact:
        ct = c.split()
        if ct and ct[0] == leaf_genus:
            # 如果该叶子被明确标过 unconfirmed（不同写法）就不兜底
            for u in unconfirmed_exact:
                ut = u.split()
                if ut and ut[0] == leaf_genus and (leaf_species and len(ut) >= 2 and ut[1] == leaf_species):
                    return False
            return True
    return False


def _draw_paper_style_tree(
    tree: Tree,
    terminal_label_map: Dict[str, str],
    highlight: Optional[List[str]],
    png_out: str,
    confirmed_set: Optional[set[str]] = None,
    display_profile: str = "noredundant",
):
    x_map, y_map, max_depth = _layout_rectangular_tree(tree)
    x_leaf = max_depth + 1.2  # 叶子统一右对齐

    fig, ax = plt.subplots(figsize=(15, 9))
    line_color = "#5f6a6a"
    node_edge = "#8fa8b3"
    node_fill = "white"

    # 递归画分支
    def draw_clade(clade: Clade):
        x_parent = x_map[id(clade)]
        if not clade.is_terminal():
            child_ys = [y_map[id(c)] for c in clade.clades]
            ax.plot([x_parent, x_parent], [min(child_ys), max(child_ys)], color=line_color, lw=1.2)
            for c in clade.clades:
                y = y_map[id(c)]
                x_child = x_leaf if c.is_terminal() else x_map[id(c)]
                ax.plot([x_parent, x_child], [y, y], color=line_color, lw=1.2)
                draw_clade(c)

        # 节点圆圈（参考图风格）：仅分叉节点和叶子节点，避免“连续圆圈”
        draw_node = clade.is_terminal() or (len(clade.clades) > 1)
        if draw_node:
            x_node = x_leaf if clade.is_terminal() else x_parent
            y_node = y_map[id(clade)]
            ax.plot(
                [x_node],
                [y_node],
                marker="o",
                markersize=5.5,
                markerfacecolor=node_fill,
                markeredgecolor=node_edge,
                markeredgewidth=1.0,
                zorder=3,
            )

    draw_clade(tree.root)

    cfg = _get_display_config(display_profile)
    canonical_prefixes = cfg["canonical_prefixes"]
    singleton_allowed = {x.lower() for x in cfg["singleton_allowed"]}

    def canonicalize_branch_name(name: str) -> str:
        s = (name or "").strip()
        ls = s.lower()
        for p in canonical_prefixes:
            if ls.startswith(p.lower()):
                return p
        return ""

    # 每个前缀只显示一次
    shown_prefix = set()
    placed_y: List[float] = []
    for clade in tree.find_clades(order="preorder"):
        if clade.is_terminal():
            continue
        name = (clade.name or "").strip()
        if not name:
            continue
        cname = canonicalize_branch_name(name)
        if not cname:
            continue
        lname = cname.lower()
        force_show = lname in {x.lower() for x in canonical_prefixes}
        # 单独只覆盖 1 个叶子的分支不显示分支名
        n_term = _count_terminals(clade)
        if (n_term < 2) and (lname not in singleton_allowed):
            continue
        if force_show and (cname in shown_prefix):
            continue
        y = y_map[id(clade)]
        # 所有标签都做避让；强制标签尝试轻微纵向偏移
        min_gap = 0.42
        y_text = y
        if any(abs(y_text - yy) < min_gap for yy in placed_y):
            if force_show:
                # 尝试上下挪动，最多 8 次
                shifted = False
                for k in range(1, 9):
                    for sign in (1, -1):
                        cand = y + sign * 0.11 * k
                        if not any(abs(cand - yy) < min_gap for yy in placed_y):
                            y_text = cand
                            shifted = True
                            break
                    if shifted:
                        break
                if not shifted:
                    continue
            else:
                continue
        placed_y.append(y_text)
        if force_show:
            shown_prefix.add(cname)
        x = x_map[id(clade)]
        # 默认轻微上移；若已避让到新位置则不额外上移
        y_shift = 0.06 if abs(y_text - y) < 1e-9 else 0.0
        fw = "bold" if force_show else "normal"
        ax.text(
            x + 0.06,
            y_text + y_shift,
            cname,
            fontsize=8.8 if force_show else 8.2,
            color="#2f3d4a",
            fontweight=fw,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
        )

    # 叶子标签（右侧）
    leaves = tree.get_terminals()
    for leaf in leaves:
        y = y_map[id(leaf)]
        raw = leaf.name or ""
        disp = terminal_label_map.get(raw, raw)
        ax.text(
            x_leaf + 0.1,
            y,
            disp,
            fontsize=10,
            fontstyle="italic",
            va="center",
            ha="left",
            color="black",
        )

    # 右侧红点列：confirmed 才打点（参考第三张图）
    if confirmed_set is None:
        confirmed_set = set()
    x_mark = x_leaf + 3.2
    for leaf in leaves:
        raw = leaf.name or ""
        y = y_map[id(leaf)]
        if _norm_name(raw) in confirmed_set:
            ax.plot([x_mark], [y], marker="s", color="#d45a8a", markersize=6, clip_on=False)

    ax.set_title("Phylogenetic (taxonomy lineage) tree of Top-K microbes", fontsize=11)
    ax.set_xlim(-0.3, x_leaf + 3.8)
    ys = [y_map[id(c)] for c in tree.find_clades()]
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(png_out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(
    input_csv: str,
    out_prefix: str,
    email: str,
    api_key: Optional[str] = None,
    top_k: Optional[int] = None,
    highlight: Optional[List[str]] = None,
    style: str = "paper",
    evidence_text_file: Optional[str] = None,
    display_profile: str = "noredundant",
) -> Dict[str, str]:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    df = pd.read_csv(input_csv)
    if "microbe" not in df.columns:
        raise ValueError(f"{input_csv} 缺少 `microbe` 列")
    if top_k is not None and top_k > 0:
        df = df.head(top_k).copy()

    out_prefix = str(Path(out_prefix))
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    valid_paths: List[List[str]] = []
    valid_leaves: List[str] = []

    for raw in df["microbe"].astype(str).tolist():
        name = _clean_name(raw)
        taxid = _search_taxid(name)
        lineage = _fetch_lineage(taxid) if taxid else None
        ok = lineage is not None and len(lineage) > 0
        rows.append(
            {
                "input_microbe": raw,
                "query_name": name,
                "taxid": taxid if taxid else "",
                "resolved": bool(ok),
                "lineage": "; ".join(lineage) if lineage else "",
            }
        )
        if ok:
            valid_paths.append(lineage)
            valid_leaves.append(raw)

    map_csv = f"{out_prefix}_taxonomy_mapping.csv"
    pd.DataFrame(rows).to_csv(map_csv, index=False)

    if len(valid_paths) < 2:
        raise RuntimeError("可解析到 taxonomy 的微生物不足 2 个，无法构建树。")

    display_paths = [_compress_lineage_for_display(p, profile=display_profile) for p in valid_paths]
    tree = _build_tree_from_paths(display_paths, valid_leaves)
    newick_out = f"{out_prefix}.nwk"
    Phylo.write(tree, newick_out, "newick")

    terminal_label_map = {leaf: f"'{_clean_name(leaf)}'" for leaf in valid_leaves}
    # confirmed/unconfirmed：默认从 Supplementary Table S6 文本提取
    if evidence_text_file:
        txt_path = Path(evidence_text_file)
    else:
        txt_path = Path(__file__).resolve().parents[1] / "tmp_docx_text.txt"
    evidence_map = _load_evidence_map_from_table_s6_text(txt_path)
    confirmed_exact, unconfirmed_exact = _build_confirmed_unconfirmed_sets(evidence_map)
    confirmed_set = set()
    for leaf in valid_leaves:
        key = _norm_name(leaf)
        if _is_confirmed_with_prefix_fallback(key, confirmed_exact, unconfirmed_exact):
            confirmed_set.add(key)

    png_out = f"{out_prefix}.png"
    if style == "paper":
        _draw_paper_style_tree(
            tree,
            terminal_label_map,
            highlight,
            png_out,
            confirmed_set=confirmed_set,
            display_profile=display_profile,
        )
    else:
        # 兼容保留：简单默认绘制
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1)
        Phylo.draw(tree, do_show=False, axes=ax, show_confidence=False)
        ax.set_title("Phylogenetic tree")
        fig.tight_layout()
        fig.savefig(png_out, dpi=200)
        plt.close(fig)

    summary = {
        "input_csv": str(Path(input_csv).resolve()),
        "out_prefix": str(Path(out_prefix).resolve()),
        "newick": str(Path(newick_out).resolve()),
        "png": str(Path(png_out).resolve()),
        "taxonomy_mapping_csv": str(Path(map_csv).resolve()),
        "input_count": int(len(df)),
        "resolved_count": int(sum(1 for r in rows if r["resolved"])),
        "unresolved_count": int(sum(1 for r in rows if not r["resolved"])),
        "evidence_text_file": str(txt_path.resolve()) if txt_path.exists() else "",
        "confirmed_dot_count": int(sum(1 for leaf in valid_leaves if _norm_name(leaf) in confirmed_set)),
    }
    summary_json = f"{out_prefix}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(Path(summary_json).resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description="由 Top-K 微生物结果生成系统发育树")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--out_prefix", type=str, required=True)
    parser.add_argument("--email", type=str, default="metamda.local@example.com")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--style", type=str, default="paper", choices=["paper", "basic"], help="默认 paper（参考图风格）")
    parser.add_argument(
        "--display_profile",
        type=str,
        default="noredundant",
        choices=["noredundant", "paper_like"],
        help="分支层级与标签策略：noredundant(当前精简版)/paper_like(更接近原论文风格)",
    )
    parser.add_argument("--evidence_text_file", type=str, default=None, help="可选：包含 Supplementary Table S6 的 txt，用于 confirmed 判定")
    parser.add_argument(
        "--highlight",
        type=str,
        default="",
        help="逗号分隔的叶子名（与 CSV microbe 列一致），用于红框高亮",
    )
    args = parser.parse_args()

    highlight_list = [x for x in args.highlight.split(",")] if args.highlight else []
    summary = run(
        input_csv=args.input_csv,
        out_prefix=args.out_prefix,
        email=args.email,
        api_key=args.api_key,
        top_k=args.top_k,
        highlight=highlight_list,
        style=args.style,
        evidence_text_file=args.evidence_text_file,
        display_profile=args.display_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


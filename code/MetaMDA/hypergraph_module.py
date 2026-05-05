"""

基于论文: Hypergraph Convolution and Hypergraph Attention (Pattern Recognition 2021)

"""

import numpy as np
from collections import defaultdict


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def build_hyperedges(G, G_sim=None, node2type=None, k: int = 20,
                     use_gene_hyperedges: bool = True,
                     use_path_hyperedges: bool = True,
                     use_similarity_hyperedges: bool = True,
                     random_hyperedges: bool = False,
                     seed: int = 42,
                     path_length: int = 2,
                     base_embeddings=None,
                     knn_from_embeddings: bool = False):
    """
    构建超边：K近邻 + 代谢物中心 + 路径 + 相似度。
    base_embeddings: dict[node]->vec，用于 embedding-KNN 时构建语义近邻超边。
    knn_from_embeddings: True 时 KNN 按 embedding 余弦相似度取 top-k，与 graph 分支更互补。
    """
    nodes = list(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    hyperedges = []

    X_norm = None
    if knn_from_embeddings and base_embeddings is not None and len(base_embeddings) > 0:
        d = len(next(iter(base_embeddings.values())))
        X = np.zeros((len(nodes), d), dtype=np.float32)
        for i, n in enumerate(nodes):
            if n in base_embeddings:
                X[i] = np.asarray(base_embeddings[n], dtype=np.float32)
        X_norm = _l2_normalize(X)

    def _topk_neighbors(node, k_val=None):
        if k_val is None:
            k_val = k
        if X_norm is not None and node in node2idx:
            i = node2idx[node]
            sims = X_norm @ X_norm[i]
            sims[i] = -1e9
            k_use = min(k_val, len(sims) - 1)
            if k_use <= 0:
                return []
            top_idx = np.argpartition(-sims, kth=k_use)[: k_use + 1]
            top_idx = top_idx[sims[top_idx] > -1e8]
            top_idx = top_idx[np.argsort(-sims[top_idx])[:k_val]]
            return [nodes[j] for j in top_idx]
        nbr_w = {}
        if G.has_node(node):
            for nbr in G.neighbors(node):
                w = 0.0
                try:
                    for ed in G[node][nbr].values():
                        w = max(w, float(ed.get("weight", 1.0)))
                except Exception:
                    w = 1.0
                nbr_w[nbr] = max(nbr_w.get(nbr, 0.0), w)
        if G_sim is not None and G_sim.number_of_nodes() > 0 and G_sim.has_node(node):
            for nbr in G_sim.neighbors(node):
                w = 0.0
                try:
                    for ed in G_sim[node][nbr].values():
                        w = max(w, float(ed.get("weight", 1.0)))
                except Exception:
                    w = 1.0
                nbr_w[nbr] = max(nbr_w.get(nbr, 0.0), w)
        if not nbr_w:
            return []
        nbrs_sorted = sorted(nbr_w.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in nbrs_sorted[:k_val]]

    # K近邻超边（每个节点一个，自适应k值）
    for v in nodes:
        # 根据节点度自适应调整k值
        degree = G.degree(v) if G.has_node(v) else 0
        sim_degree = G_sim.degree(v) if (G_sim is not None and G_sim.has_node(v)) else 0
        total_degree = degree + sim_degree
        adaptive_k = min(k, max(5, int(total_degree * 0.3)))  # 至少5个，最多k个
        
        members = _topk_neighbors(v, k_val=adaptive_k)
        members = [v] + members  # 包含自身
        # 去重 + 过滤
        seen = set()
        idxs = []
        for m in members:
            if m in node2idx and m not in seen:
                seen.add(m)
                idxs.append(node2idx[m])
        if not idxs:
            idxs = [node2idx[v]]
        hyperedges.append(idxs)

    knn_count = len(hyperedges)

    # 代谢物中心超边
    gene_count = 0
    if use_gene_hyperedges and node2type is not None:
        gene_neighbors = defaultdict(set)
        
        for edge in G.edges():
            n1, n2 = edge[0], edge[1]
            t1 = node2type.get(n1, 'unknown')
            t2 = node2type.get(n2, 'unknown')
            
            if t1 == 'gene':
                gene_neighbors[n1].add(n2)
                gene_neighbors[n1].add(n1)
            if t2 == 'gene':
                gene_neighbors[n2].add(n1)
                gene_neighbors[n2].add(n2)
        
        for gene, members in gene_neighbors.items():
            if len(members) >= 2:
                idxs = [node2idx[m] for m in members if m in node2idx]
                if len(idxs) >= 2:
                    hyperedges.append(idxs)
                    gene_count += 1

    # 路径超边：捕获多跳关系
    path_count = 0
    if use_path_hyperedges and G.number_of_nodes() > 0:
        import networkx as nx
        for v in nodes[:min(500, len(nodes))]:  # 限制数量以提高效率
            if not G.has_node(v):
                continue
            try:
                # 获取2跳邻居
                paths = list(nx.single_source_shortest_path(G, v, cutoff=path_length).values())
                if len(paths) > 1:
                    # 对于每个路径，创建包含路径上所有节点的超边
                    for path in paths[1:]:  # 跳过长度为1的路径（自身）
                        if len(path) >= 2:
                            idxs = [node2idx[n] for n in path if n in node2idx]
                            if len(idxs) >= 2 and idxs not in hyperedges:
                                hyperedges.append(idxs)
                                path_count += 1
                                if path_count >= 1000:  # 限制路径超边数量
                                    break
            except Exception:
                continue
            if path_count >= 1000:
                break

    # 相似度超边：基于相似度阈值
    sim_count = 0
    if use_similarity_hyperedges and G_sim is not None and G_sim.number_of_nodes() > 0:
        similarity_threshold = 0.7  # 相似度阈值
        for v in nodes[:min(300, len(nodes))]:  # 限制数量
            if not G_sim.has_node(v):
                continue
            similar_nodes = []
            for nbr in G_sim.neighbors(v):
                try:
                    for ed in G_sim[v][nbr].values():
                        w = float(ed.get("weight", 0.0))
                        if w >= similarity_threshold:
                            similar_nodes.append(nbr)
                            break
                except Exception:
                    continue
            
            if len(similar_nodes) >= 2:
                idxs = [node2idx[v]] + [node2idx[n] for n in similar_nodes if n in node2idx]
                idxs = list(dict.fromkeys(idxs))  # 去重保持顺序
                if len(idxs) >= 2 and idxs not in hyperedges:
                    hyperedges.append(idxs)
                    sim_count += 1

    # gene 超边下采样，避免传播过度偏向 gene 结构、对 M-D 任务无益
    gene_keep_ratio = 0.3
    if gene_count > 0:
        gene_start = knn_count
        gene_end = knn_count + gene_count
        gene_part = hyperedges[gene_start:gene_end]
        keep_n = max(1, int(len(gene_part) * gene_keep_ratio))
        rng = np.random.RandomState(0)
        keep_idx = rng.choice(len(gene_part), size=keep_n, replace=False)
        gene_part_kept = [gene_part[i] for i in keep_idx]
        hyperedges = hyperedges[:gene_start] + gene_part_kept + hyperedges[gene_end:]
        gene_count = len(gene_part_kept)

    if random_hyperedges:
        rng = np.random.RandomState(int(seed))
        N = len(nodes)
        sizes = [len(e) for e in hyperedges]
        hyperedges_rand = []
        for s in sizes:
            s = int(s)
            if s <= 0:
                hyperedges_rand.append([])
                continue
            replace = bool(s > N)
            idxs = rng.choice(N, size=s, replace=replace)
            hyperedges_rand.append([int(i) for i in idxs])
        hyperedges = hyperedges_rand
        print(
            f"[超图] K近邻超边: {knn_count}, 代谢物超边: {gene_count}, "
            f"路径超边: {path_count}, 相似度超边: {sim_count}, 总计: {len(hyperedges)}"
            f" -> 已随机打乱超边成员(random_hyperedges=True, seed={int(seed)})"
        )
    else:
        print(f"[超图] K近邻超边: {knn_count}, 代谢物超边: {gene_count}, "
              f"路径超边: {path_count}, 相似度超边: {sim_count}, 总计: {len(hyperedges)}")

    return nodes, hyperedges, node2idx


def cv_eval_embeddings_xgb(pairf: str, emb: dict, xgb_params: dict = None, seed: int = 42, n_splits: int = 10):
    """
    用 (drug, microbe) 嵌入逐元素积作为特征，XGBoost + KFold 交叉验证评估嵌入。
    Returns:
        (auroc, aupr)
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier

    if xgb_params is None:
        xgb_params = dict(
            gamma=0, learning_rate=0.049, max_depth=12,
            n_estimators=564, min_child_weight=2,
            subsample=0.81, colsample_bytree=0.54,
            tree_method="auto", scale_pos_weight=1, max_delta_step=1,
        )

    xs, ys = [], []
    with open(pairf, "r") as fin:
        lines = fin.readlines()
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        drug, microbe, label = parts[0], parts[1], parts[2]
        if drug in emb and microbe in emb:
            xs.append(emb[drug] * emb[microbe])
            ys.append(int(label))

    X = np.asarray(xs)
    y = np.asarray(ys)
    if len(y) == 0:
        raise ValueError("No usable pairs found for evaluation (pairf/embeddings mismatch).")

    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=seed)
    pred = np.zeros((len(y),), dtype=np.float32)

    for tr, te in kf.split(X):
        clf = XGBClassifier(
            base_score=0.5, booster="gbtree",
            eval_metric="error", objective="binary:logistic",
            seed=seed, **xgb_params,
        )
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]

    return roc_auc_score(y, pred), average_precision_score(y, pred)


def blend_embeddings(base_emb: dict, refined_emb: dict, alpha: float):
    """
    Elementwise blend between two embedding dicts:
        emb = (1-alpha)*base + alpha*refined

    Only blends nodes that exist in `base_emb`. Missing nodes in `refined_emb` fall back to base.
    """
    a = float(alpha)
    out = {}
    for n, v in base_emb.items():
        if n in refined_emb:
            out[n] = (1 - a) * v + a * refined_emb[n]
        else:
            out[n] = v
    return out


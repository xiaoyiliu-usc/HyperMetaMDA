"""
Trainable HGAT refiner (PyTorch) for MetaMDA.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from MetaMDA.hypergraph_module import build_hyperedges


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load_pairs(
    pairf: str,
    node2idx: Dict[str, int],
    node2type: Optional[Dict[str, str]] = None,
    allowed_pair_types: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (u_idx, v_idx, y) from pairf.
    If allowed_pair_types is provided, keep only pairs whose (type(u),type(v)) matches,
    allowing both directions when you include both in the set.
    """
    u, v, y = [], [], []
    with open(pairf, "r") as fin:
        lines = fin.readlines()

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        n1, n2, lbl = parts[0], parts[1], parts[2]
        if n1 not in node2idx or n2 not in node2idx:
            continue

        if node2type is not None and allowed_pair_types is not None:
            t1 = node2type.get(n1, "unknown")
            t2 = node2type.get(n2, "unknown")
            if (t1, t2) not in allowed_pair_types:
                continue

        u.append(node2idx[n1])
        v.append(node2idx[n2])
        y.append(int(lbl))

    return np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64), np.asarray(y, dtype=np.int64)


def _build_edge_index(G, G_sim, node2idx: Dict[str, int], include_sim_edges: bool = False) -> np.ndarray:
    """从 G（及可选 G_sim）构建普通图边列表。默认不并入 G_sim 以降低与超边视图同质。"""
    edges = set()
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            i, j = node2idx[u], node2idx[v]
            edges.add((i, j))
            edges.add((j, i))
    if include_sim_edges and G_sim is not None and G_sim.number_of_nodes() > 0:
        for u, v in G_sim.edges():
            if u in node2idx and v in node2idx:
                i, j = node2idx[u], node2idx[v]
                edges.add((i, j))
                edges.add((j, i))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(list(edges), dtype=np.int64).T


def trainable_hgat_refine(
    base_embeddings: Dict[str, np.ndarray],
    G,
    G_sim,
    node2type: Dict[str, str],
    pairf: str,
    *,
    k: int = 20,
    use_gene_hyperedges: bool = True,
    use_path_hyperedges: bool = True,
    use_similarity_hyperedges: bool = True,
    random_hyperedges: bool = False,
    path_length: int = 2,
    knn_from_embeddings: bool = True,
    num_layers: int = 2,
    hidden_dim: int = 128,
    tau: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    patience: int = 100,
    dropout: float = 0.2,
    reg_lambda: float = 1e-4,
    align_lambda: float = 1e-3,
    gate_lambda: float = 1e-3,
    out_residual: bool = True,
    use_residual_delta: bool = True,
    use_node_gate: bool = True,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
    use_multihead_fusion: bool = False,
    num_heads: int = 4,
    use_pair_level_attention: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Train a HGAT refiner to produce refined node embeddings.

    Encoder: 超边 + 普通边 -> z_hyper, z_graph，上投影得 Z_hyper, Z_graph；保存用 Z_fused=0.5*Z_hyper+0.5*Z_graph。

    Decoder 二选一（use_pair_level_attention 更易带来明显提升）：
      - True：对 (Z_hyper[d], Z_graph[d], Z_hyper[m], Z_graph[m]) 做配对级多头注意力再 MLP 预测
      - False：score = MLP( Z_fused[u] ⊙ Z_fused[v] )

    Returns:
        refined_embeddings: dict[node] -> np.ndarray (same dim as base_embeddings)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for trainable HGAT refinement. "
            "Install torch and retry."
        ) from e

    # reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # build hypergraph（embedding-KNN 超边与 graph 分支更互补）
    nodes, hyperedges, node2idx = build_hyperedges(
        G, G_sim, node2type, k,
        use_gene_hyperedges=use_gene_hyperedges,
        use_path_hyperedges=use_path_hyperedges,
        use_similarity_hyperedges=use_similarity_hyperedges,
        random_hyperedges=random_hyperedges,
        seed=int(seed),
        path_length=path_length,
        base_embeddings=base_embeddings,
        knn_from_embeddings=knn_from_embeddings,
    )
    if len(nodes) == 0:
        return base_embeddings

    d_in = int(len(next(iter(base_embeddings.values()))))
    X0 = np.zeros((len(nodes), d_in), dtype=np.float32)
    for i, n in enumerate(nodes):
        if n in base_embeddings:
            X0[i] = base_embeddings[n].astype(np.float32, copy=False)

    # incidence list (v_idx, e_idx)
    v_idx: List[int] = []
    e_idx: List[int] = []
    for e, members in enumerate(hyperedges):
        for v in members:
            v_idx.append(int(v))
            e_idx.append(int(e))
    v_idx = np.asarray(v_idx, dtype=np.int64)
    e_idx = np.asarray(e_idx, dtype=np.int64)
    E = len(hyperedges)
    N = len(nodes)

    # degrees (for proto mean)
    De = np.zeros((E,), dtype=np.float32)
    np.add.at(De, e_idx, 1.0)
    De = np.maximum(De, 1.0)

    # load supervised pairs (strict M-D only so HGAT aligns with final task)
    # nodetype 里微生物多为 "disease"，兼容 "microbe"
    allowed = {("drug", "disease"), ("disease", "drug"), ("microbe", "drug"), ("drug", "microbe")}
    u_np, v_np, y_np = _load_pairs(pairf, node2idx, node2type=node2type, allowed_pair_types=allowed)
    if len(y_np) == 0:
        if verbose:
            print("[Trainable HGAT] No usable pairs found (pairf/embeddings mismatch). Skip.")
        return base_embeddings

    # train/val split
    rng = np.random.RandomState(seed)
    all_idx = np.arange(len(y_np))
    rng.shuffle(all_idx)
    n_val = max(1, int(0.1 * len(all_idx)))
    val_idx = all_idx[:n_val]
    tr_idx = all_idx[n_val:]

    # 普通图边列表（默认不含 G_sim，降低与超边同质）
    edge_index_np = _build_edge_index(G, G_sim, node2idx, include_sim_edges=False)
    edge_index_t = torch.from_numpy(edge_index_np).to(device=device, dtype=torch.long)

    # torch tensors
    X0_t = torch.from_numpy(X0).to(device=device, dtype=torch.float32)
    v_idx_t = torch.from_numpy(v_idx).to(device=device, dtype=torch.long)
    e_idx_t = torch.from_numpy(e_idx).to(device=device, dtype=torch.long)
    De_t = torch.from_numpy(De).to(device=device, dtype=torch.float32)

    u_tr = torch.from_numpy(u_np[tr_idx]).to(device=device, dtype=torch.long)
    v_tr = torch.from_numpy(v_np[tr_idx]).to(device=device, dtype=torch.long)
    y_tr = torch.from_numpy(y_np[tr_idx].astype(np.float32)).to(device=device)

    u_va = torch.from_numpy(u_np[val_idx]).to(device=device, dtype=torch.long)
    v_va = torch.from_numpy(v_np[val_idx]).to(device=device, dtype=torch.long)
    y_va = torch.from_numpy(y_np[val_idx].astype(np.float32)).to(device=device)

    # class imbalance handling
    pos = float((y_tr == 1).sum().item())
    neg = float((y_tr == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    eps = 1e-12

    def group_softmax(scores: torch.Tensor, group: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Softmax over variable-size groups defined by `group` indices."""
        # max per group for numerical stability
        max_init = torch.full((num_groups,), -1e15, dtype=scores.dtype, device=scores.device)
        max_per = torch.scatter_reduce(max_init, 0, group, scores, reduce="amax", include_self=True)
        exp = torch.exp(scores - max_per[group])
        denom = torch.zeros((num_groups,), dtype=scores.dtype, device=scores.device).index_add(0, group, exp)
        return exp / (denom[group] + eps)

    class HGATLayer(nn.Module):
        def __init__(self, d_h: int, tau_: float, dropout_: float):
            super().__init__()
            self.lin = nn.Linear(d_h, d_h, bias=True)
            self.att_v2e = nn.Parameter(torch.empty((2 * d_h,), dtype=torch.float32))
            self.att_e2v = nn.Parameter(torch.empty((2 * d_h,), dtype=torch.float32))
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
            nn.init.xavier_uniform_(self.att_v2e.view(1, -1))
            nn.init.xavier_uniform_(self.att_e2v.view(1, -1))
            self.tau = float(tau_)
            self.drop = nn.Dropout(p=float(dropout_))
            self.act = nn.LeakyReLU(0.2)

        def forward(self, H: torch.Tensor) -> torch.Tensor:
            # H: [N, d_h]
            H1 = self.drop(self.act(self.lin(H)))

            # hyperedge proto: mean_{v in e} H1[v]
            edge_sum = torch.zeros((E, H1.shape[1]), dtype=H1.dtype, device=H1.device)
            edge_sum = edge_sum.index_add(0, e_idx_t, H1[v_idx_t])
            proto = edge_sum / De_t.unsqueeze(1)

            # vertex -> hyperedge attention within each hyperedge (group by e)
            hv = H1[v_idx_t]
            he = proto[e_idx_t]
            s = torch.cat([hv, he], dim=1) * self.att_v2e.unsqueeze(0)
            s = self.act(s.sum(dim=1) / max(self.tau, eps))
            alpha = group_softmax(s, e_idx_t, E)  # [M]
            alpha = self.drop(alpha)

            edge_emb = torch.zeros((E, H1.shape[1]), dtype=H1.dtype, device=H1.device)
            edge_emb = edge_emb.index_add(0, e_idx_t, hv * alpha.unsqueeze(1))

            # hyperedge -> vertex attention over incident hyperedges (group by v)
            ev = edge_emb[e_idx_t]
            t = torch.cat([ev, hv], dim=1) * self.att_e2v.unsqueeze(0)
            t = self.act(t.sum(dim=1) / max(self.tau, eps))
            beta = group_softmax(t, v_idx_t, N)  # [M]
            beta = self.drop(beta)

            node_out = torch.zeros((N, H1.shape[1]), dtype=H1.dtype, device=H1.device)
            node_out = node_out.index_add(0, v_idx_t, ev * beta.unsqueeze(1))

            # residual
            return self.drop(self.act(node_out)) + H

    class GraphAttnLayer(nn.Module):
        """普通图上的 GAT 风格注意力聚合层。"""
        def __init__(self, d_h: int, dropout_: float):
            super().__init__()
            self.lin = nn.Linear(d_h, d_h, bias=False)
            self.attn = nn.Parameter(torch.empty(2 * d_h, dtype=torch.float32))
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.uniform_(self.attn, -1.0 / (d_h ** 0.5), 1.0 / (d_h ** 0.5))
            self.drop = nn.Dropout(p=float(dropout_))
            self.act = nn.LeakyReLU(0.2)

        def forward(self, H: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            N, d = H.shape
            src, dst = edge_index[0], edge_index[1]
            H1 = self.drop(self.act(self.lin(H)))
            x_src, x_dst = H1[src], H1[dst]
            e = (torch.cat([x_src, x_dst], dim=-1) * self.attn.unsqueeze(0)).sum(dim=-1)
            e = self.act(e)
            e_max = torch.full((N,), -1e15, dtype=e.dtype, device=e.device)
            e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
            exp = torch.exp(e - e_max[dst])
            denom = torch.zeros((N,), dtype=e.dtype, device=e.device)
            denom.index_add_(0, dst, exp)
            alpha = self.drop(exp / (denom[dst] + eps))
            out = torch.zeros_like(H1)
            out.index_add_(0, dst, x_src * alpha.unsqueeze(-1))
            return self.drop(self.act(out))

    class GraphEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                GraphAttnLayer(hidden_dim, dropout) for _ in range(int(num_layers))
            ])

        def forward(self, H: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                H = layer(H, edge_index)
            return H

    # Q/K/V 扩展 + 注意力后 Conv2d 精炼 + 等权残差
    class MultiHeadViewFusion(nn.Module):
        """
        对 (N, 2, hidden_dim) 两路视图做融合：
        1. Q/K/V 投影到 hidden_dim*num_heads（每头保持 hidden_dim，增强容量）
        2. 注意力后 Conv2d + ReLU 做空间精炼
        3. 输出与等权融合做残差，避免覆盖原有信号
        """
        def __init__(self, hidden_dim: int, num_heads: int, dropout_: float):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.d_k = hidden_dim  # 与 MVML-MPI 一致：每头 d_k = dim_in，不压缩
            dim_k = hidden_dim * num_heads
            dim_v = hidden_dim * num_heads
            self._norm_fact = 1.0 / math.sqrt(self.d_k)
            self.linear_q = nn.Linear(hidden_dim, dim_k, bias=False)
            self.linear_k = nn.Linear(hidden_dim, dim_k, bias=False)
            self.linear_v = nn.Linear(hidden_dim, dim_v, bias=False)
            self.drop = nn.Dropout(p=dropout_)
            # MVML-MPI: conv(att) 再 flatten + mlp。这里 att 为 (N, nh, 2, hidden_dim)，用 Conv2d 精炼
            self.conv = nn.Sequential(
                nn.Conv2d(num_heads, num_heads, kernel_size=(2, 3), padding=(0, 1)),
                nn.ReLU(),
                nn.Dropout(dropout_),
            )
            # conv 后 (N, nh, 1, hidden_dim) -> flatten -> (N, nh*hidden_dim)
            self.proj = nn.Sequential(
                nn.Linear(num_heads * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_),
            )
            # 残差尺度：softplus 保证为正，初值约 0.31 使多头融合有可见贡献
            self._fusion_scale_raw = nn.Parameter(torch.tensor(-1.0))  # softplus(-1.0)≈0.31，融合初值更可见

        def forward(self, x: torch.Tensor, z_equal: torch.Tensor) -> torch.Tensor:
            # x: (N, 2, hidden_dim), z_equal: (N, hidden_dim) 等权融合，用于残差
            N, n_views, _ = x.shape
            nh, dk = self.num_heads, self.d_k
            q = self.linear_q(x).reshape(N, n_views, nh, dk).transpose(1, 2)   # (N, nh, 2, dk)
            k = self.linear_k(x).reshape(N, n_views, nh, dk).transpose(1, 2)
            v = self.linear_v(x).reshape(N, n_views, nh, dk).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact
            attn = self.drop(F.softmax(scores, dim=-1))
            att = torch.matmul(attn, v)                                         # (N, nh, 2, dk)
            att = self.conv(att)                                                 # (N, nh, 1, dk)
            att = att.view(N, -1)                                               # (N, nh*dk)
            z_attn = self.proj(att)                                             # (N, hidden_dim)，已 LayerNorm+Dropout
            scale = F.softplus(self._fusion_scale_raw)                           # 初值≈0.31，训练中可继续增大
            return z_equal + scale * z_attn

    class PairFeatureMLPDecoder(nn.Module):
        """
        对齐 XGBoost/传统链路预测的 pair 特征：[u*v, |u-v|, u, v] -> MLP -> logit
        """
        def __init__(self, d_in: int, hidden: int = 256, dropout_: float = 0.2):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(4 * d_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout_),
                nn.Linear(hidden, 1),
            )

        def forward(self, zu: torch.Tensor, zv: torch.Tensor) -> torch.Tensor:
            feat = torch.cat([zu * zv, torch.abs(zu - zv), zu, zv], dim=1)  # (B, 4*d)
            return self.mlp(feat).squeeze(-1)

    # 配对级多头注意力：对 (Z_hyper[d], Z_graph[d], Z_hyper[m], Z_graph[m]) 做 attention，直接服务预测任务
    class PairLevelMultiHeadDecoder(nn.Module):
        """输入 (B, 4, d_in)，4 个 token 做 multi-head self-attention，pool 后 MLP 得到 logit。"""
        def __init__(self, d_in: int, num_heads: int, dropout_: float, hidden: int = 256):
            super().__init__()
            self.d_in = d_in
            nh = max(1, min(num_heads, d_in))
            while d_in % nh != 0 and nh > 1:
                nh -= 1
            self.num_heads = nh
            self.d_k = d_in // nh
            self._scale = 1.0 / math.sqrt(self.d_k)
            self.lin_q = nn.Linear(d_in, d_in, bias=False)
            self.lin_k = nn.Linear(d_in, d_in, bias=False)
            self.lin_v = nn.Linear(d_in, d_in, bias=False)
            self.drop = nn.Dropout(p=dropout_)
            self.pool = nn.Linear(4 * d_in, d_in)
            self.mlp = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout_),
                nn.Linear(hidden, 1),
            )

        def forward(self, four: torch.Tensor) -> torch.Tensor:
            # four: (B, 4, d_in)
            B, _, _ = four.shape
            nh, dk = self.num_heads, self.d_k
            q = self.lin_q(four).reshape(B, 4, nh, dk).transpose(1, 2)   # (B, nh, 4, dk)
            k = self.lin_k(four).reshape(B, 4, nh, dk).transpose(1, 2)
            v = self.lin_v(four).reshape(B, 4, nh, dk).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
            attn = self.drop(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, v)   # (B, nh, 4, dk)
            out = out.transpose(1, 2).reshape(B, 4 * self.d_in)
            out = self.pool(out)
            return self.mlp(out).squeeze(-1)

    class TrainableHGATRefiner(nn.Module):
        """超边 + 普通边双分支；"""
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(d_in, hidden_dim, bias=True)
            self.norm0 = nn.LayerNorm(hidden_dim)
            self.layers_hyper = nn.ModuleList([HGATLayer(hidden_dim, tau, dropout) for _ in range(int(num_layers))])
            self.encoder_graph = GraphEncoder()
            self._fusion_hyper = 0.5
            self._fusion_graph = 0.5
            self._use_multihead_fusion = use_multihead_fusion
            self._use_pair_level_attention = False  # 强制使用对齐 decoder，反传信号与 XGBoost 特征一致
            self.pair_mlp_dec = PairFeatureMLPDecoder(d_in, hidden=256, dropout_=dropout)
            if use_multihead_fusion:
                nh = min(num_heads, hidden_dim)
                while nh > 0 and hidden_dim % nh != 0:
                    nh -= 1
                if nh < 1:
                    nh = 1
                self._view_norm = nn.LayerNorm(hidden_dim)
                self.fusion = MultiHeadViewFusion(hidden_dim, nh, dropout)
            self.up = nn.Linear(hidden_dim, d_in, bias=True)
            self._use_residual_delta = use_residual_delta
            if use_residual_delta:
                self.beta_raw = nn.Parameter(torch.tensor(-3.0))
            self.use_node_gate = bool(use_node_gate)
            gate_hid = max(64, d_in // 2)
            self.gate_norm = nn.LayerNorm(3 * d_in)  # 防止 gate 输入尺度把 sigmoid 顶满
            self.node_gate = nn.Sequential(
                nn.Linear(3 * d_in, gate_hid),
                nn.ReLU(),
                nn.Linear(gate_hid, 1),
            )
            nn.init.constant_(self.node_gate[-1].bias, -2.0)  # sigmoid(-2)≈0.12，避免初始全 1
            self.dec = nn.Linear(d_in, 1, bias=True)

        def encode(self, X: torch.Tensor, edge_index: torch.Tensor):
            H = self.norm0(self.down(X))
            H = F.leaky_relu(H, negative_slope=0.2)
            z_hyper = H
            for layer in self.layers_hyper:
                z_hyper = layer(z_hyper)
            if edge_index.shape[1] == 0:
                z = z_hyper
                z_graph = None
            else:
                z_graph = self.encoder_graph(H, edge_index)
                z_equal = self._fusion_hyper * z_hyper + self._fusion_graph * z_graph
                if self._use_multihead_fusion:
                    z_hyper_n = self._view_norm(z_hyper)
                    z_graph_n = self._view_norm(z_graph)
                    z_stack = torch.stack([z_hyper_n, z_graph_n], dim=1)
                    z = self.fusion(z_stack, z_equal)
                else:
                    z = z_equal
            delta = self.up(z)
            if out_residual:
                if self._use_residual_delta:
                    if getattr(self, "use_node_gate", False):
                        gate_in = torch.cat([X, delta, torch.abs(delta)], dim=1)  # (N, 3d)
                        gate_in = self.gate_norm(gate_in)
                        gate_temp = 2.0  # 温度，减轻 sigmoid 饱和
                        g = torch.sigmoid(self.node_gate(gate_in) / gate_temp)    # (N, 1)
                        self._last_gate = g
                        beta = torch.sigmoid(self.beta_raw)
                        Z_fused = X + (beta * g) * delta
                    else:
                        beta = torch.sigmoid(self.beta_raw)
                        Z_fused = X + beta * delta
                else:
                    Z_fused = X + delta
            else:
                Z_fused = delta
            Z_hyper = self.up(z_hyper)
            if out_residual:
                Z_hyper = X + Z_hyper
            if z_graph is not None:
                Z_graph = self.up(z_graph)
                if out_residual:
                    Z_graph = X + Z_graph
            else:
                Z_graph = None
            return Z_fused, Z_hyper, Z_graph

        def decode(self, u_idx: torch.Tensor, v_idx: torch.Tensor, X_base: torch.Tensor,
                   Z_fused: torch.Tensor, Z_hyper: torch.Tensor, Z_graph: torch.Tensor) -> torch.Tensor:
            zu_ref = Z_fused[u_idx]
            zv_ref = Z_fused[v_idx]
            return self.pair_mlp_dec(zu_ref, zv_ref)

    model = TrainableHGATRefiner().to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val = float("inf")
    bad = 0

    if verbose:
        dec_desc = "XGBoost对齐 [u*v,|u-v|,u,v]+MLP"
        fusion_desc = f"节点融合={'multi-head attention' if use_multihead_fusion else '等权'}"
        delta_desc = "残差增量(beta小)" if use_residual_delta else "残差"
        node_gate_desc = "per-node_gate=On" if use_node_gate else "per-node_gate=Off"
        print(f"[Trainable HGAT] start: 超边+普通边 {fusion_desc}, 解码={dec_desc}, 输出={delta_desc}, {node_gate_desc}, align_lambda={align_lambda}, gate_lambda={gate_lambda}, layers={num_layers}, hidden={hidden_dim}, epochs={epochs}, lr={lr}, wd={weight_decay}, tau={tau}")

    for ep in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        Z_fused, Z_hyper, Z_graph = model.encode(X0_t, edge_index_t)
        logit = model.decode(u_tr, v_tr, X0_t, Z_fused, Z_hyper, Z_graph)
        loss = loss_fn(logit, y_tr)
        if align_lambda and float(align_lambda) > 0:
            delta = Z_fused - X0_t
            loss = loss + float(align_lambda) * torch.mean(delta ** 2)
        g = getattr(model, "_last_gate", None)
        if g is not None and gate_lambda and float(gate_lambda) > 0:
            loss = loss + float(gate_lambda) * g.mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            Z_fused, Z_hyper, Z_graph = model.encode(X0_t, edge_index_t)
            logit_va = model.decode(u_va, v_va, X0_t, Z_fused, Z_hyper, Z_graph)
            val_loss = loss_fn(logit_va, y_va).item()
            if align_lambda and float(align_lambda) > 0:
                delta = Z_fused - X0_t
                val_loss = float(val_loss) + float(align_lambda) * torch.mean(delta ** 2).item()

        if verbose and (ep == 0 or (ep + 1) % 5 == 0):
            print(f"[Trainable HGAT] ep={ep+1:03d} loss={loss.item():.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                if verbose:
                    print(f"[Trainable HGAT] early stop at ep={ep+1}, best_val_loss={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        Z_fused, Z_hyper, Z_graph = model.encode(X0_t, edge_index_t)
        _ = model.decode(u_tr, v_tr, X0_t, Z_fused, Z_hyper, Z_graph)
        Z = Z_fused.detach().cpu().numpy().astype(np.float32, copy=False)
        if verbose and getattr(model, "_last_gate", None) is not None:
            g = model._last_gate
            print(f"[Trainable HGAT] 诊断 gate mean={g.mean().item():.4f}, gate p90={g.quantile(0.9).item():.4f}")
        if verbose:
            delta_norm = (Z_fused - X0_t).pow(2).sum(dim=1).sqrt().mean().item()
            print(f"[Trainable HGAT] 诊断 avg ||Z-X||={delta_norm:.4f}")
        if Z_graph is not None:
            cos = F.cosine_similarity(Z_hyper, Z_graph, dim=1).mean().item()
            scale = F.softplus(model.fusion._fusion_scale_raw).item() if (getattr(model, 'fusion', None) is not None and hasattr(model.fusion, '_fusion_scale_raw')) else float('nan')
            if verbose:
                print(f"[Trainable HGAT] 诊断 view cosine mean={cos:.4f}, fusion_scale={scale:.4f}")
        if verbose and getattr(model, 'beta_raw', None) is not None:
            beta = torch.sigmoid(model.beta_raw).item()
            print(f"[Trainable HGAT] 诊断 residual_delta beta={beta:.4f}")
        if verbose and hyperedges:
            avg_he = sum(len(e) for e in hyperedges) / len(hyperedges)
            print(f"[Trainable HGAT] 诊断 avg hyperedge size={avg_he:.2f}, #hyperedges={len(hyperedges)}")

    refined: Dict[str, np.ndarray] = {}
    for i, n in enumerate(nodes):
        if n in base_embeddings:
            refined[n] = Z[i]

    # keep any leftover nodes (defensive)
    for n, vec in base_embeddings.items():
        if n not in refined:
            refined[n] = vec

    if verbose:
        print("[Trainable HGAT] done. refined embeddings ready.")

    return refined



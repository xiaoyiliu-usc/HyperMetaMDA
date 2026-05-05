import argparse
from typing import Optional, List

import os
import math
import random
import pickle
import networkx as nx
import numpy as np
from scipy import stats
from tqdm import tqdm
import parmap
from collections import defaultdict, Counter

from MetaMDA.HeterogeneousSG import HeterogeneousSG
from MetaMDA.utils import read_graph, set_seed
from MetaMDA.hypergraph_module import (
    cv_eval_embeddings_xgb,
    blend_embeddings,
)

# Optional: trainable HGAT (PyTorch)
try:
    from MetaMDA.hypergraph_trainable_hgat import trainable_hgat_refine, torch_available
except Exception:
    trainable_hgat_refine = None

    def torch_available():  # type: ignore
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_file', type=str, required=True)
    parser.add_argument('--sim_network_file', type=str, default=None)
    parser.add_argument('--node_type_file', type=str, default=None)
    parser.add_argument('--node_file', type=str, default=None)
    parser.add_argument('--sim_node_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='embedding_file.pkl')
    parser.add_argument('--seed', type=float, default=42)
    parser.add_argument('--tp_factor', type=float, default=0.5)
    parser.add_argument('--weighted', type=bool, default=True)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--num_walks', type=int, default=100)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='if default, set to all available cpu count')
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)
    parser.add_argument('--em_max_iter', type=int, default=5,
                        help='maximum EM iteration for edge type transition matrix training')
    parser.add_argument('--net_delimiter', type=str, default='\t',
                        help='delimiter of networks file; default = tab')

    args = parser.parse_args()
    args = {
        'netf': args.network_file,
        'sim_netf': args.sim_network_file,
        'outputf': args.output_file,
        'nodetypef': args.node_type_file,
        'nodef': args.node_file,
        'simnodef': args.sim_node_file,
        'tp_factor': args.tp_factor,
        'seed': args.seed,
        'weighted': args.weighted,
        'directed': args.directed,
        'num_walks': args.num_walks,
        'walk_length': args.walk_length,
        'dimension': args.dimension,
        'window_size': args.window_size,
        'workers': args.workers,
        'em_max_iter': args.em_max_iter,
        'p': args.p,
        'q': args.q,
        'net_delimiter': args.net_delimiter
    }
    return args


# Train edge type transition matrix
def train_edgetype_transition_matrix(em_max_iter, G, networkf, net_delimiter, walk_length, p, q):
    matrix_conv_rate = 0.01
    matrices = {0: _init_edge_transition_matrix(networkf, net_delimiter)}

    for i in range(em_max_iter):  # EM iteration
        walks = _sample_edge_paths(G, matrices[i], walk_length, p, q)  # M step
        matrices[i + 1] = _update_trans_matrix(walks, matrices[i])  # E step
        matrix_diff = np.nan_to_num(np.absolute((matrices[i + 1] - matrices[i]) / matrices[i])).mean()
        if matrix_diff < matrix_conv_rate:
            break
    return matrices[i + 1]

#得到edge walks序列
def _sample_edge_paths(G, trans_matrix, walk_length, p, q):
    edges = list(G.edges(data=True))
    sampled_edges = random.sample(edges, int(len(edges) * 0.01))  # sample 1% of edges from the original network
    edge_walks = []
    for edge in sampled_edges:
        edge_walks.append(_edge_transition_walk(edge, G, trans_matrix, walk_length, p, q))
    return edge_walks


def _edge_transition_walk(edge, G, matrix, walk_length, p, q):
    edge = (edge[0], edge[1], edge[2]['type'])  # 边的两个节点及附加数据
    walk = [edge]  # 存储随机游走路径的列表
    edge_path = [edge[2]]  # 存储边类型的列表，用于记录路径中每条边的类型

    while len(walk) < walk_length:
        cur_edge = walk[-1]
        prev_node = cur_edge[0]
        cur_node = cur_edge[1]
        cur_edge_type = cur_edge[2]

        nbrs = G.neighbors(cur_node)
        # calculate edge weights for all neighbors
        nbrs_list = []
        weights_list = []
        for nbr in nbrs:
            for nbr_edge in G[cur_node][nbr].values():  # 遍历当前节点与邻居节点之间的所有边
                nbr_edge_type = nbr_edge['type']  # 提取边的类型和权重:
                nbr_edge_weight = nbr_edge['weight']
                trans_weight = matrix[cur_edge_type - 1][nbr_edge_type - 1]  # 根据当前边类型和邻居边类型从转移权重矩阵中获取相应的权重。
                if G.has_edge(nbr, prev_node) or G.has_edge(prev_node, nbr):
                    nbrs_list.append((nbr, nbr_edge_type))
                    weights_list.append(trans_weight * nbr_edge_weight)
                elif nbr == prev_node:  # p: Return parameter
                    nbrs_list.append((nbr, nbr_edge_type))
                    weights_list.append(trans_weight * nbr_edge_weight / p)
                else:  # q: In-out parameter q
                    nbrs_list.append((nbr, nbr_edge_type))
                    weights_list.append(trans_weight * nbr_edge_weight / q)
        if sum(weights_list) > 0:  # the direction_node has next_link_end_node
            next_edge = random.choices(nbrs_list, weights=weights_list)[0]
            next_edge = (cur_node, next_edge[0], next_edge[1])
            walk.append(next_edge)
            edge_path.append(next_edge[2])
        else:
            break
    return edge_path

#
def _init_edge_transition_matrix(networkf, net_delimiter):
    edgetypes = set()
    with open(networkf, 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        edgetypes.add(line.split(net_delimiter)[2])
    type_count = len(edgetypes)  # Number of edge types

    matrix = np.ones((type_count, type_count))
    matrix = matrix / matrix.sum()
    return matrix


def _update_trans_matrix(walks, matrix):
    type_count = len(matrix)
    matrix = np.zeros(matrix.shape)
    repo = defaultdict(list)

    for walk in walks:
        walk = [i - 1 for i in walk]  # 对路径中的每个节点减去1
        edge_count = Counter(walk)
        #         if edge_id in curr
        for i in range(type_count):
            repo[i].append(edge_count[i])
    for i in range(type_count):
        for j in range(type_count):
            sim_score = pearsonr_test(repo[i], repo[j])
            matrix[i][j] = sim_score
    return np.nan_to_num(matrix)


def pearsonr_test(v1, v2):  # original metric: the larger the more similar
    result = stats.mstats.pearsonr(v1, v2)[0]
    return sigmoid(result)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Teleport guided random walk
def generate_DREAMwalk_paths(G, G_sim, trans_matrix, p, q, num_walks, walk_length, tp_factor, workers, nodetype, t, base_seed: int = 42):
    tot_walks = []
    nodes = list(G.nodes())

    #     print(f'# of walkers : {min(num_walks,workers)} for {num_walks} times')
    #     使用 parmap.map 并行地对  应用   函数
    walks = parmap.map(_parmap_walks, range(num_walks),
                       nodes, G, G_sim, trans_matrix, p, q, walk_length, tp_factor,
                       nodetype, t, base_seed, pm_pbar=False, pm_processes=min(num_walks, workers))
    # 通过设置 pm_processes 参数，可以控制并行执行任务的进程数量
    # print(np.array(walks).shape) (100, 699, 10)

    # 列表的 + 操作符用于连接两个列表，将它们合并成一个新的列表
    for walk in walks:
        tot_walks += walk

    return tot_walks  # (69900, 10)


# parallel walks
def _parmap_walks(walk_i, nodes, G, G_sim, trans_matrix, p, q, walk_length, tp_factor, nodetype, t, base_seed):  # (699, 10)
    walks = []
    # Make per-process randomness deterministic
    s = int(base_seed) + int(walk_i)
    random.seed(s)
    np.random.seed(s)

    nodes_local = list(nodes)
    random.shuffle(nodes_local)
    for node in nodes_local:
        walks.append(_DREAMwalker(node, G, G_sim, trans_matrix, p, q, walk_length, tp_factor, nodetype, t))
    return walks


def _DREAMwalker(start_node, G, G_sim, trans_matrix, p, q, walk_length, tp_factor, nodetype, t):  ##(1, 10)
    k = 100
    walk = [start_node]
    edge_walk = []  # 添加的是边的类型

    # select first edge from any neighbors
    type = nodetype[start_node]  # 判断当前节点的类型
    if type == 'gene':
        next_edge = _first_choice(G, start_node)
        walk.append(next_edge[0])  # 邻居节点
        edge_walk.append(next_edge[1])  # 和邻居节点连成的边的类型
    else:
        cur = start_node  # 判断当前节点在图G中是否有邻居节点
        while len(walk) < walk_length:
            if len(sorted(G.neighbors(cur))) == 0 and len(sorted(G_sim.neighbors(cur))) == 0:
                return walk
            elif len(sorted(G.neighbors(cur))) > 0:
                next_edge = _first_choice(G, cur)
                walk.append(next_edge[0])
                edge_walk.append(next_edge[1])
                break
            elif len(sorted(G_sim.neighbors(cur))) > 0:
                next_node = _teleport_operation(cur, G_sim)
                walk.append(next_node)
                cur = next_node
    # #生成的初始walk
    # print(walk)
    # print(edge_walk)

    #这一步中肯定没有孤立点了
    if len(walk) == walk_length:
        return walk

    while len(walk) < walk_length:
        prev = walk[-2]
        cur = walk[-1]
        cur_edge_type = edge_walk[-1]

        type = nodetype[cur] #判断当前节点的类型
        if type == 'gene': #如果当前节点是代谢物，按照状态转移概率矩阵随机游走
            prev = (prev, cur_edge_type)
            next_node, next_edge_type = _network_traverse(cur, prev, G, trans_matrix, p, q)
            edge_walk.append(next_edge_type)
            walk.append(next_node)
        else: #如果是微生物或药物
            cur_nbrs = sorted(G.neighbors(cur)) #判断当前节点在图G中是否有邻居节点
            if len(cur_nbrs) == 0:#没有邻居节点，在G_sim中走
                 next_node = _teleport_operation(cur, G_sim)
                 walk.append(next_node)
            else: #有邻居节点
                max_weight = 0 #求该节点与其他节点最大的相似度值
                for nbr_sim in sorted(G_sim.neighbors(cur)):
                    for cur_sim_edge in G_sim[cur][nbr_sim].values():
                        if cur_sim_edge['weight'] > max_weight:
                            max_weight = cur_sim_edge['weight']
                pro = f(max_weight, k, t) #在G_sim中走的概率
                rand = np.random.rand()
                if rand <= pro:
                    next_node = _teleport_operation(cur, G_sim)
                    walk.append(next_node)
                else:
                    next_node, next_edge_type = _network_traverse_simple(cur, prev, G, p, q)
                    edge_walk.append(next_edge_type)
                    walk.append(next_node)

    return walk

def f(x, k, t):
    return 1 / (1+np.exp(-k*(x-t)))

def _first_choice(G, cur):
    nbrs_list = []
    weights_list = []
    for nbr in sorted(G.neighbors(cur)):
        for cur_edge in G[cur][nbr].values():
            nbrs_list.append((nbr, cur_edge['type']))
            weights_list.append(cur_edge['weight'])
    # print(nbrs_list)  # [('GBA', 3), ('MGAM', 3), ('SMC1A', 3), ('miglitol', 1)]
    # print(weights_list)  # [1.0, 1.0, 1.0, 1.0]
    next_edge = random.choices(nbrs_list,
                               weights=weights_list)[0]

    return next_edge

def _network_traverse(cur, prev, G, trans_matrix, p, q):
    prev_node = prev[0]
    cur_edge_type = prev[1]
    cur_nbrs = sorted(G.neighbors(cur))
    nbrs_list = []
    weights_list = []

    # search for reachable edges and their weights
    for nbr in cur_nbrs:
        nbr_edges = G[cur][nbr]
        for nbr_edge in nbr_edges.values():
            nbr_edge_type = nbr_edge['type']
            nbr_edge_weight = nbr_edge['weight']
            trans_weight = trans_matrix[cur_edge_type - 1][nbr_edge_type - 1]

            if G.has_edge(nbr, prev_node) or G.has_edge(prev_node, nbr):
                nbrs_list.append((nbr, nbr_edge_type))
                weights_list.append(trans_weight * nbr_edge_weight)
            elif nbr == prev_node:  # p: Return parameter
                nbrs_list.append((nbr, nbr_edge_type))
                weights_list.append(trans_weight * nbr_edge_weight / p)
            else:  # q: In-Out parameter
                nbrs_list.append((nbr, nbr_edge_type))
                weights_list.append(trans_weight * nbr_edge_weight / q)

    # sample next node and edge type from searched weights
    next_edge = random.choices(nbrs_list, weights=weights_list)[0]  # [('SOD1', 3)]-->('SOD1', 3)
    return next_edge

def _network_traverse_simple(cur, prev_node, G, p, q):
    cur_nbrs = sorted(G.neighbors(cur))
    nbrs_list = []
    weights_list = []

    # search for reachable edges and their weights
    for nbr in cur_nbrs:
        nbr_edges = G[cur][nbr]
        for nbr_edge in nbr_edges.values():
            nbr_edge_type = nbr_edge['type']
            nbr_edge_weight = nbr_edge['weight']

            if G.has_edge(nbr, prev_node) or G.has_edge(prev_node, nbr):
                nbrs_list.append((nbr, nbr_edge_type))
                weights_list.append(nbr_edge_weight)
            elif nbr == prev_node:  # p: Return parameter
                nbrs_list.append((nbr, nbr_edge_type))
                weights_list.append(nbr_edge_weight / p)
            else:  # q: In-Out parameter
                nbrs_list.append((nbr, nbr_edge_type))
                weights_list.append(nbr_edge_weight / q)

    # sample next node and edge type from searched weights
    next_edge = random.choices(nbrs_list, weights=weights_list)[0]  # [('SOD1', 3)]-->('SOD1', 3)
    return next_edge


def _teleport_operation(cur, G_sim):
    cur_nbrs = sorted(G_sim.neighbors(cur))  # 获取邻居并排序
    random.shuffle(cur_nbrs)  # 随机打乱邻居顺序（为什么要先排序后打乱）
    selected_nbrs = []
    distance_sum = 0
    for nbr in cur_nbrs:
        nbr_links = G_sim[cur][nbr]  # 防止有多条边吗
        # print(nbr_links) # {0: {'type': 1, 'weight': 0.44004276628316996, 'id': 5}}
        for i in nbr_links:
            # print(i) #0
            nbr_link_weight = nbr_links[i]['weight']
            distance_sum += nbr_link_weight
            selected_nbrs.append(nbr)
    if distance_sum == 0:
        return False
    rand = np.random.rand() * distance_sum
    threshold = 0

    for nbr in set(selected_nbrs):
        nbr_links = G_sim[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]['weight']
            threshold += nbr_link_weight
            if threshold >= rand:
                next = nbr
                break
    return next


def save_embedding_files(netf: str, sim_netf: str, outputf: str, nodef: str, simnodef: str, nodetypef: str = None,
                         tp_factor: float = 0.5, seed: int = 42,
                         directed: bool = False, weighted: bool = True, em_max_iter: int = 5,
                         num_walks: int = 100, walk_length: int = 10, workers: int = os.cpu_count(),
                         dimension: int = 1024, window_size: int = 8, p: float = 1, q: float = 1,
                         t: float = 0.9, net_delimiter: str = '\t',
                         # ===== 超图增强（可训练 HGAT）=====
                         use_hypergraph: bool = False,
                         hg_eval_n_splits: int = 10,
                         hg_use_gene_hyperedges: bool = True,
                         hg_use_path_hyperedges: bool = True,
                         hg_use_similarity_hyperedges: bool = True,
                         hg_random_hyperedges: bool = False,
                         hg_path_length: int = 2,
                         hg_knn_from_embeddings: bool = True,
                         hg_k: int = 20,
                         hg_layers: int = 2,
                         hg_alpha: float = 0.2,
                         hg_tau: float = 0.6,
                         hg_safe_eval: bool = True,
                         hg_alpha_candidates: list = None,
                         hg_alpha_fixed: Optional[float] = None,
                         hg_pairf: str = None,
                         hg_xgb_params: dict = None,
                        # ===== 可训练 HGAT 参数 =====
                        hg_trainable_hidden_dim: int = 256,
                        hg_trainable_epochs: int = 400,
                        hg_trainable_patience: int = 80,
                        hg_trainable_lr: float = 1e-3,
                        hg_trainable_weight_decay: float = 1e-4,
                        hg_trainable_dropout: float = 0.2,
                        hg_trainable_device: str = "cpu",
                        hg_trainable_verbose: bool = True,
                        hg_trainable_use_multihead_fusion: bool = True,
                        hg_trainable_num_heads: int = 4,
                        hg_trainable_use_pair_level_attention: bool = True,
                        hg_trainable_align_lambda: float = 1e-3,
                        hg_trainable_gate_lambda: float = 1e-3,
                        hg_trainable_gate_lambda_sweep: Optional[List[float]] = None,
                        hg_trainable_use_node_gate: bool = True,
                        hg_trainable_use_residual_delta: bool = True,
                         ):
    set_seed(seed)
    # print('Reading network files...')
    G1 = read_graph(netf, weighted=weighted, directed=directed,
                   delimiter=net_delimiter)
    G2 = nx.read_adjlist(nodef, nodetype=str,
                         create_using=nx.MultiDiGraph())
    G = nx.compose(G1, G2)

    if sim_netf:
        G_sim1 = read_graph(sim_netf,
                           weighted=True,
                           directed=False)
        G_sim2 = nx.read_adjlist(simnodef, nodetype=str,
                             create_using=nx.MultiDiGraph())
        G_sim = nx.compose(G_sim1, G_sim2)

    else:
        G_sim = nx.empty_graph()
        print('> No input similarity file detected!')
        tp_factor = 0

    #输入节点类型
    with open(nodetypef, 'r') as fr:
        lines = fr.readlines()
    node2type = {}
    for line in lines[1:]:
        line = line.strip().split('\t')
        node2type[line[0]] = line[1]

    # print('Training edge type transition matrix...')
    trans_matrix = train_edgetype_transition_matrix(em_max_iter, G,
                                                    netf, net_delimiter, walk_length, p, q) # trans_matrix = np.array([[5e-3, 0.8, 0.2], [0.8, 5e-3, 0.2], [0.5, 0.5, 5e-3]])

    # print(trans_matrix)
    # print('Generating paths...')
    walks = generate_DREAMwalk_paths(G, G_sim, trans_matrix, p, q, num_walks,
                                     walk_length, tp_factor, workers, node2type, t, base_seed=int(seed))

    # '''保存节点序列'''
    # with open('demo2/mmd_walks.txt', 'w') as file:
    #     for row in walks:
    #         # 将每行的元素转换为字符串，用空格隔开
    #         row_str = ';'.join(map(str, row))
    #         # 写入文件，并在每行末尾添加换行符
    #         file.write(row_str + '\n')

    # print('Generating node embeddings...')
    # with open('demo2/mmd_walks_0.9686.txt', 'r') as file:
    #     walks = file.readlines()
    #     # 去除每行的换行符
    # walks = [line.strip() for line in walks]
    # walks = [line.split(';') for line in walks]

    use_hetSG = True if nodetypef != None else False
    # Ensure deterministic node indexing (avoid unordered set)
    nodes_sorted = sorted(G.nodes())
    embeddings = HeterogeneousSG(use_hetSG, walks, nodes_sorted, nodetypef=nodetypef,
                                 embedding_size=dimension, window_length=window_size, workers=workers)
    
    # ========== 超图嵌入增强 ==========
    if use_hypergraph:
        print("\n[超图增强] 开始...")

        if hg_xgb_params is None:
            hg_xgb_params = dict(
                gamma=0, learning_rate=0.049, max_depth=12,
                n_estimators=564, min_child_weight=2,
                subsample=0.81, colsample_bytree=0.54,
            )

        if hg_pairf is None:
            print("[超图增强] hg_pairf 为空，跳过可训练 HGAT（保持原始 embedding）。")
        elif (not torch_available()) or (trainable_hgat_refine is None):
            print("[超图增强] 未检测到 PyTorch，跳过可训练 HGAT（保持原始 embedding）。")
        else:
            base_embeddings = embeddings
            alpha_candidates = [float(hg_alpha_fixed)] if hg_alpha_fixed is not None else (hg_alpha_candidates or [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0])
            sweep_list = hg_trainable_gate_lambda_sweep if (hg_trainable_gate_lambda_sweep is not None and len(hg_trainable_gate_lambda_sweep) > 0) else None

            if sweep_list is not None:
                print(f"\n[Trainable HGAT] gate_lambda 扫描: {sweep_list}")
                results_sweep = []
                for gate_lam in sweep_list:
                    print(f"\n[Trainable HGAT] 训练 gate_lambda={gate_lam} ...")
                    refined = trainable_hgat_refine(
                        base_embeddings=base_embeddings,
                        G=G, G_sim=G_sim, node2type=node2type,
                        pairf=hg_pairf,
                        k=hg_k,
                        use_gene_hyperedges=hg_use_gene_hyperedges,
                        use_path_hyperedges=hg_use_path_hyperedges,
                        use_similarity_hyperedges=hg_use_similarity_hyperedges,
                        random_hyperedges=hg_random_hyperedges,
                        path_length=hg_path_length,
                        knn_from_embeddings=hg_knn_from_embeddings,
                        num_layers=hg_layers,
                        hidden_dim=hg_trainable_hidden_dim,
                        tau=hg_tau,
                        lr=hg_trainable_lr,
                        weight_decay=hg_trainable_weight_decay,
                        epochs=hg_trainable_epochs,
                        patience=hg_trainable_patience,
                        dropout=hg_trainable_dropout,
                        reg_lambda=1e-4,
                        align_lambda=hg_trainable_align_lambda,
                        gate_lambda=gate_lam,
                        out_residual=True,
                        use_residual_delta=hg_trainable_use_residual_delta,
                        use_node_gate=hg_trainable_use_node_gate,
                        device=hg_trainable_device,
                        seed=seed,
                        verbose=hg_trainable_verbose,
                        use_multihead_fusion=hg_trainable_use_multihead_fusion,
                        num_heads=hg_trainable_num_heads,
                        use_pair_level_attention=hg_trainable_use_pair_level_attention,
                    )
                    best_emb_gl = None
                    best_score_gl = None
                    best_alpha_gl = None
                    if hg_safe_eval:
                        print(f"[Safe Eval] gate_lambda={gate_lam} 评估 alpha: {alpha_candidates}")
                        for a in alpha_candidates:
                            cand = blend_embeddings(base_embeddings, refined, alpha=float(a))
                            auroc, aupr = cv_eval_embeddings_xgb(
                                pairf=hg_pairf, emb=cand, xgb_params=hg_xgb_params, seed=seed, n_splits=int(hg_eval_n_splits)
                            )
                            print(f"  alpha={float(a):.2f} -> AUROC={auroc:.4f}, AUPR={aupr:.4f}")
                            if best_score_gl is None or (aupr > best_score_gl[1]) or (abs(aupr - best_score_gl[1]) < 1e-6 and auroc > best_score_gl[0]):
                                best_score_gl = (auroc, aupr)
                                best_emb_gl = cand
                                best_alpha_gl = float(a)
                    else:
                        best_emb_gl = blend_embeddings(base_embeddings, refined, alpha=float(hg_alpha))
                        best_alpha_gl = float(hg_alpha)
                        auroc, aupr = cv_eval_embeddings_xgb(
                            pairf=hg_pairf, emb=best_emb_gl, xgb_params=hg_xgb_params, seed=seed, n_splits=int(hg_eval_n_splits)
                        )
                        best_score_gl = (auroc, aupr)
                    best_emb_gl = best_emb_gl if best_emb_gl is not None else refined
                    results_sweep.append((gate_lam, best_alpha_gl, best_score_gl, best_emb_gl))

                def _score_key(x):
                    gl, ba, sc, _ = x
                    return (sc[1] if sc else 0.0, sc[0] if sc else 0.0)
                best_row = max(results_sweep, key=_score_key)
                best_gate_lam, best_alpha, best_score, best_emb = best_row
                embeddings = best_emb
                print("\n[Trainable HGAT] gate_lambda 扫描结果:")
                for gl, ba, sc, _ in results_sweep:
                    auroc_s, aupr_s = (sc[0], sc[1]) if sc else (float('nan'), float('nan'))
                    mark = " <-- 最优" if gl == best_gate_lam else ""
                    print(f"  gate_lambda={gl}  best_alpha={ba}  AUROC={auroc_s:.4f}  AUPR={aupr_s:.4f}{mark}")
                print(f"[Trainable HGAT] 选用 gate_lambda={best_gate_lam}, alpha={best_alpha}, AUROC={best_score[0]:.4f}, AUPR={best_score[1]:.4f}")
            else:
                print("\n[Trainable HGAT] 开始训练...")
                refined = trainable_hgat_refine(
                    base_embeddings=base_embeddings,
                    G=G, G_sim=G_sim, node2type=node2type,
                    pairf=hg_pairf,
                    k=hg_k,
                    use_gene_hyperedges=hg_use_gene_hyperedges,
                    use_path_hyperedges=hg_use_path_hyperedges,
                    use_similarity_hyperedges=hg_use_similarity_hyperedges,
                    random_hyperedges=hg_random_hyperedges,
                    path_length=hg_path_length,
                    knn_from_embeddings=hg_knn_from_embeddings,
                    num_layers=hg_layers,
                    hidden_dim=hg_trainable_hidden_dim,
                    tau=hg_tau,
                    lr=hg_trainable_lr,
                    weight_decay=hg_trainable_weight_decay,
                    epochs=hg_trainable_epochs,
                    patience=hg_trainable_patience,
                    dropout=hg_trainable_dropout,
                    reg_lambda=1e-4,
                    align_lambda=hg_trainable_align_lambda,
                    gate_lambda=hg_trainable_gate_lambda,
                    out_residual=True,
                    use_residual_delta=hg_trainable_use_residual_delta,
                    use_node_gate=hg_trainable_use_node_gate,
                    device=hg_trainable_device,
                    seed=seed,
                    verbose=hg_trainable_verbose,
                    use_multihead_fusion=hg_trainable_use_multihead_fusion,
                    num_heads=hg_trainable_num_heads,
                    use_pair_level_attention=hg_trainable_use_pair_level_attention,
                )

                if hg_safe_eval:
                    print(f"\n[Safe Eval] 评估 blend alpha: {alpha_candidates}")
                    best_emb = None
                    best_score = None
                    best_alpha = None
                    for a in alpha_candidates:
                        cand = blend_embeddings(base_embeddings, refined, alpha=float(a))
                        auroc, aupr = cv_eval_embeddings_xgb(
                            pairf=hg_pairf, emb=cand, xgb_params=hg_xgb_params, seed=seed, n_splits=int(hg_eval_n_splits)
                        )
                        print(f"  alpha={float(a):.2f} -> AUROC={auroc:.4f}, AUPR={aupr:.4f}")
                        if best_score is None:
                            best_score = (auroc, aupr)
                            best_emb = cand
                            best_alpha = float(a)
                        elif (auroc > best_score[0]) or (abs(auroc - best_score[0]) < 1e-6 and aupr > best_score[1]):
                            best_score = (auroc, aupr)
                            best_emb = cand
                            best_alpha = float(a)
                    embeddings = best_emb if best_emb is not None else refined
                    if best_score is not None:
                        print(f"[Trainable HGAT] Safe Eval 最优 alpha={best_alpha}, AUROC={best_score[0]:.4f}, AUPR={best_score[1]:.4f}")
                else:
                    embeddings = blend_embeddings(base_embeddings, refined, alpha=float(hg_alpha))
                    print(f"[Trainable HGAT] 直接应用完成 (blend alpha={hg_alpha})")
    # =====================================
    
    with open(outputf, 'wb') as fw:
        pickle.dump(embeddings, fw)

    print(f'Node embeddings saved: {outputf}')

    # return walks, embeddings


if __name__ == '__main__':
    args = parse_args()
    save_embedding_files(**args)

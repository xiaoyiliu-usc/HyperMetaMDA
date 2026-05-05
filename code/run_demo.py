from pathlib import Path
import time

from MetaMDA.experiment_io import default_similarity_graph_path
from MetaMDA.generate_embeddings_mmd import save_embedding_files
from MetaMDA.predict_final import predict_mda, predict_all_mda


if __name__ == '__main__':
    #数据集
    dataset = 'MDAD' # MDAD, aBiofilm, MASI

    #输入
    root = Path(__file__).resolve().parents[1]
    d = root / 'data' / dataset
    networkf = str(d / 'demo_graph.txt')
    simf = default_similarity_graph_path(d)
    nodetypef = str(d / 'demo_nodetypes.tsv')
    nodef = str(d / 'demo_graph_node.txt')
    simnodef = str(d / 'demo_similarity_graph_node.txt')
    pairf = str(d / "combined_mda.tsv")

    #输出
    embeddingf = f'../data/{dataset}/embedding_file.pkl'
    mdaf = f'../data/{dataset}/mda.xlsx'

    print('Start......')

    # 记录开始时间
    start_time = time.time()

    t=0.9
    dimension=1024
    window_size=8

    # XGBoost 参数
    xgb_params = dict(
        gamma=0, learning_rate=0.049, max_depth=12,
        n_estimators=564, min_child_weight=2,
        subsample=0.81, colsample_bytree=0.54,
    )

    save_embedding_files(netf=networkf, sim_netf=simf, outputf=embeddingf,
                         nodetypef=nodetypef, nodef=nodef, simnodef=simnodef,
                         dimension=dimension, window_size=window_size, t=t, seed=42,
                         use_hypergraph=True,
                         hg_k=20,
                         hg_layers=2,
                         hg_tau=0.6,
                         hg_safe_eval=True,
                         hg_alpha_candidates=[0.0, 0.5, 0.8, 0.9, 0.95, 1.0],
                         hg_pairf=pairf,
                         hg_xgb_params=xgb_params,
                         hg_trainable_hidden_dim=256,
                         hg_trainable_epochs=400,
                         hg_trainable_patience=50,
                         hg_trainable_gate_lambda=0.003,
                         hg_trainable_gate_lambda_sweep=None,
                         hg_trainable_use_node_gate=True,
                         hg_trainable_use_multihead_fusion=True,
                         hg_trainable_num_heads=4,
                         )

    result = predict_mda(embeddingf=embeddingf, pairf=pairf, modelf='XGBoost',
                         gamma=xgb_params['gamma'], rate=xgb_params['learning_rate'],
                         max_depth=xgb_params['max_depth'], n_estimators=xgb_params['n_estimators'],
                         min_child_weight=xgb_params['min_child_weight'],
                         subsample=xgb_params['subsample'], colsample_bytree=xgb_params['colsample_bytree'],
                         seed=42)

    predict_all_mda(embeddingf=embeddingf, pairf=pairf, nodetypef=nodetypef, mdaf=mdaf,
                    gamma=xgb_params['gamma'], rate=xgb_params['learning_rate'],
                    max_depth=xgb_params['max_depth'], n_estimators=xgb_params['n_estimators'],
                    min_child_weight=xgb_params['min_child_weight'],
                    subsample=xgb_params['subsample'], colsample_bytree=xgb_params['colsample_bytree'],
                    seed=42)
    print(result)

    # 记录结束时间
    end_time = time.time()

    # 计算程序运行时间
    execution_time = end_time - start_time
    print(f"程序运行时间：{execution_time} 秒")


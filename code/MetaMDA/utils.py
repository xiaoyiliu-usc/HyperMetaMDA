import os
import random
import numpy as np
import networkx as nx
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, average_precision_score


def read_graph(edgeList,weighted=True, directed=False,delimiter='\t'):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(edgeList, nodetype=str, 
                             data=(('type',int),('weight',float),('id',int)), 
                             create_using=nx.MultiDiGraph(),
                            delimiter=delimiter)
    else:
        G = nx.read_edgelist(edgeList, nodetype=str,data=(('type',int)), 
                             create_using=nx.MultiDiGraph(),
                            delimiter=delimiter)
        for edge in G.edges():
            edge=G[edge[0]][edge[1]]
            for i in range(len(edge)):
                edge[i]['weight'] = 1.0

    if not directed:
        G = G.to_undirected()

    return G

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # print(f'random seed with {seed}')

def convert_dataset(pairf, embeddingf):
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    xs, ys = [], []
    with open(pairf, 'r') as fin:
        lines = fin.readlines()

    for line in lines[1:]:
        line = line.strip().split('\t')
        drug = line[0]
        dis = line[1]
        label = line[2]
        xs.append(embedding_dict[drug] - embedding_dict[dis])
        ys.append(int(label))
    return xs, ys # list

def return_scores(target_list, pred_list):
    metric_list = [
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        f1_score
    ]

    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list, pred_list))
        else:  # accuracy_score, f1_score
            scores.append(metric(target_list, pred_list.round()))
    return scores
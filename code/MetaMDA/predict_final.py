import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, average_precision_score

from MetaMDA.utils import set_seed
from sklearn.model_selection import KFold

from MetaMDA.evaluation import PR_curve, ROC_curve, get_auroc, get_aupr, confusion_matrix
from MetaMDA.classifier import train_nn, train_RF, train_GBDT, train_AdaBoost, train_SVM, train_xgboost, train_LR, train_DT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--mda_file', type=str, default='mda.xlsx')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--plot', action='store_true', help='plot ROC curve')

    args = parser.parse_args()
    args = {'embeddingf': args.embedding_file,
            'pairf': args.pair_file,
            'mdaf': args.mda_file,
            'n_splits': args.n_splits,
            'seed': args.seed,
            'patience': args.patience,
            'modelf': args.model_checkpoint,
            'testr': args.test_ratio,
            'validr': args.valid_ratio,
            'plot': args.plot,
            }
    return args


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
        xs.append(embedding_dict[drug] * embedding_dict[dis])
        ys.append(int(label))

    return xs, ys # list


def return_scores(target_list, pred_list):
    target_list = np.asarray(target_list).reshape(-1)
    pred_list = np.asarray(pred_list).reshape(-1)
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


def _as_1d_float(arr):
    """Normalize predictions from different classifiers to 1-D float ndarray."""
    return np.asarray(arr, dtype=float).reshape(-1)


def predict_mda(embeddingf: str, pairf: str, modelf: str = 'XGBoost', seed: int = 42,
                gamma: float = 0, rate: float = 0.1, max_depth: int = 6, n_estimators: int = 500,
                min_child_weight: int = 4, subsample: float = 0.8, colsample_bytree: float = 0.9,
                n_splits: int = 10,
                plot: bool = True,
                return_details: bool = False):

    set_seed(seed)

    # 数据
    x, y = convert_dataset(pairf, embeddingf) #2147

    # 创建 KFold 对象
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=seed)

    result_acc = []
    result_AUROC = []
    result_AUPR = []
    result_F1 = []
    all_pred = []
    all_test_index = []

    # 进行10折交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(x)):

        x_train = [x[i] for i in train_index]
        x_test = [x[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index] #数据类型是<class 'numpy.ndarray'> print(type(test_index))

        if modelf == 'XGBoost':
            # XGBoost模型
            clf = XGBClassifier(base_score=0.5, booster='gbtree', eval_metric='error', objective='binary:logistic',
                                gamma=gamma, learning_rate=rate, max_depth=max_depth, n_estimators=n_estimators,
                                tree_method='auto', min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
                                scale_pos_weight=1, max_delta_step=1, seed=seed)

            clf.fit(x_train, y_train)
            preds = clf.predict_proba(np.array(x_test))[:, 1]
        if modelf == 'GBDT':
            preds = train_GBDT(x_train, y_train, np.array(x_test))
        if modelf == 'RF':
            preds = train_RF(x_train, y_train, np.array(x_test))
        if modelf == 'AdaBoost':
            preds = train_AdaBoost(x_train, y_train, np.array(x_test))
        if modelf == 'DT':
            preds = train_DT(x_train, y_train, np.array(x_test))
        if modelf == 'LR':
            preds = train_LR(x_train, y_train, np.array(x_test))
        if modelf == 'SVM':
            preds = train_SVM(x_train, y_train, np.array(x_test))
        # preds = clf.predict(np.array(x_test))

        # # DNN模型
        if modelf == 'DNN':
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            preds = train_nn(x_train, y_train, x_test)

        preds = _as_1d_float(preds)
        y_test = _as_1d_float(y_test)
        scores = return_scores(y_test, preds)
        # print(
        #         f'Fold {fold + 1} | Acc: {scores[0] * 100:.2f}% | AUROC: {scores[1]:.4f} | AUPR: {scores[2]:.4f} | F1-score: {scores[3]:.4f}')

        result_acc.append(scores[0] * 100)
        result_AUROC.append(scores[1])
        result_AUPR.append(scores[2])
        result_F1.append(scores[3])
        all_pred.extend(preds.tolist())
        all_test_index.extend(test_index)

    # print('=' * 50)
    # print(
    #     f'Mean_Acc: {np.mean(result_acc):.2f}% | Mean_AUROC: {np.mean(result_AUROC):.4f} | Mean_AUPR: {np.mean(result_AUPR):.4f} | Mean_F1-score: {np.mean(result_F1):.4f}')
    # print('=' * 50)

    df = pd.DataFrame()
    df["index"] = all_test_index
    df["pred"] = all_pred
    df_sort = df.sort_values(by='index', ascending=True, ignore_index=True) # ignore_index忽略原索引，生成新索引 #ascending=True按升序排序
    MDAs = pd.read_csv(pairf, sep='\t')
    MDAs['pred'] = pd.to_numeric(df_sort['pred'], errors='coerce')
    MDAs['pred_label'] = MDAs['pred'].round()
    # MDAs.to_excel('demo2/pred_mda.xlsx')
    # MDAs.to_csv('RMMDHN_pred_5_aBiofilm.txt', header=True, index=False)
    scores = return_scores(MDAs['label'], MDAs['pred'])
    print(
            f'Acc: {scores[0] * 100:.2f}% | AUROC: {scores[1]:.4f} | AUPR: {scores[2]:.4f} | F1-score: {scores[3]:.4f}')

    if plot:
        print('绘制 ROC 曲线')
        fpr, tpr, thresholds = roc_curve(MDAs['label'], MDAs['pred'])
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {scores[1]:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # 对角线
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    if not return_details:
        return scores

    details = {
        "seed": int(seed),
        "model": str(modelf),
        "n_splits": int(n_splits),
        "acc": float(scores[0]),
        "auroc": float(scores[1]),
        "aupr": float(scores[2]),
        "f1": float(scores[3]),
        # per-fold lists (note: acc list is stored as percentage in original code)
        "fold_acc_pct": [float(v) for v in result_acc],
        "fold_auroc": [float(v) for v in result_AUROC],
        "fold_aupr": [float(v) for v in result_AUPR],
        "fold_f1": [float(v) for v in result_F1],
        "fold_acc_pct_mean": float(np.mean(result_acc)) if len(result_acc) else float("nan"),
        "fold_acc_pct_std": float(np.std(result_acc)) if len(result_acc) else float("nan"),
        "fold_auroc_mean": float(np.mean(result_AUROC)) if len(result_AUROC) else float("nan"),
        "fold_auroc_std": float(np.std(result_AUROC)) if len(result_AUROC) else float("nan"),
        "fold_aupr_mean": float(np.mean(result_AUPR)) if len(result_AUPR) else float("nan"),
        "fold_aupr_std": float(np.std(result_AUPR)) if len(result_AUPR) else float("nan"),
        "fold_f1_mean": float(np.mean(result_F1)) if len(result_F1) else float("nan"),
        "fold_f1_std": float(np.std(result_F1)) if len(result_F1) else float("nan"),
    }
    return details


def predict_all_mda(embeddingf: str, pairf: str, nodetypef: str, mdaf: str, seed: int = 42,
                gamma: float = 0, rate: float = 0.1, max_depth: int = 6, n_estimators: int = 500,
                min_child_weight: int = 4, subsample: float = 0.8, colsample_bytree: float = 0.9):

    set_seed(seed)
    x, y = convert_dataset(pairf, embeddingf) #2147

    clf = XGBClassifier(base_score=0.5, booster='gbtree', eval_metric='error', objective='binary:logistic',
                        gamma=gamma, learning_rate=rate, max_depth=max_depth, n_estimators=n_estimators,
                        tree_method='auto', min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
                        scale_pos_weight=1, max_delta_step=1, seed=seed)

    clf.fit(x, y)
    # preds = clf.predict_proba(np.array(x))[:, 1]

    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    node = pd.read_csv(nodetypef, sep='\t')
    drugs = node[node["type"] == "drug"]
    drug_list = drugs['node'].tolist()
    microbes = node[node["type"] == "disease"]
    microbe_list = microbes['node'].tolist()
    df = pd.DataFrame(index=microbe_list, columns=drug_list)

    for drug in drug_list:
        for microbe in microbe_list:
            xs = embedding_dict[drug] * embedding_dict[microbe]
            preds = clf.predict_proba(xs.reshape(1, 1024))[:, 1]
            df.at[microbe, drug] = preds[0]

    df.to_excel(mdaf)
    print(f'All microbe-drug association saved: {mdaf}')



if __name__ == '__main__':
    args = parse_args()
    predict_mda(**args)



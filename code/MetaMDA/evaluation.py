# 混淆矩阵、P、R和F1值，作出P-R曲线、ROC曲线，并求AUC

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def PR_curve(y, pred):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  # 从大到小排序
    index = np.argsort(pred)[::-1]  # 从大到小排序
    y_sort = y[index]

    Pre = []
    Rec = []
    for i, item in enumerate(pred_sort):
        if i == 0:  # 因为计算precision的时候分母要用到i，当i为0时会出错，所以单独列出
            Pre.append(1)
            Rec.append(0)

        else:
            Pre.append(np.sum((y_sort[:i] == 1)) / i)  # 从大到小阈值排列，所以全部都预测为正例
            Rec.append(np.sum((y_sort[:i] == 1)) / pos)

    Pre.append(0)
    Rec.append(1)  # 加入最后一个点，使图像封闭
    pr_value = get_aupr(Rec, Pre)
    # 画图
    plt.figure('PR Curve')
    plt.plot(Rec, Pre, 'r')
    label = "AUPR: " + str(pr_value)

    plt.title('PR Curve')
    plt.plot([0, 1], [1, 0], 'k--', linewidth=0.8, label=label)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='lower right')
    plt.show()

    return pr_value


def ROC_curve(y, pred):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  # 从大到小排序
    index = np.argsort(pred)[::-1]  # 从大到小排序
    y_sort = y[index]
    tpr = []
    fpr = []
    thr = []
    for i, item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    tpr.append(1)
    fpr.append(1)
    auc_value = get_auroc(fpr, tpr)

    # 画图
    plt.figure('ROC Curve')
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'r')
    label = "AUROC: " + str(auc_value)
    plt.plot([[0, 0], [1, 1]], 'k--', linewidth=0.8, label=label)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")  # 若需显示label,必须在show之前加这一句
    plt.show()

    return auc_value


# 计算AUROC值
def get_auroc(fpr, tpr):
    auc_value = 0.0
    for ix in range(len(fpr[:-1])):
        x_right, x_left = fpr[ix], fpr[ix + 1]
        y_top, y_bottom = tpr[ix], tpr[ix + 1]
        temp_area = abs(x_right - x_left) * (y_top + y_bottom) * 0.5
        auc_value += temp_area
    return auc_value


# 计算AUPR值
def get_aupr(rec, pre):
    pr_value = 0.0
    for ix in range(len(rec[:-1])):
        x_right, x_left = rec[ix], rec[ix + 1]
        y_top, y_bottom = pre[ix], pre[ix + 1]
        temp_area = abs(x_right - x_left) * (y_top + y_bottom) * 0.5
        pr_value += temp_area
    return pr_value


def confusion_matrix(true_label, predict_label):
    pos_label_count = 0
    for k in range(len(true_label)):
        if true_label[k] == 1:
            pos_label_count += 1
    pre_pos_count = 0
    for k in range(len(predict_label)):
        if predict_label[k] == 1:
            pre_pos_count += 1

    correct_predict = 0
    for pre, tru in zip(predict_label, true_label):
        if pre == 1 and tru == 1:
            correct_predict += 1

    TP = correct_predict
    FN = pos_label_count - TP
    FP = pre_pos_count - TP
    TN = len(true_label) - pos_label_count - FP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    tpr = recall
    fpr = FP / (TN + FP)
    F1 = 2 * precision * recall / (precision + recall)

    return TP, FN, FP, TN, precision, recall, F1
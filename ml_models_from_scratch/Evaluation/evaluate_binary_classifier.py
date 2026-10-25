import numpy as np


def evaluate(y_act, y_pred):
    y_act = np.asarray(y_act)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_act == 1) & (y_pred == 1))
    TN = np.sum((y_act == 0) & (y_pred == 0))
    FP = np.sum((y_act == 0) & (y_pred == 1))
    FN = np.sum((y_act == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_act)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 1
    recall = TP / (TP + FN) if (TP + FN) != 0 else 1
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }

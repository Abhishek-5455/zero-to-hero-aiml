import numpy as np

def evaluate(y_act, y_pred):
    y_act = np.asarray(y_act).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate((y_act, y_pred)))
    C = len(classes)

    cm = np.zeros((C, C), dtype=int)

    classes_to_idx = {label: i for i, label in enumerate(classes)}

    for actual, predicted in zip(y_act, y_pred):
        cm[classes_to_idx[actual]][classes_to_idx[predicted]] += 1

    TP = np.diag(cm)
    FP = cm.sum(axis = 0) - TP
    FN = cm.sum(axis = 1) - TP

    precision = np.divide(
        TP, TP + FP,
        out = np.zeros(C, dtype=float),
        where = (TP + FP) != 0
    )

    recall = np.divide(
        TP, TP + FN,
        out = np.zeros(C, dtype=float),
        where = (TP + FN) != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(C, dtype=float),
        where=(precision + recall) != 0
    )


    return {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'cm': cm
    }


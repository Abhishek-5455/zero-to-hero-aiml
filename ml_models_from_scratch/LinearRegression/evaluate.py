import numpy as np

def evaluate(y_act, y_pred):
    y_act = np.asarray(y_act)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_act - y_pred) ** 2)
    mae = np.mean(np.abs(y_act - y_pred))
    r2 = 1 - ((np.sum((y_act - y_pred) ** 2)) / np.sum((y_act - np.mean(y_act)) ** 2))

    return {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R2 Score": r2
    }


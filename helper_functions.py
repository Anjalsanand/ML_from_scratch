import numpy as np
def MSE(y_preds, y_true):
    mse=np.mean((y_preds-y_true)**2)
    return mse


def accuracy(y_preds, y_true):
    return np.sum(y_preds==y_true)/len(y_true)
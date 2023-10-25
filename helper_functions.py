import numpy as np
def MSE(y_preds, y_true):
    mse=np.mean((y_preds-y_true)**2)
    return mse
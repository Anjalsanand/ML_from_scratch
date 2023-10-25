import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
    


class KNN:

    def __init__(self, k=3):
        self.k=k

    def fit(self, X,Y):
        self.X_train=X
        self.Y_train=Y
    

    def predict(self,X):
        #prediction
        prediction=[self._predict(x) for x in X]
        return np.array(prediction)

    def _predict(self, x):
        #compute distance
        distances=[euclidean_dist(x1=x, x2=x_train) for x_train in self.X_train]
        #get k nearest samples, labels
        k_nearest_ind=np.argsort(distances)[:self.k]
        K_indices_labels=[ self.Y_train[i] for i in k_nearest_ind]
        #most common class
        most_common=Counter(K_indices_labels).most_common(1)
        return most_common[0][0]








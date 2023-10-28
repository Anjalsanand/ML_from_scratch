import numpy as np

class SVM:
    def __init__(self, lr=0.01, lambda_param=0.01,num_iters=1000):
        self.lr=lr
        self.num_iters=num_iters
        self.lambda_param=lambda_param
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        y=np.where(y<=0,-1,1)
        n_samples, n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.num_iters):
            for idx, xi in enumerate(X):
                condition=y[idx]*(np.dot(xi, self.weights)+self.bias)
                if condition>=1:
                    self.weights=self.weights-self.lr*(2*self.lambda_param*self.weights)
                else:
                    self.weights=self.weights-self.lr*(2*self.lambda_param*self.weights-np.dot(xi,y[idx]))
                    self.bias=self.bias-self.lr*y[idx]

            


    def predict(self, X_test):
        linear=np.dot(X_test, self.weights)+self.bias
        return np.sign(linear)
    
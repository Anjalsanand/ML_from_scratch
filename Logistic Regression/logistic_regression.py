import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.01, num_iter=100):
        self.lr=lr
        self.num_iter=num_iter
        self.weights=None
        self.bias=None

    def fit(self, X_train, y_train): #convention of sklearn library
        num_samples, num_features=X_train.shape
        self.weights=np.zeros(num_features)
        self.bias=0

        for _ in range(self.num_iter):
            linear_model=np.dot(X_train, self.weights)+self.bias
            y_preds=self._sigmoid(linear_model)

            dw=(1/num_samples)* np.dot(X_train.T, (y_preds-y_train))
            db=(1/num_features)*np.sum(y_preds-y_train)

            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self, X_test):
        linear_model=np.dot(X_test, self.weights)+self.bias
        y_preds=self._sigmoid(linear_model)
        y_class=[1 if i>=0.5 else 0 for  i in y_preds]
        return np.array(y_class)


    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))







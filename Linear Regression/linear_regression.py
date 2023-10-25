'''In regression we have to predict continous values, whereas in classification we want to predict discreate value
approximate data with linear function
we use linear function to predict values
 y=mx+b
'''
import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, num_iter=500):
        self.lr=0.01
        self.num_iter=num_iter
        self.weights=None
        self.bias=None

    def fit(self, X_train, y_train):
        num_samples, num_features=X_train.shape
        self.weights=np.zeros(num_features)
        self.bias=0

        for _ in range(self.num_iter):
             
             y_preds=np.dot(X_train, self.weights)+self.bias
             dw=(1/num_samples)* np.dot(X_train.T, (y_preds-y_train))
             db=(1/num_features)*np.sum(y_preds-y_train)
             #weight and bias updates
             self.weights=self.weights-self.lr*dw
             self.bias=self.bias-self.lr*db

    def predict(self,X_test):
        return np.dot(X_test, self.weights)+self.bias
        
             
        


                       


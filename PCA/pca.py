
import numpy as np

class PCA:
    def __init__(self, n_dims ):
        self.n_dims=n_dims
        self.components=None
        self.mean=None

    def fit(self,X, y):
        #calculate mean
        self.mean=np.mean(X, axis=0)
        X=X-self.mean
        #covariance
        """ input od covariance n_feautres x n_samples ,each colum is one sample, thus transponse X"""
        cov=np.cov(X.T)

        #eigenvalues and eigenvectors
        eigenvalues, eigenvectors=np.linalg.eig(cov) 
        """ numpy return eigenvector in column vise, thus tranponse """
        #sort eigenvalues
        eigenvectors=eigenvectors.T
        idx=np.argsort(eigenvalues)[::-1]
        eigenvalues=eigenvalues[idx]
        eigenvectors=eigenvectors[idx]
        #conculate  n components
        self.components=eigenvectors[0:self.n_dims]

    def transform(self, X):
        X=X-self.mean
        return  np.dot(X, self.components.T)
        



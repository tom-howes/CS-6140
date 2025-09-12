import numpy as np

class Normal_linear:

    def fit(self, X, y):
        X_transpose = X.T
        X_transpose_X = X_transpose.dot(X)
        X_transpose_y = X_transpose.dot(y)
        w = np.linalg.solve(X_transpose_X, X_transpose_y)
        return w
    
    def add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def predict(self, X, w):
        return X.dot(w)

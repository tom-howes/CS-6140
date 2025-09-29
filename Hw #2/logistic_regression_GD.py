import numpy as np
import sys

class Logistic_regression:
    
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.accuracy_updates = []
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.x = X
        self.y = y
        
        self.w = np.random.standard_normal(size=self.n)
        for current_iteration in range(self.epochs):
            y_pred = self.predict(self.x)
            error = y - y_pred
            self.w = self.w + self.lr*(self.x.T @ error)
            train_acc = self.get_accuracy(y, y_pred)
            self.accuracy_updates.append(train_acc)
        return self.accuracy_updates

    def predict(self, x):
        y_pred = x @ self.w
        return self.sigmoid(y_pred)
    
    def get_accuracy(self, true, pred):
        for i in range(len(pred)):
            if pred[i] > 0.50:
                pred[i] = 1
            else:
                pred[i] = 0
        accuracy = np.mean(true == pred)
        return accuracy
    
    def confusion_matrix(self, true, pred, matrix):
        for i in range(len(pred)):
            # Binary classify based on 0.5 threshold
            if pred[i] > 0.50:
                pred[i] = 1
            else:
                pred[i] = 0
        
            # Update confusion matrix
            if true[i] == 0:
                if pred[i] == 0:
                    matrix[0][0] += 1
                else:
                    matrix[0][1] += 1
            else:
                if pred[i] == 1:
                    matrix[1][1] += 1
                else:
                    matrix[1][0] += 1
        return matrix
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


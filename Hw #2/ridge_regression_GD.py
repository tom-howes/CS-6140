import numpy as np
from sklearn.metrics import mean_squared_error
import sys

class Ridge_regression:

    def __init__(self, learning_rate, lambda_val, epochs, batch_size=32):
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_updates = []
        self.mse_updates = []
        self.accuracy_updates = []
    
    def fit(self, X, y, data: str):
        self.m, self.n = X.shape
        self.x = X
        self.y = y
        # Create random weight vector with standard normal distribution
        self.w = np.random.standard_normal(size=self.n)
        for current_iteration in range(self.epochs):
            y_pred = X @ self.w
            mse = mean_squared_error(y, y_pred)

            l2_term = self.lambda_val * np.sum(self.w **2)
                
            cost = mse + l2_term
            if data == "housing":
                self.cost_updates.append(cost)
                self.mse_updates.append(mse)
            if data == "spambase":
                accuracy = self.get_accuracy(y, y_pred)
                self.accuracy_updates.append(accuracy)
                self.cost_updates.append(cost)
                # print(f"cost: {cost:.2f} accuracy: {accuracy:.3f} iteration: {current_iteration}")
            # Update weights
            self.update_weights_minibatch()

            ## Print cost to see how values change after every iteration
            # print(f"cost: {cost:.2f} mse: {mse:.2f} iteration: {current_iteration}")
            
        if data == "housing":
            return self.cost_updates, self.mse_updates
        else:
            return self.cost_updates, self.accuracy_updates
    
    def update_weights_minibatch(self, batch_size=32):
        indices = np.random.permutation(self.m)
        for i in range(0, self.m, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = self.x[batch_indices]
            y_batch = self.y[batch_indices]
            y_pred_batch = X_batch.dot(self.w)

            # Calculate gradient
            dw = (-( 2 * (X_batch.T).dot(y_batch - y_pred_batch)) + (2 * self.lambda_val * self.w)) / len(batch_indices)

            self.w = self.w - self.learning_rate * dw

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
            if pred[i] > 0.50:
                pred[i] = 1
            else:
                pred[i] = 0
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

    def predict(self, X):
        return X.dot(self.w)
    



            

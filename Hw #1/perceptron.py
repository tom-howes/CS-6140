import numpy as np
import sys

class Perceptron:

    def __init__(self, learning_rate=0.1, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation_function = self.unit_step_func
        self.weights = None
        self.total_mistakes = []
    def unit_step_func(self, x):
        return np.where(x>=0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        y_ = np.array([1 if i > 0 else 0 for i in y])
        for _ in range(self.iterations):
            mistakes = 0
            for index, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights)
                y_hat = self.activation_function(linear_output)
                update = self.learning_rate *(y_[index] - y_hat)
                if update != 0:
                    mistakes += 1

                self.weights += update * x_i
            self.total_mistakes.append(mistakes)
    
    def print_results(self):
        for i, mistakes in enumerate(self.total_mistakes):
            print(f"Iteration {i + 1}, total mistakes {mistakes}")
            if mistakes == 0:
                break
        print("Classifier weights: ", [w for w in self.weights])
        w_0 = self.weights[0]
        print("Normalized with threshold: ", [w / -w_0 for w in self.weights[1:]])
        
        

import numpy as np
import sys

class SVM_SMO:

    def __init__(self, c, tolerance, max_passes):
        self.c = c
        self.tol = tolerance
        self.max_passes = max_passes
        self.support_vectors = []
    
    def fit(self, X, y):
        m, n = X.shape
        print(X.shape)
        self.alphas = np.zeros(m)
        self.b = 0
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                f_xi = np.sum(self.alphas * y * self.linear_kernel(X, X[i])) + self.b
                E_i = f_xi - y[i]
                if (y[i] * E_i < -self.tol and self.alphas[i] < self.c) or (y[i] * E_i > self.tol and self.alphas[i] > 0):
                    valid_numbers = np.setdiff1d(range(m), i)
                    j = np.random.choice(valid_numbers)
                    f_xj = np.sum(self.alphas * y * self.linear_kernel(X, X[j])) + self.b
                    E_j = f_xj - y[j]
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    L, H = 0, 0
                    if y[i] != y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.c, self.c + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.c)
                        H = min(self.c, alpha_i_old + alpha_j_old)
                    if L == H:
                        print(f"L = H, skipping: L={L}, H={H}")
                        continue
                    eta = self.linear_kernel(X[i], X[i])  + self.linear_kernel(X[j], X[j]) - 2 * self.linear_kernel(X[i], X[j])
                    print(f"Eta: {eta}")
                    if eta <= 0:
                        continue
                    alpha_j_new = alpha_j_old + (y[j] * (E_i - E_j) / eta)
                    print(f"Alpha_j before clipping: {alpha_j_new}")
                    if alpha_j_new > H:
                        alpha_j_new = H
                    elif alpha_j_new < L:
                        alpha_j_new = L
                    else:
                        pass
                    print(f"Alpha_j after clipping: {alpha_j_new}")
                    if abs(alpha_j_new - alpha_j_old) < np.power(10.0, -5):
                        print("Insufficient alpha change, skipping")
                        continue
                    self.alphas[j] = alpha_j_new
                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

                    b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * (X[i] @ X[i]) - y[j] * (self.alphas[j] - alpha_j_old) * (X[i] @ X[j])
                    b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * (X[i] @ X[j]) - y[j] * (self.alphas[j] - alpha_j_old) * (X[j] @ X[j])

                    if 0 < self.alphas[i] < self.c:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                break
            else:
                passes += 1

        for i in range(m):
            if self.alphas[i] > 0:
                self.support_vectors.append([X[i], self.alphas[i], y[i]])
                    
        
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def predict(self, X_test):
        predictions = []
        for xi in X_test:
            f_xi = 0
            for sv_xi, alpha, sv_yi in self.support_vectors:
                f_xi += alpha * sv_yi * self.linear_kernel(sv_xi, xi)
            f_xi += self.b
            predictions.append(np.sign(f_xi))
        return np.array(predictions)
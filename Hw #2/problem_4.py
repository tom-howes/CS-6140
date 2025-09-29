import numpy as np
from problem_1 import spambase_data_handling
from sklearn.model_selection import train_test_split
from svm_SMO import SVM_SMO
import sys

def print_accuracy(true, pred, data: str=""):
    accuracy = np.mean(true==pred)
    print(f"{data} Accuracy: {accuracy}")
def problem_4():
    X, y = spambase_data_handling()
    y = np.where(y == 0, -1, 1)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
    model = SVM_SMO(c=0.8, tolerance=0.001, max_passes=50)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print_accuracy(y_train, y_train_pred, "Train")
    print_accuracy(y_test, y_test_pred, "Test")

def test():
    # Simple linearly separable data
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, -1, -1])

    # Make sure labels are correct
    print("Labels:", np.unique(y))  # Should be [-1, 1]

    svm = SVM_SMO(c=1.0, tolerance=1e-3, max_passes=100)
    svm.fit(X, y)

    print("Final alphas:", svm.alphas)
    print("Support vectors found:", len(svm.support_vectors))
    print("Final b:", svm.b)

    # Test prediction
    test_X = np.array([[2, 2], [1, 1]])
    predictions = svm.predict(test_X)
    print("Predictions:", predictions)
if __name__=="__main__":
    problem_4()
    # test()
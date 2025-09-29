from problem_1 import spambase_data_handling, add_bias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import sys
path = "datasets/"
def create_array_from_file(file: str):
    with open (path + file, "r") as f:
        np_array = np.loadtxt(f, dtype=np.float32)
        return np_array
    
def digits_data_handling():
    train_features = np.loadtxt(path + "training_image.txt", delimiter=",", dtype=np.float32)
    train_labels = np.loadtxt(path + "training_label.txt", delimiter=",", dtype=np.float32)
    test_features = np.loadtxt(path + "testing_image.txt", delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(path + "testing_label.txt", delimiter=",", dtype=np.float32)
    return train_features, test_features, train_labels, test_labels

def run_svm(kernel: str, c: float, X_train, X_test, y_train, y_test):
    svm = SVC(kernel=kernel, C=c)
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    print(f"{kernel} Kernel | c = {c}")
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
def svm_part_a():
    X, y = spambase_data_handling()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size = 0.2)
    print("Spambase Part A)\n")
    run_svm('linear', 1.0, X_train, X_test, y_train, y_test)
    run_svm('rbf', 4.0, X_train, X_test, y_train, y_test)
    run_svm('poly', 100.0, X_train, X_test, y_train, y_test)

def svm_part_b():
    X_train, X_test, y_train, y_test = digits_data_handling()
    print("Digits Part B)\n")
    run_svm('linear', 1.0, X_train, X_test, y_train, y_test)

if __name__=="__main__":
    svm_part_a()
    svm_part_b()
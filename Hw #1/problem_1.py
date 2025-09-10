import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from data_handling import normalize_data, create_array_from_file, get_features, get_labels
import sys

float_formatter = "{:.3f}".format

np.set_printoptions(formatter={'float_kind':float_formatter})
from ucimlrepo import fetch_ucirepo
path = "datasets/"

# Adds bias - x is df, m is number of samples, n is no. of features
# creates column of ones and appends to start of array
def add_bias(x, rows: int, cols: int):
    x_bias = np.ones((rows, 1))
    x = np.reshape(x, (rows, cols))
    updated_x = np.append(x_bias, x, axis=1)
    return updated_x

def mean_sum_of_squares(y, y_hat):
    squared_differences = (y - y_hat) ** 2
    mse = np.mean(squared_differences)
    return mse

def print_errors(training_error, test_error):
    print("Training error: ", training_error)
    print("Test error: ", test_error)

# # Reverse normalizing to get original value
# def reverse_normalize(df):
#     return value * (max - min) + min

def kfolds(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    errors = []
    for (train_index, test_index) in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        w = np.dot(np.linalg.inv(np.dot(np.transpose(X_train), X_train)), np.dot(np.transpose(X_train), y_train))

        y_train_hat = np.dot(X_train, w)
        print(y_train_hat)



def housing_part_a():
    housing_training = create_array_from_file("housing_training.txt")
    housing_test = create_array_from_file("housing_test.txt")
    X_training = get_features(housing_training)
    X_test = get_features(housing_test)
    X_training = normalize_data(X_training)
    X_test = normalize_data(X_test)

    y_test = get_labels(housing_test)
    y_training = get_labels(housing_training)   
    
    # reshape to 2d array and add bias
    n, m = X_training.shape
    X_training = add_bias(X_training, n, m)
    y_training = y_training.reshape(n, 1)

    # Calculate w with normal equation
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X_training), X_training)), np.dot(np.transpose(X_training), y_training))

    # Get dot product of w and training features to predict on training
    y_training_hat = np.dot(X_training, w)

    # Calculate mean sum of squares on training data
    training_error = mean_sum_of_squares(y_training, y_training_hat)

    # reshape to 2d array and add bias
    n, m = X_test.shape
    X_test = add_bias(X_test, n, m)
    y_test = y_test.reshape(n, 1)

    # Get dot product of w and test features to predict on test
    y_test_hat = np.dot(X_test, w)
    test_error = mean_sum_of_squares(y_test, y_test_hat)

    print_errors(training_error, test_error)

def spambase_part_a():
    # Load file to np array
    data = np.loadtxt(path + "spambase.data", delimiter=",", dtype=np.float32)
    n_samples, n_features = data.shape
    n_features -= 1

    # get features and targets
    X = data[:, 0:n_features]
    y = data[:, n_features]
    print(X.shape, y.shape)

    # k fold cross validation
    kfolds(X, y)


    # shuffle dataset

def main():
    housing_part_a()
    # spambase_part_a()

if __name__ == "__main__":
    main()

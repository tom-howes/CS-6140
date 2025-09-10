import pandas as pd
import numpy as np
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


def housing_part_a():
    housing_training = create_array_from_file("housing_training.txt")
    housing_test = create_array_from_file("housing_test.txt")
    X_training = get_features(housing_training)
    X_test = get_features(housing_test)
    X_training, X_test = normalize_data(X_training, X_test)

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
    # fetch dataset 
    spambase = fetch_ucirepo(id=94) 

    # data (as pandas dataframes) 
    X = spambase.data.features 
    y = spambase.data.targets 

    # metadata
    print(spambase.metadata) 

    # variable information 
    print(spambase.variables) 

    np_array = np.array(spambase.data.features)
    print(np_array)

    # shuffle dataset
    X_training = X_training.sample(frac=1)
    print(X_training.head())

def main():
    # housing_part_a()
    spambase_part_a()

if __name__ == "__main__":
    main()

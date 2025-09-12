import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from data_handling import min_max_normalize, zero_mean_normalize, create_array_from_file, get_features, get_labels
from normal_linear import Normal_linear
import sys

float_formatter = "{:.3f}".format

np.set_printoptions(formatter={'float_kind':float_formatter})
path = "datasets/"

# Adds bias - x is df, m is number of samples, n is no. of features
# creates column of ones and appends to start of array
def add_bias(x):
    return np.c_[np.ones((x.shape[0], 1)), x]

def mean_sum_of_squares(y, y_hat):
    squared_differences = (y - y_hat) ** 2
    mse = np.mean(squared_differences)
    return mse

def print_errors(training_error, test_error):
    print("Training error: ", training_error)
    print("Test error: ", test_error)

def get_accuracy(true, pred):
    for i in range(len(pred)):
        if pred[i] > 0.49:
            pred[i] = 1
        else:
            pred[i] = 0
    accuracy = np.mean(true == pred)
    return accuracy

def print_accuracy(accuracies: list):
    train_accuracy = np.mean(accuracies[:][0])
    test_accuracy = np.mean(accuracies[:][1])
    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)

def kfolds(X, y):
    # Split into 10 folds and shuffle
    kf = KFold(n_splits=10, shuffle=True)

    # Array to store model test/train prediction accuracy over each iteration
    accuracies = []

    # Normal linear model
    model = Normal_linear()

    for (train_index, test_index) in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        
        # Min-max normalization
        X_train = zero_mean_normalize(X_train)
        X_test = zero_mean_normalize(X_test)

        X_train = add_bias(X_train)

        # Apply normal equation on training set
        w = model.fit(X_train, y_train)
        
        # use weights to predict on training set
        y_train_hat = model.predict(X_train, w)
        
        # use weights to predict on test set
        X_test = add_bias(X_test)
        y_test_hat = model.predict(X_test, w)

        # use threshold to get the accuracy of the predictions
        train_accuracy = get_accuracy(y_train, y_train_hat)
        test_accuracy = get_accuracy(y_test, y_test_hat)

        # append to accuracies array and repeat
        accuracies.append([train_accuracy, test_accuracy])
    
    print_accuracy(accuracies)

def housing_part_a():
    housing_training = create_array_from_file("housing_training.txt")
    housing_test = create_array_from_file("housing_test.txt")
    X_training = get_features(housing_training)
    X_test = get_features(housing_test)

    X_training = min_max_normalize(X_training)
    X_test = min_max_normalize(X_test)

    y_test = get_labels(housing_test)
    y_training = get_labels(housing_training)   
    
    # reshape to 2d array and add bias
    X_training = add_bias(X_training)
    y_training = y_training.reshape(y_training.shape[0], 1)

    # Init Normal Equation model
    housing_model = Normal_linear()

    # Get w by applying normal equation
    w = housing_model.fit(X_training, y_training)

    # predict on training data with dot product of X_train and w
    y_training_hat = housing_model.predict(X_training, w)

    # Calculate mean sum of squares on training data
    training_error = mean_sum_of_squares(y_training, y_training_hat)

    # Repeat on test dataset
    X_test = add_bias(X_test)
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_test_hat = housing_model.predict(X_test, w)
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

    # k fold cross validation
    kfolds(X, y)


    # shuffle dataset

def main():
    # housing_part_a()
    spambase_part_a()

if __name__ == "__main__":
    main()

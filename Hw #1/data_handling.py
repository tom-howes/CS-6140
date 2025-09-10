import pandas as pd
import numpy as np
import sys

path = "datasets/"

# Reading file and creating pandas dataframe
def create_array_from_file(file: str):
    with open (path + file, "r") as f:
        np_array = np.loadtxt(f, dtype=float)
        return np_array
    
# Min-max normalization of data, using train min/max to avoid data leakage from test set
def normalize_data(train, test):
    normalized_train = ((train - train.min(axis=0)) / (train.max(axis=0) - train.min(axis=0)))
    normalized_test = ((test - train.min(axis=0)) / (train.max(axis=0) - train.min(axis=0)))
    return normalized_train, normalized_test

# Returns all but last column (features) of dataset
def get_features(data):
    return data[:, :-1] 

# Returns last column (labels) of dataset
def get_labels(data):
    return data[:, -1]
    
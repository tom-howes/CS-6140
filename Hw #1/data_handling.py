import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sys

path = "datasets/"

# Reading file and creating pandas dataframe
def create_array_from_file(file: str):
    with open (path + file, "r") as f:
        np_array = np.loadtxt(f, dtype=np.float32)
        return np_array
    
# Min-max normalization of data, using train min/max to avoid data leakage from test set
def min_max_normalize(data):
    return ((data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)))

def zero_mean_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std
# Returns all but last column (features) of dataset
def get_features(data):
    return data[:, :-1] 

# Returns last column (labels) of dataset
def get_labels(data):
    return data[:, -1]

        
        
    
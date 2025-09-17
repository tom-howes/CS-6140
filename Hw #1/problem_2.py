import pandas as pd
import numpy as np
from data_handling import housing_data_handling, add_bias, min_max_normalize
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def plot_graph(x_axis, y_axis, dataset: str):
    plt.plot(x_axis, y_axis)
    plt.title('L1 Regression Performance')
    plt.xlabel('Lambda Value')
    plt.ylabel(f'{dataset} Performance')
    plt.show()
def part_a():
    # Extract data from txt files and normalize
    X_train, X_test, y_train, y_test = housing_data_handling()
    
    # Add bias to X and reshape y
    X_train = add_bias(X_train)
    y_train = y_train.reshape(y_train.shape[0], 1)
    X_test = add_bias(X_test)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # Create an array of lambda values ranging 0-1 in 0.01 increments
    alphas = np.arange(0, 1, 0.01, dtype=float)

    # Arrays to store results
    x_axis = []
    y_axis_train = []
    y_axis_test = []
    # init model
    for i in alphas:
        model = Lasso(alpha=i)

        # Get coefficients with given alpha value = 1.0
        model.fit(X_train, y_train)

        # Predict on training set and test set
        y_train_hat = model.predict(X_train)
        y_test_hat = model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_hat)
        test_mse = mean_squared_error(y_test, y_test_hat)
        print(f"Lambda value: {i}")
        print(f"Train MSE: {train_mse}")
        print(f"Test MSE: {test_mse}")
        x_axis.append(i)
        y_axis_test.append(test_mse)
        y_axis_train.append(train_mse)
    
    plot_graph(x_axis, y_axis_train, 'Train')
    plot_graph(x_axis, y_axis_test, 'Test')
    

def main():
    part_a()
if __name__ == "__main__":
    main()
    
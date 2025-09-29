from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from ridge_regression_GD import Ridge_regression
from logistic_regression_GD import Logistic_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy import integrate
path = "datasets/"

def create_array_from_file(file: str):
    with open (path + file, "r") as f:
        np_array = np.loadtxt(f, dtype=np.float32)
        return np_array
def add_bias(x):
    return np.c_[np.ones((x.shape[0], 1)), x]

def get_accuracy(true, pred):
    for i in range(len(pred)):
        if pred[i] > 0.50:
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
    

def housing_data_handling():
    housing_train = create_array_from_file("housing_training.txt")
    housing_test = create_array_from_file("housing_test.txt")
    X_train = housing_train[:, :-1]
    y_train = housing_train[:, -1]
    X_test = housing_test[:, :-1]
    y_test = housing_test[:, -1]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    return X_train, X_test, y_train, y_test

def spambase_data_handling():
    data = np.loadtxt(path + "spambase.data", delimiter=",", dtype=np.float32)
    n_features = data.shape[1]
    n_features -= 1

    # get features and targets
    X = data[:, 0:n_features]
    y = data[:, n_features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def kfolds(X, y, modelType: str=""):
    # Split into 10 folds and shuffle
    kf = KFold(n_splits=10, shuffle=True)

    # Array to store model test/train prediction accuracy over each iteration
    accuracies = []

    # Array to store tn (0, 0), fn (1, 0), fp (0, 1), tp (1, 1) values
    confusion_matrix = np.zeros((2, 2))

    # Arrays to store true and preds
    y_true, y_pred = [], []
    # Ridge Model
    model = Ridge_regression(learning_rate=0.001, lambda_val=0.58, epochs=1000)

    if modelType == "logistic":
        model = Logistic_regression(learning_rate=0.0015, epochs=2600)

    for (train_index, test_index) in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        # Fit model on training set
        train_accuracy = []
        if modelType == "logistic":
            train_accuracy = model.fit(X_train, y_train)
        else:
            train_cost, train_accuracy = model.fit(X_train, y_train, "spambase")

        
        # use model to predict on test set
        y_test_hat = model.predict(X_test)
        confusion_matrix = model.confusion_matrix(y_test, y_test_hat, confusion_matrix)
        # use threshold to get the accuracy of the predictions
        train_acc = train_accuracy[-1]
        test_acc = get_accuracy(y_test, y_test_hat)
        # append to accuracies array and repeat
        print(train_acc, test_acc)
        accuracies.append([train_acc, test_acc])
    return np.mean(accuracies[0][:]), np.mean(accuracies[1][:]), confusion_matrix, model,




def housing_part_a():
    X_train, X_test, y_train, y_test = housing_data_handling()

    ## Finding optimal values - lambda 0.72, learning rate 0.001, epochs 3000
    # lambdas = np.arange(0.65, 0.75, 0.01)
    # epochs = np.arange(400, 1200, 100)
    # learning_rates = np.arange(0.0004, 0.0025, 0.0004)
    # best_combo = []
    # for i in lambdas:
    #     for j in epochs:
    #         for k in learning_rates:
    #             ridge_model = Ridge_regression(learning_rate=k, lambda_val=i, epochs=j)
    #             train_cost, train_mse = ridge_model.fit(X_train, y_train, "housing")
    #             best_string = f"lambda = {i} epochs = {j} lr = {k}"
    #             print(best_string)
    #             print(f"Cost: {train_cost[-1]} MSE: {train_mse[-1]}")
    #             y_test_pred = ridge_model.predict(X_test)
    #             test_mse = mean_squared_error(y_test, y_test_pred)
    #             print(f"Test MSE: {test_mse}")
    #             if len(best_combo) == 0:
    #                 best_combo.append([best_string, train_mse[-1], 100.0])
    #             if test_mse < best_combo[0][2] and train_mse[-1] < test_mse:
    #                 best_combo[0] = [best_string, train_mse[-1], test_mse]
    #             print(best_combo[0])
    ridge_model = Ridge_regression(learning_rate=0.001, lambda_val=0.72, epochs=3000)
    train_cost, train_mse = ridge_model.fit(X_train, y_train, "housing")
    print(f"Cost: {train_cost[-1]} MSE: {train_mse[-1]}")
    y_test_pred = ridge_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE: {test_mse}")

def spambase_part_a():
    X, y = spambase_data_handling()
    X = add_bias(X)
    ## For optimal lambda (0.58)
    train_acc, test_acc, confusion_matrix, ridge_model = kfolds(X, y)
    print(f"Train Accuracy: {train_acc} Test Accuracy: {test_acc}")
    ## Finding optimal values - lambda 0.72, learning rate 0.002, epochs 700
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    # disp.plot(values_format=".0f")
    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.title("Ridge Regression Confusion Matrix")
    # plt.show()
    return ridge_model


def spambase_part_b():
    X, y = spambase_data_handling()
    X = add_bias(X)
  # Finding optimal values - learning rate 0.0015, epochs 2600
    train_acc, test_acc, confusion_matrix, logistic_model = kfolds(X, y, "logistic")
    print(f"Train Accuracy: {train_acc} Test Accuracy: {test_acc}")
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    # disp.plot(values_format=".0f")
    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.title("Logistic Regression Confusion Matrix")
    # plt.show()
def calculate_rates(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    return tpr, fpr

def plot_roc_curve(y_true, y_proba, model, color):
    thresholds = np.linspace(0, 1, 101)

    tprs = []
    fprs = []

    for threshold in thresholds:
        tpr, fpr = calculate_rates(y_true, y_proba, threshold)
        tprs.append(tpr)
        fprs.append(fpr)
    
    tprs, fprs = np.array(tprs), np.array(fprs)

    sorted_indices = np.argsort(fprs)
    tprs_sorted = tprs[sorted_indices]
    fprs_sorted = fprs[sorted_indices]
    
    auc = integrate.trapezoid(tprs_sorted, fprs_sorted)
    plt.plot(fprs_sorted, tprs_sorted, color=color, label=f'{model} (AUC = {auc:.3f})')
    return auc
def spambase_roc():
    "Trains Ridge and Logistic Regression Models and Plots ROC Curves to compare their AUC"
    X, y = spambase_data_handling()
    X = add_bias(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    ridge_model = Ridge_regression(learning_rate=0.001, lambda_val=0.58, epochs=1000)
    logistic_model = Logistic_regression(learning_rate=0.0015, epochs=2600)
    ridge_model.fit(X_train, y_train, "spambase")
    logistic_model.fit(X_train, y_train)
    ridge_proba = ridge_model.predict(X_test)
    logistic_proba = logistic_model.predict(X_test)
    
    auc_logistic = plot_roc_curve(y_test, logistic_proba, 'Logistic Regression', 'green')
    auc_ridge = plot_roc_curve(y_test, ridge_proba, 'Ridge Regression', 'red')

    # Random classifier
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
    print(f"Logistic Regression AUC: {auc_logistic:.3f}")
    print(f"Ridge Regression AUC: {auc_ridge:.3f}")
if __name__=="__main__":
    # housing_part_a()
    # spambase_part_a()
    # spambase_part_b()
    spambase_roc()
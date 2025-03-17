import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# SVM classification class
class SVM_Classification:
    def __init__(self) -> None:
        self.model = None

    def dataset_read(self, dataset_path):
        """
        YOUR CODE HERE
        Task: Read the dataset from the JSON file and split it into features (X) and target (y).

        :param dataset_path: The file path to the dataset in JSON format.
        :return: Features (X) and target variable (y).
        """
        pass

    def preprocess(self, X, y):
        """
        YOUR CODE HERE
        Task: Handle missing values and standardize the features using StandardScaler.

        :param X: Features (input variables).
        :param y: Target (output variable).
        :return: Preprocessed features (X) and target (y).
        """
        pass

    def train_classification_model(self, X_train, y_train):
        """
        YOUR CODE HERE
        Task: Initialize an SVC model and train it on the training data.

        :param X_train: Training set features.
        :param y_train: Training set labels.
        """
        pass

    def predict_accuracy(self, X_test, y_test):
        # Predict the target values using the test data
        y_pred = self.model.predict(X_test)
        
        # Calculate and return the accuracy score between true values and predicted values
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


# SVM regression class
class SVM_Regression:
    def __init__(self) -> None:
        self.model = None

    def dataset_read(self, dataset_path):
        """
        YOUR CODE HERE
        Task: Read the dataset from the JSON file and split it into features (X) and target (y).

        :param dataset_path: The file path to the dataset in JSON format.
        :return: Features (X) and target variable (y).
        """
        pass

    def preprocess(self, X, y):
        """
        YOUR CODE HERE
        Task: Handle missing values and standardize the features using StandardScaler.

        :param X: Features (input variables).
        :param y: Target (output variable).
        :return: Preprocessed features (X) and target (y).
        """
        pass

    def train_regression_model(self, X_train, y_train):
        """
        YOUR CODE HERE
        Task: Initialize an SVR model and train it on the training data.

        :param X_train: Training set features.
        :param y_train: Training set target values.
        """
        pass

    def predict_accuracy(self, X_test, y_test):
        # Predict the target values using the test data
        y_pred = self.model.predict(X_test)
        
        # Calculate mean absolute percentage error (MAPE) and subtract from 1 to get accuracy
        err = mean_absolute_percentage_error(y_test, y_pred)
        return 1 - err


    def visualize(self, X_test, y_test, y_pred):
        """
        Provided for students.
        This function visualizes the comparison between actual and predicted target values.
        Use this to see how your model is performing on the test set.

        :param X_test: Test set features.
        :param y_test: Actual target values.
        :param y_pred: Predicted target values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', alpha=0.6, edgecolor='k', label='Actual Target')
        plt.scatter(X_test, y_pred, color='red', alpha=0.6, edgecolor='k', label='Predicted Target')
        plt.title('X vs Target')
        plt.xlabel('X')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        plt.show()

# SVM Spiral Classification class
class SVM_Spiral:
    def __init__(self) -> None:
        self.model = None

    def dataset_read(self, dataset_path):
        """
        YOUR CODE HERE
        Task: Read the dataset from a JSON file and split it into features (X) and target (y).

        :param dataset_path: The file path to the dataset in JSON format.
        :return: Features (X) and target variable (y).
        """
        pass

    def preprocess(self, X, y):
        """
        YOUR CODE HERE
        Task: Handle missing values and standardize the features using StandardScaler.

        :param X: Features (input variables).
        :param y: Target (output variable).
        :return: Preprocessed features (X) and target (y).
        """
        pass

    def train_spiral_model(self, X_train, y_train):
        """
        YOUR CODE HERE
        Task: Initialize an SVC model with a suitable kernel, and train it on the training data.

        :param X_train: Training set features.
        :param y_train: Training set labels.
        """
        pass

    def predict_accuracy(self, X_test, y_test):
        # Predict the target values using the test data
        y_pred = self.model.predict(X_test)
        
        # Calculate and return the accuracy score between true values and predicted values
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# SVM classification class
class SVM_Classification:
    def __init__(self) -> None:
        self.model = SVC()

    def dataset_read(self, dataset_path):
        data = pd.read_json(dataset_path)
        X = data.iloc[:, :-1]  # All columns except the last are features
        y = data.iloc[:, -1]   # The last column is the target
        return X, y

    def preprocess(self, X, y):
        X.fillna(X.mean(), inplace=True)  # Fill missing values with mean
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features
        return X_scaled, y

    def train_classification_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_accuracy(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


# SVM regression class
class SVM_Regression:
    def __init__(self) -> None:
        self.model = SVR()

    def dataset_read(self, dataset_path):
        data = pd.read_json(dataset_path)
        X = data.iloc[:, :-1]  # All columns except the last are features
        y = data.iloc[:, -1]   # The last column is the target
        return X, y

    def preprocess(self, X, y):
        X.fillna(X.mean(), inplace=True)  # Fill missing values with mean
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features
        return X_scaled, y

    def train_regression_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_accuracy(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        err = mean_absolute_percentage_error(y_test, y_pred)
        return 1 - err

    def visualize(self, X_test, y_test, y_pred):
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
        self.model = SVC(kernel='rbf')  # RBF kernel is suitable for spiral data

    def dataset_read(self, dataset_path):
        data = pd.read_json(dataset_path)
        X = data.iloc[:, :-1]  # All columns except the last are features
        y = data.iloc[:, -1]   # The last column is the target
        return X, y

    def preprocess(self, X, y):
        X.fillna(X.mean(), inplace=True)  # Fill missing values with mean
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Standardize features
        return X_scaled, y

    def train_spiral_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_accuracy(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

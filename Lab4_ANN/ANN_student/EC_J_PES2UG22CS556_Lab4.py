import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and preprocess the data
# input: filepath: str (path to the CSV file)
# output: tuple of X (features), y (target)
def load_and_preprocess_data(filepath):
    # TODO: Implement this function
    df = pd.read_csv(filepath)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    return X,y

# Split the data into training and testing sets
# input: 1) X: ndarray (features)
#        2) y: ndarray (target)
# output: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    # TODO: Implement this function
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    return X_train,X_test,y_train,y_test

# Create and train 2 MLP classifiers with different parameters
# input:  1) X_train: ndarray
#         2) y_train: ndarray
# output: tuple of models (model1, model2)
def create_model(X_train, y_train):
    # TODO: Implement this function
    model1 = MLPClassifier(hidden_layer_sizes=(100,45,90),max_iter=30,activation='tanh',random_state=42)
    model1.fit(X_train,y_train)
    model2 = MLPClassifier(hidden_layer_sizes=(150,10,65),max_iter=300,activation='logistic',random_state=42)
    model2.fit(X_train,y_train)
    return model1,model2
    

# Predict and evaluate the model
# input: 1) model: MLPClassifier after training
#        2) X_test: ndarray
#        3) y_test: ndarray
# output: tuple - accuracy, precision, recall, fscore, confusion_matrix
def predict_and_evaluate(model, X_test, y_test):
    # TODO: Implement this function
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precison = precision_score(y_test,y_pred,average='weighted')
    recall = recall_score(y_test,y_pred,average='weighted')
    fscore = f1_score(y_test,y_pred,average='weighted')
    conf_matrix = confusion_matrix(y_test,y_pred)
    return accuracy,precison,recall,fscore,conf_matrix

   

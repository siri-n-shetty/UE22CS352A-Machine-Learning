import sys
import importlib
import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', required=True, help="Specify the module name to be tested.")
    return parser.parse_args()

def load_module(subname):
    try:
        print("module:",subname)
        module = importlib.import_module(subname)
        return module
    except Exception as e:
        print(e)
        print("Rename your written program as per specified format and run 'python3 test.py --ID file_name'")
        sys.exit(1)

def preprocess_and_split_data(model, data_path):
    
    # Read and preprocess the data using the model
    X, y = model.dataset_read(data_path)
        
    X, y = model.preprocess(X, y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_test(model, X_test, y_test, accuracy_range, task_name):
    try:
        accuracy = model.predict_accuracy(X_test, y_test)
        # print(f"Accuracy for {task_name}: {accuracy}")
        if accuracy_range[0] <= accuracy <= accuracy_range[1]:
            print(f"Test Case 1 for the function {task_name} PASSED")
        else:
            print(f"Test Case 1 for the function {task_name} FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"Test Case 1 for the function {task_name} FAILED [ERROR: {e}]")

def main():
    args = parse_arguments()
    subname = args.ID
    
    # Load the module containing SVM classes
    module = load_module(subname)

    classification_path = './cyyoung.json'
    regression_path = './flalc.json'
    spiral_path = './spiral.json'

    # Dynamically load SVM classes from the module
    SVM_Classification = getattr(module, 'SVM_Classification')
    SVM_Regression = getattr(module, 'SVM_Regression')
    SVM_Spiral = getattr(module, 'SVM_Spiral')

    # Initialize models
    classification_model = SVM_Classification()
    regression_model = SVM_Regression()
    spiral_model = SVM_Spiral()

    # Preprocess and split datasets
    X_train1, X_test1, y_train1, y_test1 = preprocess_and_split_data(classification_model, classification_path)
    X_train2, X_test2, y_train2, y_test2 = preprocess_and_split_data(regression_model, regression_path)
    X_train3, X_test3, y_train3, y_test3 = preprocess_and_split_data(spiral_model, spiral_path)

    # Train models
    classification_model.train_classification_model(X_train1, y_train1)
    regression_model.train_regression_model(X_train2, y_train2)
    spiral_model.train_spiral_model(X_train3, y_train3)

    # Run tests
    run_test(classification_model, X_test1, y_test1, accuracy_range=(0.83, 1.0), task_name="Classification")
    run_test(regression_model, X_test2, y_test2, accuracy_range=(0.91,1.0), task_name="Regression")
    run_test(spiral_model, X_test3, y_test3, accuracy_range=(0.8, 1.0), task_name="Spiral")

if __name__ == "__main__":
    main()


    # classification_path = './cyyoung.json'
    # regression_path = './flalc.json'
    # spiral_path = './spiral.json'

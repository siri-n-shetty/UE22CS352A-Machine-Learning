import sys
import importlib
import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)

args = parser.parse_args()
subname = args.ID

try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print("Rename your program as CAMPUS_SECTION_SRN_Lab.py and run python Test.py --ID CAMPUS_SECTION_SRN_Lab")
    sys.exit()

prepareData = mymodule.prepareData
CNN = mymodule.CNN
train = mymodule.train
evaluate = mymodule.evaluate

def test_case():
    # Load data
    path='data/'
    train_loader, test_loader = prepareData(path)
    
    # Initialize model
    model = CNN()

    # Train the model
    train_accuracy = train(model, train_loader)

    # Evaluate the model
    test_accuracy = evaluate(model, test_loader)

    try: 
        if train_accuracy >=80:
            print("Test Case 1 for train function PASSED")
        else:
            print("Test Case 1 for train function FAILED")
    except Exception as e:
        print(e)
        print("Test Case 1 for train function FAILED [ERROR]")

    try: 
        if test_accuracy >= 85:
            print("Test Case 2 for evaluate function PASSED")
        else:
            print("Test Case 2 for evaluate function FAILED")
    except Exception as e:
        print(e)
        print("Test Case 2 for evaluate function FAILED [ERROR]")

    

if __name__ == "__main__":
    test_case()

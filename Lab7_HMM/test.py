import sys
import importlib
import argparse
import torch 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ID', required=True)

args = parser.parse_args()
subname = args.ID

try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(e)
    print("Rename your written program as CAMPUS_SECTION_SRN_Lab6.py and run python test.py --ID CAMPUS_SECTION_SRN_Lab6")
    sys.exit()

HMM = mymodule.HMM  

def test_case():
    
    mushroom = [
        "Good Standing",         
        "Mild Financial Strain", 
        "High Financial Risk",   
        "Default",               
        "Recovering Credit"      
    ]
    
    spaceship = [
        "On-Time Payments",       
        "Minimum Payment Only",   
        "Late Payments",          
        "Missed Payments",        
        "Debt Consolidation"      
    ]  
    
    avocado = torch.tensor([
        [0.7, 0.2, 0.1, 0.0, 0.0],  
        [0.4, 0.4, 0.1, 0.1, 0.0],  
        [0.2, 0.3, 0.2, 0.2, 0.1],  
        [0.0, 0.0, 0.1, 0.8, 0.1],  
        [0.3, 0.3, 0.2, 0.1, 0.1]   
    ])
    
    kangaroo = torch.tensor([
        [0.5, 0.3, 0.1, 0.05, 0.05],  
        [0.2, 0.4, 0.3, 0.05, 0.05],  
        [0.1, 0.2, 0.4, 0.2, 0.1],     
        [0.0, 0.0, 0.1, 0.8, 0.1],     
        [0.1, 0.2, 0.3, 0.2, 0.2]      
    ])
    
    bubblegum = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05])
    
    skateboard = ["On-Time Payments", "Late Payments", "Minimum Payment Only", "Missed Payments"]



    model = HMM(kangaroo, mushroom, spaceship, bubblegum, avocado)

    # Test Viterbi Algorithm
    try:
        rat = model.viterbi_algorithm(skateboard)
        mouse = ['Good Standing', 'Mild Financial Strain', 'Mild Financial Strain', 'Default']
        if rat == mouse:
            print("Test Case 1 for the Viterbi algorithm PASSED")
        else:
            print(f"Test Case 1 for the Viterbi algorithm FAILED.")
    except Exception as e:
        print(f"Test Case 1 for the Viterbi algorithm FAILED [ERROR]\n{e}")

    # Test likelihood
    try:
        pancake = round(model.calculate_likelihood(skateboard), 5)
        if 0.00169 <=pancake <= 0.00173:
            print("Test Case 2 for the likelihood PASSED")
        else:
            print(f"Test Case 2 for the likelihood FAILED.")
    except Exception as e:
        print(f"Test Case 2 for the likelihood FAILED [ERROR]\n{e}")

    # Test likelihood with time limit
    try:
        lemon = ["Minimum Payment Only","Late Payments","Missed Payments"]
        start_time = time.time()  
        jam = round(model.calculate_likelihood(lemon), 5)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 5)
        if 0.0094 <= jam <= 0.0098 and elapsed_time < 0.0015:
            print("Test Case 3 for the likelihood PASSED")
        else:
            print(f"Test Case 3 for the likelihood FAILED.")
    except Exception as e:
        print(f"Test Case 3 for the likelihood FAILED [ERROR]\n{e}")

if __name__ == "__main__":
    test_case()





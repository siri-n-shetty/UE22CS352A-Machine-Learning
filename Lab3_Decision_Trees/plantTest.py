import sys
import importlib
import argparse
import torch
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)
parser.add_argument("--data", default=None, required=True)

args = parser.parse_args()
subname = args.ID
data_path = args.data

try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print("Rename your written program as  CAMPUS_SECTION_SRN_Lab3.py and run python plantTest.py --ID CAMPUS_SECTION_SRN_Lab3 --data plantData.csv")
    sys.exit()


get_selected_attribute = mymodule.get_selected_attribute
get_information_gain = mymodule.get_information_gain
get_avg_info_of_attribute = mymodule.get_avg_info_of_attribute
get_entropy_of_dataset = mymodule.get_entropy_of_dataset

def print_node_info(node_info, level):
    print(f"{'|  ' * level}Level {level}: Node Info - {node_info}")

def construct_tree(tensor, cols, used_attributes=set(), level=0):
    if len(tensor) == 0:
        return None
    
    # Print the entropy of the current dataset
    entropy = get_entropy_of_dataset(tensor)
    print_node_info(f"Entropy = {entropy:.4f}", level)
    
    # Get the selected attribute and print the information gain for all attributes
    gain_dict, selected_attribute = get_selected_attribute(tensor)
    
    # Filter out already used attributes
    gain_dict = {attr: gain for attr, gain in gain_dict.items() if attr not in used_attributes}
    
    # Stop if no more information gain or if entropy is 0
    if entropy == 0 or not gain_dict or max(gain_dict.values()) == 0:
        label_column = [t[-1].item() for t in tensor]
        majority_class = max(set(label_column), key=label_column.count)
        print_node_info(f"Hypothesis: {cols[int(majority_class)]}", level)
        return majority_class
    
    # Select the attribute with the highest information gain
    selected_attribute = max(gain_dict, key=gain_dict.get)
    
    print_node_info(f"Information Gain: {gain_dict}, Selected Attribute: {cols[selected_attribute]}", level)
    
    unique_values = torch.unique(tensor[:, selected_attribute])
    used_attributes.add(selected_attribute)  # Mark this attribute as used
    
    for value in unique_values:
        subset_tensor = tensor[tensor[:, selected_attribute] == value]
        if len(subset_tensor) == len(tensor):
            # If the subset has the same number of rows as the original, no further split is possible
            continue
        print_node_info(f"Branch {cols[int(selected_attribute)]} = {value.item()}", level)
        construct_tree(subset_tensor, cols, used_attributes.copy(), level + 1)


def test_case():
    df = pd.read_csv(data_path)
    # data = np.array([torch.tensor([row[col] for col in df.columns]) for index, row in df.iterrows()])
    data = df.to_numpy(dtype=np.float32)
    dataset, cols = torch.from_numpy(data), list(df.columns)

    try:
        entropy = get_entropy_of_dataset(dataset)
        if entropy >= 0.999 and entropy <= 1:
            print("Test Case 1 for the function get_entropy_of_dataset PASSED")
        else:
            print("Test Case 1 for the function get_entropy_of_dataset FAILED")
    except:
        print("Test Case 1 for the function get_entropy_of_dataset FAILED [ERROR]")
    
    try:
        avg_info_0 = get_avg_info_of_attribute(dataset, 0)
        if ( avg_info_0 >=0.992 and avg_info_0 <= 0.994 ):
            print("Test Case 2 for the function get_avg_info_of_attribute PASSED")
        else:
            print("Test Case 2 for the function get_avg_info_of_attribute FAILED")
    except:
        print("Test Case 2 for the function get_avg_info_of_attribute FAILED [ERROR]")
    
    try:
        avg_info_1 = get_avg_info_of_attribute(dataset, 1)
        if ( avg_info_1 >= 0.975 and avg_info_1 <= 0.99 ):
            print("Test Case 3 for the function get_avg_info_of_attribute PASSED")
        else:
            print("Test Case 3 for the function get_avg_info_of_attribute FAILED")

    except:
        print("Test Case 3 for the function get_avg_info_of_attribute FAILED [ERROR]")

    try:
        ans = get_selected_attribute(dataset)
        dictionary = ans[0]
        flag = (
            (dictionary[0] >= 0.0067 and dictionary[0] <= 0.0069)
            and (dictionary[1] >= 0.0182 and dictionary[1] <= 0.0192)
            and (dictionary[2] >= 0.0004 and dictionary[2] <= 0.0006)
            and (dictionary[3] >= 0.0837 and dictionary[3] <= 0.0845)
            and (ans[1] == 3)
        )
        if flag:
            print("Test Case 4 for the function get_selected_attribute PASSED")
        else:
            print("Test Case 4 for the function get_selected_attribute FAILED")
    except:
        print("Test Case 4 for the function get_selected_attribute FAILED [ERROR]")

    # Recursive Decision Tree Construction
    try:
        print("\n--- Constructing Decision Tree ---\n")
        tree = construct_tree(dataset, cols=cols)
    except Exception as e:
        print(e)
        print("Failed to construct the decision tree")

if __name__ == "__main__":
    test_case()
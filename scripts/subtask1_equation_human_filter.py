import json
import os

# ori_data_path = "./subtask1_equation_unified/equation_gpt-gen-wrong-eq_gpt-filtered_1449.json"
idx = 2
ori_data_path = f"./subtask1_equation_unified/human_filter_seperate/1449_{idx}.human_filter.json"

with open(ori_data_path, "r") as f:
    ori_data = json.load(f)
    
    
target_data_path = f"./subtask1_equation_unified/human_filter_seperate/1449_{idx}.human_filter.done.json"

# read the existing annotated data
if os.path.exists(target_data_path):
    with open(target_data_path, "r") as f:
        target_data = json.load(f)
else:
    target_data = []

existing_data_len = len(target_data)

for i, data in enumerate(ori_data):
    all_options = data["options_list"]
    answer = data["answer"]
    # if this data has been annotated, skip it
    if i < existing_data_len:
        continue
    # "A" ==> 0, "B" ==> 1, "C" ==> 2, "D" ==> 3
    answer_idx = ord(answer) - ord("A")
    ground_truth = all_options[answer_idx]
    other_options = [option for idx, option in enumerate(all_options) if idx != answer_idx]
    print("\n\n" + "="*50)
    print(f"This is the {i} th data\n")
    print("="*20 + " Target " + "="*20)
    print("(A) : ", ground_truth)
    print()
    print("="*20 + " Others " + "="*20)
    for ii, wr_option in enumerate(other_options):
        # print (B), (C), (D)
        # print("(" + (ii+1) + ")" + f": {wr_option}")
        print(f"({chr(ord('A') + ii + 1)}) : {wr_option}")
    print()
    # recieve the user input
    print("="*20 + " User Input " + "="*20)
    user_input = input("Is there any other options the same as the Target equation? (1: has same; 0: no same): ")
    if user_input == "1":
        data["keep"] = False
    else:
        data["keep"] = True
        
    target_data.append(data)
    # save the data every 10 iterations
    # if i % 10 == 0:
    with open(target_data_path, "w") as f:
        json.dump(target_data, f, indent=2)

with open(target_data_path, "w") as f:
    json.dump(target_data, f, indent=2)
    
print("===> save data at ", target_data_path)
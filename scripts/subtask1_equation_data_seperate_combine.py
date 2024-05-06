import json
import os

# ====== seperate data ======
# ori_data_path = "./subtask1_equation_unified/equation_gpt-gen-wrong-eq_gpt-filtered_1449.json"
# seperate_num = 3
# target_path = "./subtask1_equation_unified/human_filter_seperate"

# os.makedirs(target_path, exist_ok=True)

# with open(ori_data_path, "r") as f:
#     ori_data = json.load(f)

# total_num = len(ori_data)
# target_len_list = []
# seperate_len = total_num // seperate_num
# for i in range(seperate_num):
#     target_data_path = os.path.join(target_path, f"1449_{i}.human_filter.json")
#     start_idx = i * seperate_len
#     end_idx = (i + 1) * seperate_len
#     if i == seperate_num - 1:
#         end_idx = total_num
#     target_data = ori_data[start_idx:end_idx]
#     with open(target_data_path, "w") as f:
#         json.dump(target_data, f, indent=4)
#     target_len_list.append(len(target_data))
        
# print("===> save data at ", target_path)
# for i, l in enumerate(target_len_list):
#     print(f" {i} ===> data length: ", l)


# ====== combine annotated data ======
data_path = "./subtask1_equation_unified/human_filter_seperate"
seperate_num = 3

all_data = []
for i in range(seperate_num):
    data_file = f"1449_{i}.human_filter.done.json"
    data_file_path = os.path.join(data_path, data_file)
    with open(data_file_path, "r") as f:
        data = json.load(f)
    all_data.extend(data)
    
print("==> totoal data length: ", len(all_data))

filtered_data = []
for i, d in enumerate(all_data):
    keep_sig = d.pop("keep")
    if keep_sig:
        filtered_data.append(d)

print("==> filtered data length: ", len(filtered_data))
print("==> percentage: ", len(filtered_data) / len(all_data))

target_data = f"./subtask1_equation_unified/{len(filtered_data)}.human_filter.json"

with open(target_data, "w") as f:
    json.dump(filtered_data, f, indent=2)

print("===> save data at ", target_data)  
# just read over the score results of all the papers, get the avg score

import json
import os
import sys
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='./', help='Path to the source directory')
args = parser.parse_args()

# source_paths = ["claude-3-5-sonnet-20240620-3500-oracle---202410022110" , "gemini-1.5-pro-3000-oracle---202409211345", "gpt-4o-3000-oracle---202409291738",
#                 "gpt-4-3000-oracle---202409162134", "o1-preview-3000-oracle---202409162140", "Qwen_Qwen2.5-72B-Instruct-2000-oracle---202501131422", 
#                 "meta-llama_Meta-Llama-3.1-70B-Instruct-2000-oracle---202501131252",
#                 "mistralai_Mistral-7B-Instruct-v0.3-1000-oracle---202501141124",
#                 "mistralai_Mixtral-8x22B-Instruct-v0.1-2000-oracle---202501141053",
#                 "google_gemma-2-27b-1000-oracle---202501141131",
#                 "tiiuae_falcon-40b-1000-oracle---202501141303",
#                 "allenai_OLMo-7B-0724-Instruct-hf-1000-oracle---202501141254"
#                 ]
# get all the subfolder name under ./ as the source paths
root_dir = args.root_dir
# source_paths = [f.path for f in os.scandir("./") if f.is_dir()]
source_paths = [f.path for f in os.scandir(root_dir) if f.is_dir()]
# get the subfolder name instead of the full path
source_paths = [f.split("/")[-1] for f in source_paths]
# sort name
source_paths.sort()

exclude_ids = []  # "2303.17651"

for source_dir in source_paths:
    print("\n\n==> for model: ", source_dir)
    # get all the subfolder name under source dir,
    subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
    # and exlucde those dirs that wihout  a "eval_results.json" file under this dir 
    subfolders = [f for f in subfolders if os.path.exists(f+"/eval_results.json")]
    print(f"===> totally {len(subfolders)} subfolders")
    if len(subfolders) == 0:
        continue
    # exclude the ids 
    # subfolders = [f for f in subfolders if f.split("/")[-1] not in exclude_ids]

    all_recall, all_precision = [], []
    all_f1 = []
    novel_exp_cnt = 0
    for paper_folder in subfolders:
        paper_id = paper_folder.split("/")[-1]
        if paper_id in exclude_ids:
            continue
        with open(os.path.join(paper_folder, "eval_results.json"), "r") as f:
            data = json.load(f)
            if len(data["prediction"]) > 0:
                recall = data["recall_gt_entail_score"]
                precision = data["precision_pred_entail_score"]
                all_precision.append(precision)
                all_recall.append(recall)
                f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
                all_f1.append(f1)   
                
                novel_exp_cnt += len(data["novel_exps"])
            else:
                print(f"!!! paper {paper_folder} has no prediction")
    # avg_recall = sum(all_recall) / len(all_recall)
    # avg_precision = sum(all_precision) / len(all_precision)
    # avg_f1 = sum(all_f1) / len(all_f1)
    avg_recall = sum(all_recall) / len(subfolders)
    avg_precision = sum(all_precision) / len(subfolders)
    avg_f1 = sum(all_f1) / len(subfolders)
    print(f"==> totally {len(all_recall)} papers has valid prediciton")
    print(f"==> avg_recall: {avg_recall}, avg_precision: {avg_precision}, avg_f1: {avg_f1}")
    print(f"==> novel_exp_cnt: {novel_exp_cnt}")

 
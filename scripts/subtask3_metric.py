'''
calculate the performance and statistics of the model predictions
'''
import copy
import random
import re
import dataclasses
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/rml6079/.cache/huggingface"
os.environ["HF_HOME"] = "/scratch/rml6079/.cache/huggingface"

import torch
from sentence_transformers import SentenceTransformer, util
from calculate_metrics_src import soft_f1, SentenceSemanticMetric, soft_score, rouge_score, soft_accumulate

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
nltk.download('punkt_tab')


def word_num(input_text):
    words = word_tokenize(input_text)
    number_of_words = len(words)
    return number_of_words

def sentence_num(input_text):
    sentences = sent_tokenize(input_text)
    number_of_sentences = len(sentences)
    return number_of_sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./subtask3_review_final_v2/eval_results", help="the directory save the prediction results.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing performance.json")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # iterate all the subfolders and subsubfolders
    # get all the `eval_results.json` of different paths
    all_pred_files = []
    for root, dirs, files in os.walk(args.root_dir):
        for file in files:
            if file == "eval_results.json":
                pred_file = os.path.join(root, file)
                all_pred_files.append(pred_file)
    print("==> total found prediction files: ", len(all_pred_files))
    
    for pred_file in all_pred_files:
        folder_name = os.path.dirname(pred_file)
        save_file = os.path.join(folder_name, "performance.json")
        if os.path.exists(save_file) and not args.overwrite:
            print(f"==> {save_file} already exists, skip")
            continue
        print(f"==> processing {folder_name}")
        with open(pred_file, "r") as f:
            pred_data = json.load(f)
        
        f1_list, precision_list, recall_list = [], [], []
        review_num_list, item_len_list_list = [], []
        pred_len_list = []
        empty_num = 0
        pred_word_len_list, pred_sent_len_list = [], []
        gt_word_len_list, gt_sent_len_list = [], []
        for item in tqdm(pred_data):
            pred = item["predicton"]  # a list of strings
            ref = item["output"] # a list of list of strings
            f1, precision, recall = soft_accumulate(pred, ref, model)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            
            num_reviewer = len(ref)
            item_len_list = [len(item) for item in ref]
            review_num_list.append(num_reviewer)
            item_len_list_list.append(item_len_list)
            
            pred_len_list.append(len(pred))
            if len(pred) == 0:
                empty_num += 1
                
            # get the avg word length of each item in pred
            if len(pred) != 0:
                pred_avg_word_len = sum([word_num(pd) for pd in pred])/len(pred)
                pred_word_len_list.append(pred_avg_word_len)
                pred_avg_sent_len = sum([sentence_num(pd) for pd in pred])/len(pred)
                pred_sent_len_list.append(pred_avg_sent_len)
            
            # combine the nested list into a flat list
            falt_ref = [item for sublist in ref for item in sublist]
            if len(falt_ref) != 0:
                gt_avg_word_len = sum([word_num(gt) for gt in falt_ref])/len(falt_ref)
                gt_word_len_list.append(gt_avg_word_len)
                gt_avg_sent_len = sum([sentence_num(gt) for gt in falt_ref])/len(falt_ref)
                gt_sent_len_list.append(gt_avg_sent_len)
        
        # save the result back to the dir
        with open(save_file, "w") as f:
            json.dump({
                "f1": sum(f1_list)/len(f1_list),
                "precision": sum(precision_list)/len(precision_list),
                "recall": sum(recall_list)/len(recall_list),
                "empty_num": empty_num,
                "pred_len": {
                    "avg": sum(pred_len_list)/len(pred_len_list),
                    "max": max(pred_len_list),
                    "min": min(pred_len_list)
                },
                "review_num": {
                    "avg": sum(review_num_list)/len(review_num_list),
                    "max": max(review_num_list),
                    "min": min(review_num_list)
                },
                "pred_word_len": {
                    "avg": sum(pred_word_len_list)/len(pred_word_len_list),
                    "max": max(pred_word_len_list),
                    "min": min(pred_word_len_list)
                },
                "pred_sent_len": {
                    "avg": sum(pred_sent_len_list)/len(pred_sent_len_list),
                    "max": max(pred_sent_len_list),
                    "min": min(pred_sent_len_list)
                },
                "reference_word_len": {
                    "avg": sum(gt_word_len_list)/len(gt_word_len_list),
                    "max": max(gt_word_len_list),
                    "min": min(gt_word_len_list)
                },
                "reference_sent_len": {
                    "avg": sum(gt_sent_len_list)/len(gt_sent_len_list),
                    "max": max(gt_sent_len_list),
                    "min": min(gt_sent_len_list)
                },
                "weakness_list_len": item_len_list_list
            }, f, indent=4)
    
if __name__ == "__main__":
    main()

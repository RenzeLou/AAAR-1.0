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

import torch
from sentence_transformers import SentenceTransformer, util
from calculate_metrics_src import soft_f1, SentenceSemanticMetric

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
nltk.download('punkt_tab')


def word_num(input_text):
    words = word_tokenize(input_text)
    number_of_words = len(words)
    return number_of_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./subtask2_experiment_human_anno/eval_results/gpt-4o-1000-oracle---202409160206", help="the directory save the prediction results.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    pred_word_num_experiment, pred_word_num_explanation = [], []
    gt_word_num_experiment, gt_word_num_explanation = [], []
    experiment_f1_list, experiment_p_list, experiment_r_list = [], [], []
    explanation_f1_list, explanation_p_list, explanation_r_list = [], [], []
    gt_experiments_len, gt_explanations_len = [], []
    pred_experiments_len, pred_explanations_len = [], []
    pred_num = 0
    for paper_id in tqdm(os.listdir(args.root_dir)):
        pred_file = os.path.join(args.root_dir, paper_id, "eval_results.json")
        if not os.path.exists(pred_file):
            print(f"File not found: {pred_file}")
            continue
        pred_num += 1
        with open(pred_file, "r") as f:
            pred_data = json.load(f)
        gt_experiments = pred_data["output"]["What experiments do you suggest doing?"]
        gt_explanations = pred_data["output"]["Why do you suggest these experiments?"]
        pred_experiments = pred_data["predicton"]["What experiments do you suggest doing?"]
        pred_explanations = pred_data["predicton"]["Why do you suggest these experiments?"]
        oracle = pred_data["oracle"]
        
        if not oracle:
            raise NotImplementedError("Only support oracle for now.")
        else:
            # oracle setting, just compare the pred_explanations with gt_explanations, cuz we used the oracle to generate the explanations
            ## for experiment list
            experiment_f1, experiment_p, experiment_r = soft_f1(pred_experiments, gt_experiments, model)
            ## for explanation list
            explanation_f1, explanation_p, explanation_r = soft_f1(pred_explanations, gt_explanations, model)
            
            
        for item_experiment, item_explanation in zip(gt_experiments, gt_explanations):
            gt_word_num_experiment.append(word_num(item_experiment))
            gt_word_num_explanation.append(word_num(item_explanation))
        for item_experiment_pred in pred_experiments:
            pred_word_num_experiment.append(word_num(item_experiment_pred))
        for item_explanation_pred in pred_explanations:
            pred_word_num_explanation.append(word_num(item_explanation_pred))
        gt_experiments_len.append(len(gt_experiments))
        gt_explanations_len.append(len(gt_explanations))
        pred_experiments_len.append(len(pred_experiments))
        pred_explanations_len.append(len(pred_explanations))
        
        experiment_f1_list.append(experiment_f1)
        experiment_p_list.append(experiment_p)
        experiment_r_list.append(experiment_r)
        explanation_f1_list.append(explanation_f1)
        explanation_p_list.append(explanation_p)
        explanation_r_list.append(explanation_r)
        score = {
            "experiment": {
                "f1": experiment_f1,
                "precision": experiment_p,
                "recall": experiment_r
            },
            "explanation": {
                "f1": explanation_f1,
                "precision": explanation_p,
                "recall": explanation_r
            }
        }
        
        pred_data["score"] = score
        # save back the results
        with open(pred_file, "w") as f:
            json.dump(pred_data, f, indent=4)
    
    
    print("="*20)
    print(f"Total number of predictions: {pred_num}")
    print("="*20)
    print(f"Experiment metrics:")
    print(f"Average F1: {sum(experiment_f1_list)/len(experiment_f1_list):.4f}")
    print(f"Average Precision: {sum(experiment_p_list)/len(experiment_p_list):.4f}")
    print(f"Average Recall: {sum(experiment_r_list)/len(experiment_r_list):.4f}")
    print("="*20)
    print(f"Explanation metrics:")
    print(f"Average F1: {sum(explanation_f1_list)/len(explanation_f1_list):.4f}")
    print(f"Average Precision: {sum(explanation_p_list)/len(explanation_p_list):.4f}")
    print(f"Average Recall: {sum(explanation_r_list)/len(explanation_r_list):.4f}")
    print("="*20)
    print(f"Word number statistics:")
    print(f"GT experiment: AVG: {sum(gt_word_num_experiment)/len(gt_word_num_experiment)}, MAX: {max(gt_word_num_experiment)}, MIN: {min(gt_word_num_experiment)}")
    print(f"Pred experiment: AVG: {sum(pred_word_num_experiment)/len(pred_word_num_experiment)}, MAX: {max(pred_word_num_experiment)}, MIN: {min(pred_word_num_experiment)}")
    print(f"GT explanation: AVG: {sum(gt_word_num_explanation)/len(gt_word_num_explanation)}, MAX: {max(gt_word_num_explanation)}, MIN: {min(gt_word_num_explanation)}")
    print(f"Pred explanation: AVG: {sum(pred_word_num_explanation)/len(pred_word_num_explanation)}, MAX: {max(pred_word_num_explanation)}, MIN: {min(pred_word_num_explanation)}")
    print("="*20)
    print(f"List length statistics:")
    print(f"GT experiment: AVG: {sum(gt_experiments_len)/len(gt_experiments_len)}, MAX: {max(gt_experiments_len)}, MIN: {min(gt_experiments_len)}")
    print(f"Pred experiment: AVG: {sum(pred_experiments_len)/len(pred_experiments_len)}, MAX: {max(pred_experiments_len)}, MIN: {min(pred_experiments_len)}")
    print(f"GT explanation: AVG: {sum(gt_explanations_len)/len(gt_explanations_len)}, MAX: {max(gt_explanations_len)}, MIN: {min(gt_explanations_len)}")
    print(f"Pred explanation: AVG: {sum(pred_explanations_len)/len(pred_explanations_len)}, MAX: {max(pred_explanations_len)}, MIN: {min(pred_explanations_len)}")
    
    # save all the metrics
    with open(os.path.join(args.root_dir, "all_metrics.json"), "w") as f:
        json.dump({
            "experiment": {
                "f1": sum(experiment_f1_list)/len(experiment_f1_list),
                "precision": sum(experiment_p_list)/len(experiment_p_list),
                "recall": sum(experiment_r_list)/len(experiment_r_list)
            },
            "explanation": {
                "f1": sum(explanation_f1_list)/len(explanation_f1_list),
                "precision": sum(explanation_p_list)/len(explanation_p_list),
                "recall": sum(explanation_r_list)/len(explanation_r_list)
            },
            "word_num": {
                "gt_experiment": {
                    "avg": sum(gt_word_num_experiment)/len(gt_word_num_experiment),
                    "max": max(gt_word_num_experiment),
                    "min": min(gt_word_num_experiment)
                },
                "pred_experiment": {
                    "avg": sum(pred_word_num_experiment)/len(pred_word_num_experiment),
                    "max": max(pred_word_num_experiment),
                    "min": min(pred_word_num_experiment)
                },
                "gt_explanation": {
                    "avg": sum(gt_word_num_explanation)/len(gt_word_num_explanation),
                    "max": max(gt_word_num_explanation),
                    "min": min(gt_word_num_explanation)
                },
                "pred_explanation": {
                    "avg": sum(pred_word_num_explanation)/len(pred_word_num_explanation),
                    "max": max(pred_word_num_explanation),
                    "min": min(pred_word_num_explanation)
                }
            },
            "list_len": {
                "gt_experiment": {
                    "avg": sum(gt_experiments_len)/len(gt_experiments_len),
                    "max": max(gt_experiments_len),
                    "min": min(gt_experiments_len)
                },
                "pred_experiment": {
                    "avg": sum(pred_experiments_len)/len(pred_experiments_len),
                    "max": max(pred_experiments_len),
                    "min": min(pred_experiments_len)
                },
                "gt_explanation": {
                    "avg": sum(gt_explanations_len)/len(gt_explanations_len),
                    "max": max(gt_explanations_len),
                    "min": min(gt_explanations_len)
                },
                "pred_explanation": {
                    "avg": sum(pred_explanations_len)/len(pred_explanations_len),
                    "max": max(pred_explanations_len),
                    "min": min(pred_explanations_len)
                }
            }
        }, f, indent=4)
    
if __name__ == "__main__":
    main()

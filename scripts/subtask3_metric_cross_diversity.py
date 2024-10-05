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
from calculate_metrics_src import cross_focus_diversity

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
nltk.download('punkt_tab')


def get_track_id(data_dir):
    '''
    return a dict like:
    {
        "ML scientific paper": [id1, id2, ...],
        "LLM generated review": [id1, id2, ...],
    }
    and a dict like:
    
    {
        "id1": output list 1,
        "id2": output list 2,
    }
    '''
    # get all gthe subfoler under the data_dir
    all_data_files = []
    for file_dirs in os.listdir(data_dir):
        data_text_file = os.path.join(data_dir, file_dirs, "data_text.json")
        if os.path.isdir(os.path.join(data_dir, file_dirs)) and os.path.exists(data_text_file):
            all_data_files.append(data_text_file)
    print("==> total found data files: ", len(all_data_files))
    
    track2id = {}
    id2output = {}
    for data_file in all_data_files:
        with open(data_file, "r") as f:
            data = json.load(f)
        id = data['ID']
        track = data['track']
        out = data['output']

        if track not in track2id:
            track2id[track] = [id]
        else:
            track2id[track].append(id)
        
        id2output[id] = out
    
    print("==> total found tracks: ", len(track2id))
    print("==> total found ids: ", len(id2output))
    
    return track2id, id2output
            
    

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
    parser.add_argument("--data_dir", type=str, default="./subtask3_review_final_v2/ICLR_2023", help="the directory of the data") 
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing performance.json")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for the cross paper diversity, the similarity below this threshold will be considered as 0")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for the cross paper diversity")
    parser.add_argument("--papaer_top_k", type=int, default=2, help="top k items in a papaer's weakness list be saved")
    parser.add_argument("--track_top_k", type=int, default=20, help="top k items in a track's weakness list be saved")
    parser.add_argument("--only_human_score", action="store_true", help="only calculate the human idf score")
    parser.add_argument("--pick_choice", type=int, default=1, help="pick one weakness list from multi human reviewer. 1 for randomly pick; 2 for select the longest; 3 for combine all the reviewer's weakness list")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # iterate all the subfolders and subsubfolders
    # get all the `eval_results.json` of different paths
    all_pred_files = []
    for root, dirs, files in os.walk(args.root_dir):
        for file in files:
            if file == "eval_results.json" and "human_score" not in root:
                pred_file = os.path.join(root, file)
                all_pred_files.append(pred_file)
    print("==> total found prediction files: ", len(all_pred_files))
    
    track2id, id2gtoutput = get_track_id(args.data_dir)
    
    
    # human score calculation is separated
    if args.only_human_score:
        # only calculate the human written ground truth weakness score
        PICK = args.pick_choice # 1 for randomly pick; 2 for select the longest; 3 for combine all the reviewer's weakness list
        save_path = os.path.join(args.root_dir, "human_score")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"cross_diversity.{args.threshold}.{PICK}.json")

        # get cross paper diversity
        all_papers_weakness_list = []
        for id, gt_outputs in id2gtoutput.items():
            # 1. randomly pick one reviewer's weakness list, except the empty list
            if PICK == 1:
                gt_outputs = [x for x in gt_outputs if len(x) > 0]
                if len(gt_outputs) == 0:
                    raise ValueError("all the reviewer's weakness list is empty")
                gt_output = random.choice(gt_outputs)  # list of strings
            elif PICK == 2:
                # 2. select the longest review weakness list
                gt_output = max(gt_outputs, key=lambda x: len(x))
            elif PICK == 3:
                # 3. combine all the reviewer's weakness list
                gt_output = []
                for x in gt_outputs:
                    gt_output.extend(x)
            else:
                raise ValueError("PICK should be 1 or 2 or 3")
            all_papers_weakness_list.append((id, gt_output))
        cross_paper_itf_idf_score, cross_paper_id2scorelis = cross_focus_diversity(all_papers_weakness_list, model, threshold=args.threshold, inverse_tf=True, batch_size=args.batch_size)
        # also, for each papaer, save the top k items in the weakness list
        papaer_id2items_save = {}
        for id, weakness_list in all_papers_weakness_list:
            scores = cross_paper_id2scorelis[id]
            # try:
            assert len(scores) == len(weakness_list), "length of scores and weakness list should be the same, and one-by-one correspondence, but got {} and {}".format(len(scores), len(weakness_list))
            # except:
            #     import pdb; pdb.set_trace()
            #     wait = True
            # pick the top k items
            top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.papaer_top_k]
            top_k_items = [weakness_list[i] for i in top_k_idx]
            papaer_id2items_save[id] = top_k_items
        # get cross track diversity
        all_tracks_weakness_list = []
        for track_name, paper_ids in track2id.items():
            track_weakness_list = []
            for id in paper_ids:
                gt_outputs = id2gtoutput[id]
                if PICK == 1:
                    # 1. randomly pick one reviewer's weakness list, except the empty list
                    gt_outputs = [x for x in gt_outputs if len(x) > 0]
                    if len(gt_outputs) == 0:
                        raise ValueError("all the reviewer's weakness list is empty")
                    gt_output = random.choice(gt_outputs)  # list of strings
                elif PICK == 2:
                    # 2. select the longest review weakness list
                    gt_output = max(gt_outputs, key=lambda x: len(x))
                elif PICK == 3:
                    # 3. combine all the reviewer's weakness list
                    gt_output = []
                    for x in gt_outputs:
                        gt_output.extend(x)
                else:
                    raise ValueError("PICK should be 1 or 2 or 3")
                track_weakness_list.extend(gt_output)
            all_tracks_weakness_list.append((track_name, track_weakness_list))
            
        cross_track_tf_idf_score, cross_track_id2scorelis = cross_focus_diversity(all_tracks_weakness_list, model, threshold=args.threshold, inverse_tf=False, batch_size=args.batch_size)
        # also save the top k items in the weakness list, for each track
        track_id2items_save = {}
        for id, weakness_list in all_tracks_weakness_list:
            scores = cross_track_id2scorelis[id]
            assert len(scores) == len(weakness_list), "length of scores and weakness list should be the same, and one-by-one correspondence, but got {} and {}".format(len(scores), len(weakness_list))
            # pick the top k items
            top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.track_top_k]
            top_k_items = [weakness_list[i] for i in top_k_idx]
            track_id2items_save[id] = top_k_items
        
        print("*"*50)
        print("==> for human written ground truth weakness score")
        print("==> cross papaer diversity (ITF-IDF): ", round(cross_paper_itf_idf_score, 4))
        print("==> cross track diversity (TF-IDF): ", round(cross_track_tf_idf_score, 4))
        print("==> save the result to: ", save_file)
        with open(save_file, "w") as f:
            json.dump({"cross_paper_itf_idf_score": cross_paper_itf_idf_score,
                    "cross_track_tf_idf_score": cross_track_tf_idf_score,
                    "track_id2items_save": track_id2items_save,
                    "papaer_id2items_save": papaer_id2items_save,
                    }, f, indent=4)
            
    
    # the below is for all model's prediction score calculation       
    else:
        # for each model's prediction
        for pred_file in all_pred_files:
            folder_name = os.path.dirname(pred_file)
            save_file = os.path.join(folder_name, f"cross_diversity.{args.threshold}.json")
            if os.path.exists(save_file) and not args.overwrite:
                print(f"==> {save_file} already exists, skip")
                continue
            print(f"==> processing {folder_name}")
            with open(pred_file, "r") as f:
                pred_data = json.load(f)
            
            # get all papers' weakness list
            all_papers_weakness_list = []
            for pred_item in pred_data:
                id = pred_item["id"]
                pred_weakness_list = pred_item["predicton"]
                if len(pred_weakness_list) == 0:
                    pred_weakness_list = ["None"] # if the prediction is empty, then use "None" as the weakness   
                all_papers_weakness_list.append((id, pred_weakness_list))
            
            # calculate cross paper diversity
            cross_paper_itf_idf_score, cross_paper_id2scorelis = cross_focus_diversity(all_papers_weakness_list, model, threshold=args.threshold, inverse_tf=True, batch_size=args.batch_size)
            # also, for each papaer, save the top k items in the weakness list
            papaer_id2items_save = {}
            for id, weakness_list in all_papers_weakness_list:
                scores = cross_paper_id2scorelis[id]
                assert len(scores) == len(weakness_list), "length of scores and weakness list should be the same, and one-by-one correspondence, but got {} and {}".format(len(scores), len(weakness_list))
                # pick the top k items
                top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.papaer_top_k]
                top_k_items = [weakness_list[i] for i in top_k_idx]
                papaer_id2items_save[id] = top_k_items
            
            
            # import pdb; pdb.set_trace()
            # get all the tracks' weakness list
            all_tracks_weakness_list = []
            for track_name, paper_ids in track2id.items():
                # found all the papers' weakness list within this track
                track_weakness_list = []
                for id, pred_weakness_list in all_papers_weakness_list:
                    if id in paper_ids:
                        track_weakness_list.extend(pred_weakness_list)
                all_tracks_weakness_list.append((track_name, track_weakness_list))
            
            # calculate cross track diversity
            # import pdb; pdb.set_trace()
            cross_track_tf_idf_score, cross_track_id2scorelis = cross_focus_diversity(all_tracks_weakness_list, model, threshold=args.threshold, inverse_tf=False, batch_size=args.batch_size)
            # also save the top k items in the weakness list, for each track
            track_id2items_save = {}
            for id, weakness_list in all_tracks_weakness_list:
                scores = cross_track_id2scorelis[id]
                assert len(scores) == len(weakness_list), "length of scores and weakness list should be the same, and one-by-one correspondence, but got {} and {}".format(len(scores), len(weakness_list))
                # pick the top k items
                top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.track_top_k]
                top_k_items = [weakness_list[i] for i in top_k_idx]
                track_id2items_save[id] = top_k_items
            
            print("*"*50)
            print("==> for prediction file: ", pred_file)
            print("==> cross papaer diversity (ITF-IDF): ", round(cross_paper_itf_idf_score, 4))
            print("==> cross track diversity (TF-IDF): ", round(cross_track_tf_idf_score, 4)) 
            print("==> save the result to: ", save_file)
            
            with open(save_file, "w") as f:
                json.dump({"cross_paper_itf_idf_score": cross_paper_itf_idf_score,
                        "cross_track_tf_idf_score": cross_track_tf_idf_score,
                        "track_id2items_save": track_id2items_save,
                        "papaer_id2items_save": papaer_id2items_save,
                        }, f, indent=4)
       
    
if __name__ == "__main__":
    main()

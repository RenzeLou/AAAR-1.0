'''
let the models (close_source) predict the experiments and explanations for the subtask2
'''
import base64
import copy
import random
import re
import sys
import numpy as np
import openai
import dataclasses
import logging
import tenacity
import tiktoken
from typing import Optional, Sequence, Union, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from datetime import datetime
import os
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm

from chat_completion import openai_chat_completion
from prompt_templates import Exp_entailment
from calculate_metrics_src import metric_max_over_ground_truths, exact_match_score

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 720  # make sue to set this value smaller, otherwise, the repeated reduce will be very slow
    temperature: float = 1
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

def cut_word(input_context:str, max_word_len:int):
    words = word_tokenize(input_context)
    # import pdb; pdb.set_trace()
    words = words[:max_word_len]
    cutted_text = TreebankWordDetokenizer().detokenize(words)
    return cutted_text


def read_tex(file):
    '''
    not sure with the encoding of the tex file, try multiple encodings to read the tex file
    '''
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'windows-1252']  # Add more as needed.
    tex_content = None
    for enc in encodings:
        try:
            with open(file, "r", encoding=enc) as f:
                tex_content = f.readlines()
            # print(f"Success with encoding: {enc}")
            break  # Stop trying after the first successful read.
        except UnicodeDecodeError:
            # print(f"Failed with encoding: {enc}")
            continue
    
    return tex_content

def tex_cleaning(tex_content):
    '''
    for a given read tex content, clean the content to remove all the comments (those lines start with "%")
    '''
    cleaned_tex = []
    for line in tex_content:
        # delete all the comments in this line
        line = re.sub(r"%.*", "", line)
        # each line should end with "/n"
        if not line.endswith("\n"):
            line += "\n"
        cleaned_tex.append(line)
    
    return cleaned_tex

def save_intermediate_results(all_items, save_file, save_path, message):
    file_name = os.path.basename(save_file)
    file_name = file_name.rsplit(".", 1)[0] + f".{message}.json"
    terminate_save_path = os.path.join(save_path, "terminated_results")
    os.makedirs(terminate_save_path, exist_ok=True)
    with open(os.path.join(terminate_save_path, file_name), "w") as f:
        json.dump(all_items, f, indent=2)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_all_used_images(path):
    '''
    read all the images under the path (these are all the images used in the input context, according to the tex file)
    assume that all the images are processed as png files!!!
    '''
    # get all the png files under the path
    all_png_files = []
    for f in os.listdir(path):
        if f.endswith(".png"):  # TODO: only consider the png files!!! all images should already be processed to png files
            all_png_files.append(os.path.join(path, f))

    # encode all the png files into base64
    all_images_str = []
    for png_file in all_png_files:
        all_images_str.append(encode_image(png_file))
    
    return all_images_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-4-1106-preview", help="the name of the openai model to use.")
    parser.add_argument("--paper_selection_path", type=str, default="/data/rml6079/projects/scientific_doc/temp/subtask2_experiment_human_anno/2_human_eval_exp_entail/paper_select.json")
    parser.add_argument("--model_prediction_path", type=str, default="gpt-4o-3000-oracle---202409291738")
    parser.add_argument("--root_dir", type=str, default="/data/rml6079/projects/scientific_doc/temp/subtask2_experiment_human_anno/eval_results")
    parser.add_argument("--save_dir", type=str, default="/data/rml6079/projects/scientific_doc/temp/subtask2_experiment_human_anno/2_human_eval_exp_entail")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    # openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    if 'gemini' in args.api_name:
        client = None
    else:
        client = openai.OpenAI()

    decoding_args = OpenAIDecodingArguments()
    
    template = Exp_entailment()
    
    random.seed(args.seed)
    
    
    # read the paper id selected
    with open(args.paper_selection_path, "r") as f:
        paper_id_selected = json.load(f)    
    all_paper_ids = []
    for ids in paper_id_selected.values():
        all_paper_ids.extend(ids)
    
    print(f"==> for the model results: {args.model_prediction_path}")
    print(f"==> you select {len(set(all_paper_ids))} papers to evaluate.")
    
    save_dir = os.path.join(args.save_dir, args.model_prediction_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # read the save_dir to see if there is any paper has already been predicted
    # get all the subfolder name under the save_dir (only folder, exlucde the file)
    all_subfolders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f)) and os.path.exists(os.path.join(save_dir, f, "eval_results.json"))]
    print(f"==> already predicted {len(set(all_subfolders))} papers.")
    papers_ids_to_predict = list(set(all_paper_ids) - set(all_subfolders))
    print(f"==> {len(papers_ids_to_predict)} papers left to be predicted.")
    
    source_dir = os.path.join(args.root_dir, args.model_prediction_path)
    for paper_id in tqdm(papers_ids_to_predict):
        model_prediction_file = os.path.join(source_dir, paper_id, "eval_results.json")
        with open(model_prediction_file, "r") as f:
            model_prediction = json.load(f)
        gt_exps = model_prediction["output"]["What experiments do you suggest doing?"]  # m human-annotated experiments
        pred_exps = model_prediction["predicton"]["What experiments do you suggest doing?"] # n model-predicted experiments
        # get pair-wise decision
        pairwise_decision = []  # m x n matrix
        for gt_exp in tqdm(gt_exps):
            current_gt_decision = []
            for pred_exp in pred_exps:
                # prompt gpt4o to decide whether the pred_exp is the same as the gt_exp
                prompt_dict = {
                    "EXP1": gt_exp,
                    "EXP2": pred_exp
                }
                decision, _ = openai_chat_completion(client, prompt_dict, template, decoding_args, model_name=args.api_name)
                if decision is not None:
                    decision = int(decision) 
                else: 
                    decision = 0  # if the model response format is wrong (no 0 or 1 found), set it to 0 by default
                    print("** get wrong response from the model, set it to 0 by default.")
                current_gt_decision.append(decision)
            pairwise_decision.append(current_gt_decision)
        # calculate the entailment score
        matrix = np.array(pairwise_decision)
        gt_entail_res = np.any(matrix, axis=1).astype(int)  # m 
        pred_entail_res = np.any(matrix, axis=0).astype(int) # n
        gt_entail_score = np.mean(gt_entail_res)
        pred_entail_score = np.mean(pred_entail_res)
        # get those novel exp in the model prediction exps (get the exps corresponding to the 0 in pred_entail_res)
        novel_exps = [pred_exps[i] for i in range(len(pred_exps)) if pred_entail_res[i] == 0]
        # save the pairwise decision
        results = {
            "id": paper_id,
            "ground_truth": gt_exps,
            "prediction": pred_exps,
            "pairwise_decision": pairwise_decision,
            "recall_gt_entail_score": gt_entail_score,
            "precision_pred_entail_score": pred_entail_score,
            "novel_exps": novel_exps
        }
        temp_dir = os.path.join(save_dir, paper_id)
        os.makedirs(temp_dir, exist_ok=True)
        save_file = os.path.join(temp_dir, "eval_results.json")
        with open(save_file, "w") as f:
            json.dump(results, f, indent=2)
    
    
    
    print("="*20)
    print(f"Results saved to: {save_dir}")
    
    
if __name__ == "__main__":
    main()

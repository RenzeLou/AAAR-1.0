'''
let the models (close_source) predict the experiments and explanations for the subtask2
'''
import copy
import random
import re
import sys
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
from prompt_templates import Exp_eval, Exp_explanation_eval
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-4-1106-preview", help="the name of the openai model to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask2_experiment_human_anno/final_data", help="the directory save the data.")
    parser.add_argument("--save_dir", type=str, default="./subtask2_experiment_human_anno/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--max_word_len", type=int, default=2800, help="the maximum length of the context to provide to the model. For GPT4-o, default 2800 words, otherwise might exceed the limit.")
    parser.add_argument("--oracle", action="store_true", help="whether to use the oracle to generate the explanation list.")
    parser.add_argument("--max_retry", type=int, default=3, help="the maximum number of retries for each instance.")
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
    
    experiment_template = Exp_eval()
    explanation_template = Exp_explanation_eval()
    
    random.seed(args.seed)
    
    # read the eval data
    print("==> reading the eval data")
    eval_data = []
    for subfolder in tqdm(os.listdir(args.root_dir)):
        if subfolder == "all_paper_ids.txt":
            continue
        data_json = os.path.join(args.root_dir, subfolder, "data_text.json")
        with open(data_json, "r") as f:
            data_text = json.load(f)
        # TODO: add image data in the future
        eval_data.append(data_text)
    print(f"==> total instances: {len(eval_data)}\n")
    
    
    # sometimes the api_name will be like a path (for the open source LLM), e.g., `mistralai/Mistral-7B-Instruct-v0.1`
    # replace `/` with `_` to avoid the error of creating a directory with the name of the path
    api_name_save = args.api_name.replace("/", "_")
    api_name_save = api_name_save + "-" + str(args.max_word_len)
    api_name_save = api_name_save + "-oracle" if args.oracle else api_name_save
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    api_name_save = api_name_save + "---" + current_time 
    for eval_ins in tqdm(eval_data):
        paper_id = eval_ins["id"]
        target_dir = os.path.join(args.save_dir, f"{api_name_save}", paper_id)
        os.makedirs(target_dir, exist_ok=True)
        
        input_text = eval_ins["input"]
        input_text = "".join(input_text)
        input_text_cut = cut_word(input_text, args.max_word_len)
        gt_experiment = eval_ins["output"]["What experiments do you suggest doing?"]
        gt_explanation = eval_ins["output"]["Why do you suggest these experiments?"]
        
        # experiment prediction
        eval_dict_experiment = {
            "context_input": input_text_cut,
        }
        retry_falg = True
        retry_cnt = 0
        while retry_falg:
            pred_experiment, _ = openai_chat_completion(client, eval_dict_experiment, experiment_template, decoding_args, model_name=args.api_name)
            if isinstance(pred_experiment, list):
                retry_falg = False
            else:
                print("*** expect got a list, but got: '", pred_experiment, "' retrying...")
                retry_cnt += 1
                if retry_cnt > args.max_retry:
                    print("*** max retry reached, skip this instance.")
                    pred_experiment = []
                    retry_falg = False
        
        # explanation prediction
        if args.oracle:
            eval_dict_explanation = {
                "context_input": input_text_cut,
                "experiment_list": "\n".join(gt_experiment)
            }
            experiment_list_len = len(gt_experiment)
        else:
            eval_dict_explanation = {
                "context_input": input_text_cut,
                "experiment_list": "\n".join(pred_experiment)
            }
            experiment_list_len = len(pred_experiment)
            
        if eval_dict_explanation["experiment_list"] == "":
            pred_explanation = []
        else:
            retry_falg = True
            retry_cnt = 0
            while retry_falg:
                pred_explanation, _ = openai_chat_completion(client, eval_dict_explanation, explanation_template, decoding_args, model_name=args.api_name)
                if isinstance(pred_explanation, list) and len(pred_explanation) == experiment_list_len:  # TODO: note that, for open source LLM, you cannot expect the model will well follow instruction to produce a valid list, then the check should be chill
                    retry_falg = False
                else:
                    if not isinstance(pred_explanation, list):
                        print("*** expect got a list, but got: '", pred_explanation, "' retrying...")
                    elif len(pred_explanation) != experiment_list_len:
                        print(f"*** expect the same length of the experiment list, but got: {len(pred_explanation)} vs {experiment_list_len}")
                    retry_cnt += 1
                    if retry_cnt > args.max_retry:
                        print("*** max retry reached, skip this instance.")
                        pred_explanation = []
                        retry_falg = False
        
        # save all the prediction list (will calculate metrics later)
        save_result_dict = {
            "id": paper_id,
            "output": eval_ins["output"],
            "predicton": {
                "What experiments do you suggest doing?": pred_experiment,
                "Why do you suggest these experiments?": pred_explanation
            },
            "api_name": args.api_name,
            "max_word_len": args.max_word_len,
            "oracle": args.oracle,
            "time": current_time,
            "input": input_text,
            "input_cut": input_text_cut,
        }
        with open(os.path.join(target_dir, "eval_results.json"), "w") as f:
            json.dump(save_result_dict, f, indent=4)
    
    
    print("="*20)
    print(f"Model: {args.api_name}")
    print(f"Input max word len: {args.max_word_len}")
    print(f"Oracle: {args.oracle}")
    print("="*20)
    print(f"Results saved to: {os.path.join(args.save_dir, api_name_save)}")
    
    
if __name__ == "__main__":
    main()

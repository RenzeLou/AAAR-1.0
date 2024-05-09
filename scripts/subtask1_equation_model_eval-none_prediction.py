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
import shutil
import subprocess
import time
import arxiv
from arxiv import Client
import requests
import os
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from acl_anthology import Anthology
import nltk

os.environ["NLTK_DATA"] = "/data/rml6079/nltk_data"

from chat_completion import openai_chat_completion
from prompt_templates import EquationRewrite_Difficult, EquationRewrite_Easy, Equation_eval_none
from calculate_metrics_src import metric_max_over_ground_truths, exact_match_score

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 4000
    temperature: float = 1
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


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
    parser.add_argument("--template", type=int, default=1, help="the type of prompt template to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask1_equation_unified", help="the directory save the data.")
    parser.add_argument("--eval_data_file", type=str, default="equation_completion_1_per_paper.json", help="the file to save the generated instances.")
    parser.add_argument("--save_dir", type=str, default="./subtask1_equation_unified/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--context_max_len", type=int, default=100, help="the maximum length of the context to provide to the model.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    client = openai.OpenAI()
    decoding_args = OpenAIDecodingArguments()
    if args.template == 1:
        template = Equation_eval_none()
    else:
        raise ValueError(f"Unknown template type: {args.template}")
    
    
    random.seed(args.seed)
    
    eval_data = os.path.join(args.root_dir, args.eval_data_file)
    with open(eval_data, "r") as f:
        eval_data = json.load(f)
    
    os.makedirs(os.path.join(args.save_dir, f"{args.eval_data_file}", f"{args.api_name}"), f"{args.context_max_len}", exist_ok=True)
    record_save_file = os.path.join(args.save_dir, f"{args.eval_data_file}", f"{args.api_name}", f"{args.context_max_len}", "performances.json")
    prediction_save_file = os.path.join(args.save_dir, f"{args.eval_data_file}", f"{args.api_name}", f"{args.context_max_len}", "eval_results.json")
    
    results_list = []
    try:
        for eval_ins in tqdm(eval_data):
            answer = eval_ins.pop("answer")  # either A, B, C, or D
            options_str = eval_ins["options"]
            option_list = eval_ins.get("options_list", None)
            # options is a string: f"(A). `{options[0]}`;\n(B). `{options[1]}`;\n(C). `{options[2]}`;\n(D). `{options[3]}`"
            # use re to split the options into four parts
            def extract_options(text):
                # Pattern to capture content inside backticks including backslashes and newlines
                pattern = r"`([^`]*)`"
                
                # Find all matches and return them as a list
                return re.findall(pattern, text)
            
            if option_list is None:
                options = extract_options(options_str)
                if len(options) > 4:
                    # try to do some simple combination
                    combine_num = len(options) // 4
                    new_options = []
                    for i in range(combine_num):
                        new_options.append(" ".join(options[i*4:(i+1)*4]))
                    options = new_options
                    print(f"==> len(options) > 4, combine the options into {combine_num} new options")
                elif len(options) < 4:
                    # # try to do some simple split (by empty strings, e.g., " ", "/n", etc.)
                    # split_num = 4 // len(options)
                    # new_options = []
                    # for i in range(len(options)):
                    #     new_options.extend(re.split(r"\s+", options[i]))
                    # options = new_options
                    # print(f"==> len(options) < 4, split the options into {split_num} new options")
                    # simply add some empty options
                    add_num = 4 - len(options)
                    options = options + [""] * add_num
                    print(f"==> len(options) < 4, add {add_num} empty options")
            else:
                options = option_list
                   
            assert len(options) == 4, f"options: {options}, len: {len(options)}"
            
            # delete the answer from the options
            answer_idx = ord(answer) - ord("A")
            wrong_options = options[:answer_idx] + options[answer_idx+1:]
            assert len(wrong_options) == 3, f"wrong_options: {wrong_options}, len: {len(wrong_options)}"
            
            random.shuffle(wrong_options)
            wrong_option_str = f"(A). `{wrong_options[0]}`;\n(B). `{wrong_options[1]}`;\n(C). `{wrong_options[2]}`"
            
            # only use context_max_len words for the context input
            context_before = eval_ins["context_before"]
            context_after = eval_ins["context_after"]
            
            # for context_before, use the last context_max_len words
            context_before = " ".join(context_before.split()[-args.context_max_len:])
            # for context_after, use the first context_max_len words
            context_after = " ".join(context_after.split()[:args.context_max_len])
            
            # # for the context_before, use the last 3 paragraphs
            # MAX_PARA = 3
            # # use "\n", "\n\n", "\n\n\n", etc. namely the paragraph separator
            # sentence_split_pattern = re.compile(r"\n+")
            # context_before = sentence_split_pattern.split(context_before)
            # context_after = sentence_split_pattern.split(context_after)
            # context_before = " ".join(context_before[-MAX_PARA:])
            # context_after = " ".join(context_after[:MAX_PARA])
            
            value_dict = {
                "context_before": context_before,
                "context_after": context_after,
                "options": wrong_option_str
            }
            
            pred, cost = openai_chat_completion(client,value_dict, template, decoding_args, model_name=args.api_name)
            results_list.append({
                "context_before": context_before,
                "context_after": context_after,
                "options": wrong_options,
                "pred": str(pred),
                "answer": "None", # the ground truth answer is None --- we hope the model can predict None since all three options are wrong
                "cost": cost
            })
    except tenacity.RetryError as e:
            print("==> Error: {}".format(e))
            print("\nOpenAI API rate limit reached. Please increase the waiting/retry times in the tenacity decorator.\n")
            save_intermediate_results(results_list, "eval_results.json", os.path.join(args.save_dir, f"{args.eval_data_file}", f"{args.api_name}"), e)
            sys.exit(1)
    
    # calculate the EM --- how much the model predict as none, i.e., identify the wrong options successfully
    exact_match = 0
    for res in results_list:
        pred = res["pred"]
        pred = pred.strip().lower()
        score = "none" in pred
        score = int(score)
        exact_match += score
    exact_match = 100.0 * exact_match / len(results_list)
    
    
    # save the performances and those important args
    records = {
        "EM (none-predict percentage)": exact_match,
        "total_instances": len(results_list),
        "model_name": args.api_name,
        "eval_data_file": args.eval_data_file,
        "context_max_len": args.context_max_len
    }
    
    with open(record_save_file, "w") as f:
        json.dump(records, f, indent=4)

    with open(prediction_save_file, "w") as f:
        json.dump(results_list, f, indent=4)
    
    
    print("="*20)
    print(f"Model: {args.api_name}")
    print(f"Eval data file: {args.eval_data_file}")
    print("="*20)
    print(f"EM: {exact_match:.2f}")
    print(f"Total instances: {len(results_list)}")
    print("="*20)
    print("save the performances to ", record_save_file)
    print("save the predictions to ", prediction_save_file)
    
    
if __name__ == "__main__":
    main()

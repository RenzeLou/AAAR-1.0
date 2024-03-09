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

from chat_completion import openai_chat_completion
from prompt_templates import EquationRewrite_Difficult, EquationRewrite_Easy


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1024
    temperature: float = 1.2  # TODO: is it good to encourage diverse wrong equations?
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
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo-1106", help="the name of the openai model to use.")
    parser.add_argument("--template", type=int, default=1, help="the type of prompt template to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask1_equation", help="the directory to save the downloaded papers.")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the existing files.")
    parser.add_argument("--ins_per_paper", type=int, default=1, help="the number of instances to generate per paper. for an eval set, we can use one paper for multiple instances, but not for training set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    client = openai.OpenAI()
    decoding_args = OpenAIDecodingArguments()
    if args.template == 1:
        template = EquationRewrite_Easy()
    elif args.template == 2:
        template = EquationRewrite_Difficult()
    else:
        raise ValueError(f"Unknown template type: {args.template}")
    
    
    random.seed(args.seed)
    
    rewrite_ins_num = 0
    origin_total_ins_num = 0
    
    # for each subdir under root_dir
    for subfolder in tqdm(os.listdir(args.root_dir)):
        # paper_id = subfolder
        # meta_path = os.path.join(args.root_dir, subfolder, f"{subfolder}_metadata.json")
        # main_tex = read_tex(os.path.join(args.root_dir, subfolder, "cleaned_tex.tex"))
        # if main_tex is None:
        #     raise ValueError(f"Cannot read the tex file of {subfolder}")
        
        # read the equations.json file
        equation_file = os.path.join(args.root_dir, subfolder, "equations.json")
        with open(equation_file, "r") as f:
            ins_list = json.load(f)
        
        save_folder = os.path.join(args.root_dir, subfolder)    
        save_file = os.path.join(save_folder, "equations_cls.json")
        # check if there is already a file under the subfolder
        if os.path.exists(save_file) and not args.overwrite:
            continue
        
        origin_total_ins_num += len(ins_list)
        
        sample_ins_num = min(args.ins_per_paper, len(ins_list))
        
        ins_list = ins_list[:sample_ins_num]
        
        try:
            # for each instance in the ins_list, create a new instance 
            new_ins_list = []
            for ins in ins_list:
                context_before = ins["context_before"]
                context_after = ins["context_after"]
                equation = ins["equation"]
                gpt_query = {
                    "ori_equation": equation
                }
                # ask GPT to rewrite the equation
                wrong_eq_1, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                wrong_eq_2, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                wrong_eq_3, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                
                # make a multi-choice question with these 4 options
                options = [equation, wrong_eq_1, wrong_eq_2, wrong_eq_3]
                random.shuffle(options)
                # find the idx of the correct option after shuffling
                correct_idx = options.index(equation)
                ordered_options = f"(A). `{options[0]}`;\n(B). `{options[1]}`;\n(C). `{options[2]}`;\n(D). `{options[3]}`"  # in our dataset, we provide the shuffled order of the options, to ensure others could reproduce the results
                correct_option = chr(ord('A') + correct_idx)
                
                new_ins = copy.deepcopy(ins)
                new_ins["wrong_equations"] = [wrong_eq_1, wrong_eq_2, wrong_eq_3]
                new_ins["options"] = ordered_options
                new_ins["answer"] = correct_option
                new_ins_list.append(new_ins)
        # except openai.error.RateLimitError as e:
        except tenacity.RetryError as e:
            print("==> Error: {}".format(e))
            print("\nOpenAI API rate limit reached. Please increase the waiting/retry times in the tenacity decorator.\n")
            # save_intermediate_results(outputs, args, "RateLimitError")
            # sys.exit(0)  # Exit the program gracefully
        
        # save the new_ins_list to the new file, under the same folder
        with open(save_file, "w") as f:
            json.dump(new_ins_list, f, indent=4)
        
        rewrite_ins_num += len(new_ins_list)
    
    
    print("="*50)
    print(f"Total instances before rewriting: {origin_total_ins_num}")
    print(f"Select the first (at most) {args.ins_per_paper} instances for each paper, and rewrite the equations")
    print(f"Total instances after rewriting: {rewrite_ins_num}")
    
    
    
if __name__ == "__main__":
    main()

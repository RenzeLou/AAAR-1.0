'''
gather all the instances distributed under various subfolders into one file
and delete some verbose fields in the instances
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./subtask1_equation", help="the directory to save the downloaded papers.")
    parser.add_argument("--save_dir", type=str, default="./subtask1_equation_unified", help="the directory to save the unified instances.")
    parser.add_argument("--save_file", type=str, default="equation_completion.json", help="the file to save the generated instances.")
    parser.add_argument("--ins_per_paper", type=int, default=1, help="the number of instances to generate for each paper.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    
    random.seed(args.seed)
    
    origin_total_ins_num = 0
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    all_ins = []
    # for each subdir under root_dir
    for subfolder in tqdm(os.listdir(args.root_dir)):
        # paper_id = subfolder
        # meta_path = os.path.join(args.root_dir, subfolder, f"{subfolder}_metadata.json")
        # main_tex = read_tex(os.path.join(args.root_dir, subfolder, "cleaned_tex.tex"))
        # if main_tex is None:
        #     raise ValueError(f"Cannot read the tex file of {subfolder}")
        
        # read the equations.json file
        equation_file = os.path.join(args.root_dir, subfolder, "equations_cls.json")
        with open(equation_file, "r") as f:
            ins_list = json.load(f)
        
        origin_total_ins_num += len(ins_list)
        sample_num = min(args.ins_per_paper, len(ins_list))
        ins_list = ins_list[:sample_num]
        
        for ins in ins_list:
            context_before = ins["context_before"]
            context_after = ins["context_after"]
            options = ins["options"]
            answer = ins["answer"]
            new_ins = {
                "context_before": context_before,
                "context_after": context_after,
                "options": options,
                "answer": answer
            }
            all_ins.append(new_ins)
            
    with open(os.path.join(args.save_dir, args.save_file), "w") as f:
        json.dump(all_ins, f, indent=2)
    
    print(f"Original instances: {origin_total_ins_num}")
    print(f"Sampled instances per paper: {args.ins_per_paper}")
    print(f"Total final instances: {len(all_ins)}")
    print(f"Saved to {os.path.join(args.root_dir, args.save_file)}")
    
    

if __name__ == "__main__":
    main()
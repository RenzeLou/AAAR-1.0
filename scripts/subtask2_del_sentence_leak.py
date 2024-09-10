'''
use this script for seperating the context before and after the "experiment" section of a paper
'''
import dataclasses
import random
import re
import openai
import requests
import os
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm

from prompt_templates import Experiment_leak
from chat_completion import openai_chat_completion

MAX_TOLERACE = 8

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 512
    temperature: float = 1  
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def del_leak_sentences(client,tex_content:list, exp_list:str, api_name="gpt-4o"):
    # del all the empty string in the tex_content
    tex_content = [line for line in tex_content if line.strip()]
    template = Experiment_leak()
    decoding_args = OpenAIDecodingArguments()
    
    cleaned_tex_content = []
    decision_list = []
    for sen in tqdm(tex_content):
        retry_flag = True
        retry_cnt = 0
        while retry_flag:
            gpt_query = {
                            "experiment_list": exp_list,
                            "sentence": sen
                        }
            decision, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=api_name)
            try:
                decision = int(decision)
                retry_flag = False
            except:
                print("wrong reponse format, should be int, but got: ", decision, "the sentence is ", sen)
                retry_cnt += 1
                if retry_cnt > MAX_TOLERACE:
                    print("too many retries, exit.")
                    retry_flag = False
                    decision = 0  # just keep the sentence
        
        if decision == 0:
            cleaned_tex_content.append(sen)
            
        decision_list.append(decision)
    
    # remain_sen_percentage = len(cleaned_tex_content) / len(tex_content)
            
    return cleaned_tex_content, tex_content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./subtask2_experiment_human_anno")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing files.")
    parser.add_argument("--api_key", type=str, default=None, help="the openai api key.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    client = openai.OpenAI()
    
    total_paper, final_used_paper = 0, 0
    suitable_paper_num = 0  # note not every crawled paper is suitable for this task, for example, we omit those papers with complex latex structure and has no explicit "experiment" section

    # for each subdir under root_dir
    process_paper_num = 0
    has_processed = 0
    del_percentage_list = []
    for subfolder in os.listdir(args.root_dir):
        if "_seperate" in subfolder:
            print("==> processing subfolder: {}".format(subfolder))
            # get all the dir under the subfolder
            all_paper_dir = os.listdir(os.path.join(args.root_dir, subfolder))
            for paper_dir in tqdm(all_paper_dir):
                if "_dep" in paper_dir:
                    # skip those deperated papers
                    continue
                context_file = os.path.join(args.root_dir, subfolder, paper_dir, "context_before_after_exp.json")
                annotation_file = os.path.join(args.root_dir, subfolder, paper_dir, "annotation.json")
                if not os.path.exists(context_file):
                    continue
                try:
                    with open(context_file, "r") as f:
                        context = json.load(f)
                except Exception as e:
                    print(f"Error: {e} in loading {context_file}")
                    exit(0)
                
                # read the annotation experiment list, used for gpt to make decision 
                with open(annotation_file, "r") as f:
                    annotation = json.load(f)
                what_list = annotation["What experiments do you suggest doing?"]
                what_list_sen = "\n".join(what_list)
                
                # if this paper's context has already been processed, then skip
                if context.get("context_before_exp_cleaned", None) is not None and not args.overwrite:
                    print(f"*** {subfolder}/{paper_dir} already processed, skip.")
                    has_processed += 1
                    del_percentage = context.get("del_percentage", None)
                    if del_percentage:
                        del_percentage_list.append(del_percentage)
                    continue    
                
                context_before_exp = context["context_before_exp"]
                context_before_exp_del, context_before_exp_ori = del_leak_sentences(client,context_before_exp, what_list_sen)
                context["context_before_exp_cleaned"] = context_before_exp_del
                
                del_percentage = (len(context_before_exp_ori) - len(context_before_exp_del)) / len(context_before_exp_ori)
                del_percentage = round(del_percentage, 5)
                context["del_percentage"] = del_percentage
                del_percentage_list.append(del_percentage)
                
                with open(context_file, "w") as f:
                    json.dump(context, f, indent=4)
                process_paper_num += 1
                

    print("="*50)
    print("Already processed {} papers before.".format(has_processed))
    print("Totally {} papers context have been processed now.".format(process_paper_num))
    print("The deletion percentage: min={}, max={}, mean={}".format(min(del_percentage_list), max(del_percentage_list), sum(del_percentage_list)/len(del_percentage_list)))
    
    
    
if __name__ == "__main__":
    main()
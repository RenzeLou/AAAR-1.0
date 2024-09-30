'''
let the models (open_source) predict the experiments and explanations for the subtask2
v2 version will forward each time in the experiment list to the model one by one
'''
import copy
import random
import re
import sys
import dataclasses
import logging
from typing import Optional, Sequence, Union, List
from datetime import datetime
import os
import json
import argparse
from tqdm import tqdm

from prompt_templates import Exp_eval, Exp_explanation_eval_v2
from vllm import LLM, SamplingParams
from huggingface_hub import login
with open("huggingface_key.txt", "r") as f:
    hf_key = f.read().strip()
login(hf_key)

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


def process_output(outputs:list):
    '''
    convert the RequestOutput objects of the vllm to the list of generated text
    '''
    output_text = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_text.append(generated_text)
    return output_text

def process_text_to_list(response:str):
    response = response.strip()
    response_list = re.findall(r"\d+\..*", response)
    # since we are using open source model, the response might doesn't follow a list format
    # if cannot extract the list, then return the whole response
    if len(response_list) == 0:
        # attach "1. "
        response = "1. " + response
        response_list = [response]
    
    return response_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="the name of the model to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask2_experiment_human_anno/final_data", help="the directory save the data.")
    parser.add_argument("--save_dir", type=str, default="./subtask2_experiment_human_anno/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--max_word_len", type=int, default=3000, help="the maximum length of the context to provide to the model. For GPT4-o, default 2800 words, otherwise might exceed the limit.")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p for sampling")
    parser.add_argument("--oracle", action="store_true", help="whether to use the oracle to generate the explanation list.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--gpu_num", type=int, default=1, help="the number of gpt model")
    parser.add_argument("--max_model_len", type=int, default=10000, help="the maximum input length of the model")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p)
    llm = LLM(model=args.api_name, tensor_parallel_size=args.gpu_num, max_model_len=args.max_model_len)
                      
    experiment_template = Exp_eval()
    explanation_template = Exp_explanation_eval_v2()
    
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
    
    all_input_prompts_experiments, all_input_prompts_explanations = [], []
    # target_dir_list = []
    print("===> constructing prompts for experiments generation ...")
    for eval_ins in tqdm(eval_data):
        paper_id = eval_ins["id"]
        input_text = eval_ins["input"]
        input_text = "".join(input_text)
        input_text_cut = cut_word(input_text, args.max_word_len)
        gt_experiment = eval_ins["output"]["What experiments do you suggest doing?"]
        gt_explanation = eval_ins["output"]["Why do you suggest these experiments?"]
        
        # experiment prediction
        eval_dict_experiment = {
            "context_input": input_text_cut,
        }
        experiment_input_prompt = experiment_template.query_prompt.format_map(eval_dict_experiment)
        all_input_prompts_experiments.append(experiment_input_prompt)
    
    print("===> experiment list generation ...")
    experiment_outputs = llm.generate(all_input_prompts_experiments, sampling_params)
    experiment_outputs_text = process_output(experiment_outputs)
    experiment_outputs_list = [process_text_to_list(out) for out in experiment_outputs_text]
    
    print("===> constructing prompts for explanations generation ...")
    for i, eval_ins in tqdm(enumerate(eval_data)):
        paper_id = eval_ins["id"]        
        input_text = eval_ins["input"]
        input_text = "".join(input_text)
        input_text_cut = cut_word(input_text, args.max_word_len)
        gt_experiment = eval_ins["output"]["What experiments do you suggest doing?"]
        gt_explanation = eval_ins["output"]["Why do you suggest these experiments?"]
        
        # explanation prediction
        if args.oracle:
            experiment_list = gt_experiment
        else:
            experiment_list = experiment_outputs_text[i]
        experiment_list_len = len(experiment_list)
        
        
        if experiment_list_len == 0:
            prompts_for_this_experiment_list = []
        else:
            prompts_for_this_experiment_list = []
            for i, exp_item in enumerate(experiment_list):
                # remove "1.", "2." in the beginning of the item
                exp_item_used = re.sub(r"^\d+\.", "", exp_item)
                eval_dict_explanation = {
                    "context_input": input_text_cut,
                    "experiment_list": exp_item_used
                }
                # each item has its own prompt
                explanation_input_prompt = explanation_template.query_prompt.format_map(eval_dict_explanation)
                prompts_for_this_experiment_list.append(explanation_input_prompt)
        # 
        all_input_prompts_explanations.append(prompts_for_this_experiment_list)

    
    print("===> explanation list generation ...")
    explanation_outputs_list = []
    for i, prompts_for_this_experiment_list in tqdm(enumerate(all_input_prompts_explanations)):
        explanation_outputs_for_this_experiment = llm.generate(prompts_for_this_experiment_list, sampling_params)
        explanation_outputs_text_for_this_experiment = process_output(explanation_outputs_for_this_experiment)
        # attach "1.", "2." back to the beginning of the item
        explanation_outputs_text_for_this_experiment = [f"{i+1}. {exp}" for i, exp in enumerate(explanation_outputs_text_for_this_experiment)]
        explanation_outputs_list.append(explanation_outputs_text_for_this_experiment)   
            
    
    print("===> saving the results ...")
    assert len(eval_data) == len(experiment_outputs_list) == len(explanation_outputs_list), "eval_data: {}, experiment_outputs_list: {}, explanation_outputs_list: {}, target_dir_list: {}".format(len(eval_data), len(experiment_outputs_list), len(explanation_outputs_list))
    for i, eval_ins in tqdm(enumerate(eval_data)):  
        paper_id = eval_ins["id"] 
        target_dir = os.path.join(args.save_dir, f"{api_name_save}", paper_id)
        os.makedirs(target_dir, exist_ok=True)

        # save all the prediction list (will calculate metrics later)
        save_result_dict = {
            "id": eval_ins["id"],
            "output": eval_ins["output"],
            "predicton": {
                "What experiments do you suggest doing?": experiment_outputs_list[i],
                "Why do you suggest these experiments?": explanation_outputs_list[i],
            },
            "api_name": args.api_name,
            "max_word_len": args.max_word_len,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "oracle": args.oracle,
            "gpu_num": args.gpu_num,
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

'''
use open source LLM model (e.g., llama 3) instead of the openai model
'''
import copy
import random
import re
import sys
import time
import openai
import dataclasses
from typing import Optional, Sequence, Union, List
import os
import json
import argparse
from tqdm import tqdm

from prompt_templates import EquationRewrite_Difficult, EquationRewrite_Easy, Equation_eval
from vllm import LLM, SamplingParams
from calculate_metrics_src import metric_max_over_ground_truths, exact_match_score

from huggingface_hub import login
with open("huggingface_key.txt", "r") as f:
    hf_key = f.read().strip()
login(hf_key)

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="the name of the model to use.")
    parser.add_argument("--template", type=int, default=1, help="the type of prompt template to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask1_equation_unified", help="the directory save the data.")
    parser.add_argument("--eval_data_file", type=str, default="equation_completion_1_per_paper.json", help="the file to save the generated instances.")
    parser.add_argument("--save_dir", type=str, default="./subtask1_equation_unified/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--context_max_len", type=int, default=100, help="the maximum length of the context to provide to the model.")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p for sampling")
    parser.add_argument("--gpu_num", type=int, default=1, help="the number of gpt model")
    parser.add_argument("--max_model_len", type=int, default=10000, help="the maximum input length of the model")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    if args.template == 1:
        template = Equation_eval()
    else:
        raise ValueError(f"Unknown template type: {args.template}")
    
    random.seed(args.seed)
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p)
    llm = LLM(model=args.api_name, tensor_parallel_size=args.gpu_num, max_model_len=args.max_model_len)
    
    # read the eval data
    print("==> reading the eval data")
    eval_data = os.path.join(args.root_dir, args.eval_data_file)
    with open(eval_data, "r") as f:
        eval_data = json.load(f)
    print(f"==> total instances: {len(eval_data)}")
    
    # sometimes the api_name will be like a path (for the open source LLM), e.g., `mistralai/Mistral-7B-Instruct-v0.1`
    # replace `/` with `_` to avoid the error of creating a directory with the name of the path
    api_name_save = args.api_name.replace("/", "_")
    target_dir = os.path.join(args.save_dir, f"{args.eval_data_file}", f"{api_name_save}", str(args.context_max_len))
    os.makedirs(target_dir, exist_ok=True)
    record_save_file = os.path.join(target_dir, "performances.json")
    prediction_save_file = os.path.join(target_dir, "eval_results.json")
    
    
    print("===> constructing prompts ...")
    input_prompts = []
    answer_list = []
    for eval_ins in tqdm(eval_data):
        answer = eval_ins.pop("answer")
        answer_list.append(answer)
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
        
        # update the context_before and context_after
        eval_ins["context_before"] = context_before
        eval_ins["context_after"] = context_after
        
        input_prompt = template.query_prompt.format_map(eval_ins)
        input_prompts.append(input_prompt)
    
    print("===> equation prediction ...")
    # import pdb; pdb.set_trace()
    start_time = time.time()
    outputs = llm.generate(input_prompts, sampling_params)
    outputs = process_output(outputs)
    outputs = [output.strip() for output in outputs]
    end_time = time.time()
    
    assert len(outputs) == len(eval_data), f"len(outputs): {len(outputs)}, len(eval_data): {len(eval_data)}"
    
    print("===> calculating the performances ...")
    results_list = []
    exact_match = 0.0
    for i, ins in tqdm(enumerate(eval_data)):
        answer = answer_list[i]
        answer = str(answer)
        pred = outputs[i]
        pred = str(pred)
        results_list.append({
            "context_before": ins["context_before"],
            "context_after": ins["context_after"],
            "options": ins["options"],
            "pred": pred,
            "answer": answer
        })
        score = exact_match_score(pred, answer)
        score = int(score)
        exact_match += score
    exact_match = 100.0 * exact_match / len(results_list)
    
    # save the performances and those important args
    records = {
        "EM": exact_match,
        "total_instances": len(results_list),
        "model_name": args.api_name,
        "eval_data_file": args.eval_data_file,
        "context_max_len": args.context_max_len,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gpu_num": args.gpu_num,
        "time_cost (in minutes)": (end_time - start_time) / 60.0
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

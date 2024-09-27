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
from prompt_templates import Weakness_eval

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

def cut_word_by_piece(input_context:str, num_pieces:int):
    '''
    cut the string into `num_pieces` pieces
    
    return a list contains all the pieces
    '''
    words = word_tokenize(input_context)
    input_text_length = len(words)
    num_words_per_piece = input_text_length // num_pieces + 1
    all_pieces = []
    for i in range(num_pieces):
        start_idx = i * num_words_per_piece
        end_idx = (i + 1) * num_words_per_piece
        piece = words[start_idx:end_idx]
        piece_text = TreebankWordDetokenizer().detokenize(piece)
        all_pieces.append(piece_text)
    return all_pieces

def word_num(input_text):
    words = word_tokenize(input_text)
    number_of_words = len(words)
    return number_of_words

def process_input_text(input_dict:dict):
    title = input_dict["title"]
    main_text = []
    for section in input_dict["sections"]:
        main_text.append(str(section["heading"]) + " " + str(section["text"]))
    main_text = "\n".join(main_text)
    abs = input_dict["abstractText"]
    
    # make sure every text is a string, avoid bytes-like object
    title = str(title)
    main_text = str(main_text)
    abs = str(abs)
    
    input_text = abs + "\n\n" + title + "\n\n" + main_text
    return input_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-4-1106-preview", help="the name of the openai model to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask3_review_final_light", help="the directory save the data.")
    parser.add_argument("--save_dir", type=str, default="./subtask3_review_final_light/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--split", action="store_true", help="whether to split the long input context into multiple pieces.")
    parser.add_argument("--max_word_len", type=int, default=3500, help="if split, this is the max len of each piece; if not split, this is the max cut len of the input.")
    parser.add_argument("--pick_num", type=int, default=None, help="how many instances to pick for the evaluation. Pick those shorter input paper first. If None, then use all the instances.")
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
    
    template = Weakness_eval()
    
    random.seed(args.seed)
    
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    # read the eval data
    print("\n==> reading the eval data ...")
    conf_list = ["ICLR_2022", "NeurIPS_2021", "NeurIPS_2022", "ICLR_2023"]
    all_subfolders = []
    for conf in conf_list:
        conf_dir = os.path.join(args.root_dir, conf)
        all_files_dirs = os.listdir(conf_dir)
        # make all subfolders the whole path
        all_files_dirs = [os.path.join(conf_dir, x) for x in all_files_dirs]
        # del those dir that do not have `data_text.json`
        all_files_dirs = [x for x in all_files_dirs if os.path.exists(os.path.join(x, "data_text.json"))]
        all_subfolders.extend(all_files_dirs)
    print(f"==> {len(all_subfolders)} instances found under {args.root_dir} (should be 1,925)\n")
    # import pdb; pdb.set_trace()
    
    print("==> processing the eval data ...")
    processed_input_list = []
    for subfolder_path in tqdm(all_subfolders):  
        # TODO: also should consider the images in the future
        text_file = os.path.join(subfolder_path, "data_text.json")
        with open(text_file, "r") as f:
            text_data = json.load(f)
        paper_id = text_data["ID"]
        input_text = process_input_text(text_data["input"])
        gt_output = text_data["output"]
        # if split, then cut the input text into pieces, each pieces should be less than `max_word_len`
        if args.split:
            input_text_length = word_num(input_text)
            num_pieces = input_text_length // args.max_word_len + 1
            all_pieces = cut_word_by_piece(input_text, num_pieces)
            for piece in all_pieces:
                processed_input_list.append({"id": paper_id, "input": piece, "output": gt_output, "whole_input_length": input_text_length})
        else:
            input_text_length = word_num(input_text)
            input_text_cut = cut_word(input_text, args.max_word_len)
            processed_input_list.append({"id": paper_id, "input": input_text_cut, "output": gt_output, "whole_input_length": input_text_length})    
    if args.pick_num is not None:
        # sort the processed_input_list by the input length, then id
        # TODO: should select those paper to form a new dataset instead of picking when running the code
        print(f"==> pick {args.pick_num} instances out of {len(all_subfolders)} for evaluation.")
        # short go first
        processed_input_list_sorted = sorted(processed_input_list, key=lambda x: (x["whole_input_length"], x["id"]))
        picked_piece = []
        id_set = set()
        for each_p in processed_input_list_sorted:
            this_id = each_p["id"]
            id_set.add(this_id)
            if len(id_set) > args.pick_num:
                break
            picked_piece.append(each_p)
        processed_input_list = picked_piece
        # import pdb; pdb.set_trace()
    print(f"==> {len(processed_input_list)} pieces have to be feed into the model.\n")
    
    
    print("==> start the prediction ...")
    st_time = datetime.now()
    res_dict = dict()  # {xxxx-id: [pred_1, pred_2, ...]}
    for input_piece in tqdm(processed_input_list):
        fill_in_dict = {
            "context_input": input_piece["input"]  # either the cutted text (not split) or the piece (split)
        }
        weakness_list, _ = openai_chat_completion(client, fill_in_dict, template, decoding_args, model_name=args.api_name)
        if input_piece["id"] not in res_dict:
            res_dict[input_piece["id"]] = [weakness_list]
        else:
            res_dict[input_piece["id"]].append(weakness_list)
    ed_time = datetime.now()
    
    
    def find_the_gt(id, processed_input_list):
        for item in processed_input_list:
            if item["id"] == id:
                return item["output"]
    print("==> process the predictions ...")
    # import pdb; pdb.set_trace()
    # combine all the pred_list within the same id
    for id, pred_list in tqdm(res_dict.items()):
        # each pred_x in the pred_list is a list of weakness
        combined_list = []
        for pred in pred_list:
            combined_list.extend(pred)
        # make sure the item num is consistent, e.g., 1. XXX, 2. XXX, 3. XXX
        # use re to replace "xxx. " with "i. "
        new_combined_list = []
        for i, item in enumerate(combined_list):
            new_item = re.sub(r"^\d+\.", f"{i+1}.", item)
            # import pdb; pdb.set_trace()
            new_combined_list.append(new_item)
        res_dict[id] = new_combined_list
    
    
    print("==> save the results ...")
    api_name_save = args.api_name.replace("/", "_")
    api_name_save = api_name_save + "-" + f"pick_{str(args.pick_num)}"
    api_name_save = api_name_save + "-split" if args.split else api_name_save
    # api_name_save = api_name_save + "-" + str(args.max_word_len)
    save_dir = os.path.join(args.save_dir, api_name_save, f"{str(args.max_word_len)}")
    os.makedirs(save_dir, exist_ok=True)
    
    save_list = []
    for id, pred_list in tqdm(res_dict.items()):
        gt_output = find_the_gt(id, processed_input_list)
        save_list.append({
            "id": id,
            "predicton": pred_list,
            "output": gt_output
        })
    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(save_list, f, indent=4)
        
    state_dict = {
        "total_instances": len(all_subfolders),
        "pick_instances": args.pick_num,
        "total_pieces": len(processed_input_list),
        "api_name": args.api_name,
        "start time": current_time,
        "time cost (in minutes)": (ed_time - st_time).seconds / 60,
        "split": args.split,
        "max_word_len": args.max_word_len,
        "seed": args.seed
    }
    with open(os.path.join(save_dir, "stat.json"), "w") as f:
        json.dump(state_dict, f, indent=4)
    
    
    print("="*20)
    print(f"Model: {args.api_name}")
    print(f"Total instances: {len(all_subfolders)}")
    print(f"Pick instances: {args.pick_num}")
    print(f"Split: {args.split}")
    print(f"Input max word len: {args.max_word_len}")
    print(f"Time cost: {(ed_time - st_time).seconds / 60} minutes")
    print("="*20)
    print(f"Results saved to: {save_dir}")
    
    
if __name__ == "__main__":
    main()

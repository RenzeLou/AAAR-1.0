'''
let the models (open_source) predict the weaknesses for the subtask3
also supports the multi-modal input
'''
import base64
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
from PIL import Image

from prompt_templates import Weakness_eval
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


def get_all_used_images(path, max_image_num:int=3, select_method="random", tables=True, figures=True):
    '''
    use `PIL.Image.open` to load all the images
    assume that all the images are processed as png files!!!
    '''    
    # get all the png files under the path
    all_png_files = []
    for f in os.listdir(path):
        if f.endswith(".png"):  # TODO: only consider the png files!!! all images should already be processed to png files
            all_png_files.append(os.path.join(path, f))

    figure_img, table_img = [], []
    for img_path in all_png_files:
        if "-Figure" in img_path:
            figure_img.append(img_path)
        elif "-Table" in img_path:
            table_img.append(img_path)

    used_png_files = []
    if tables:
        used_png_files.extend(table_img)
    if figures:
        used_png_files.extend(figure_img)

    # TODO: should make sure, if --tables and --figures are both True, then the selected images should make sure inlucde both tables and figures
    if select_method == "random":
        # randomly select `max_image_num` images
        select_num = min(max_image_num, len(used_png_files))
        selected_png_files = random.sample(used_png_files, select_num)
    elif select_method == "first":
        # select the first `max_image_num` images
        selected_png_files = used_png_files[:max_image_num]
    elif select_method == "last":
        # select the last `max_image_num` images
        selected_png_files = used_png_files[-max_image_num:]
    else:
        raise ValueError(f"==> select method: {select_method} is not supported, pls check.")

    all_images = []
    for img_path in selected_png_files:
        img = Image.open(img_path)
        all_images.append(img)
    
    if len(all_images) == 0:
        # make a fake image to avoid the error
        fake_path = "/data/rml6079/projects/scientific_doc/test.png"
        return [Image.open(fake_path)]
    else:
        return all_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-4-1106-preview", help="the name of the openai model to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask3_review_final_v2", help="the directory save the data.")
    parser.add_argument("--save_dir", type=str, default="./subtask3_review_final_v2/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--split", action="store_true", help="whether to split the long input context into multiple pieces.")
    parser.add_argument("--max_word_len", type=int, default=3500, help="if split, this is the max len of each piece; if not split, this is the max cut len of the input.")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p for sampling")
    parser.add_argument("--pick_num", type=int, default=None, help="how many instances to pick for the evaluation. Pick those shorter input paper first. If None, then use all the instances.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--gpu_num", type=int, default=1, help="the number of gpt model")
    parser.add_argument("--max_model_len", type=int, default=10000, help="the maximum input length of the model")
    parser.add_argument("--tables", action="store_true", help="whether to include the tables in the input context.")
    parser.add_argument("--figures", action="store_true", help="whether to include the figures in the input context.")
    parser.add_argument("--max_image_num", type=int, default=3, help="the maximum number of images to use")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=2048)
    # llm = LLM(model=args.api_name, tensor_parallel_size=args.gpu_num, max_model_len=args.max_model_len)
    # llm = LLM(
    #     model=args.api_name,
    #     trust_remote_code=True,  # Required for loading some models, such as Phi-3.5-vision
    #     max_model_len=args.max_model_len,
    #     limit_mm_per_prompt={"image": args.max_image_num} if args.max_image_num > 0 else None,
    #     tensor_parallel_size=args.gpu_num,
    #     enforce_eager=True
    # )
    llm = LLM(
        model=args.api_name,
        trust_remote_code=True,  # Required for loading some models, such as Phi-3.5-vision
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_image_num} if args.max_image_num > 0 else None,
        tensor_parallel_size=args.gpu_num,
        enforce_eager=True
    )
    
    template = Weakness_eval()
    
    random.seed(args.seed)
    
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    # read the eval data
    print("\n==> reading the eval data ...")
    conf_list = ["ICLR_2023"]  # "ICLR_2022", "NeurIPS_2021", "NeurIPS_2022", 
    all_subfolders = []
    for conf in conf_list:
        conf_dir = os.path.join(args.root_dir, conf)
        all_files_dirs = os.listdir(conf_dir)
        # make all subfolders the whole path
        all_files_dirs = [os.path.join(conf_dir, x) for x in all_files_dirs]
        # del those dir that do not have `data_text.json`
        all_files_dirs = [x for x in all_files_dirs if os.path.exists(os.path.join(x, "data_text.json"))]
        all_subfolders.extend(all_files_dirs)
    print(f"==> {len(all_subfolders)} instances found under {args.root_dir} (should be 993)\n")
    # import pdb; pdb.set_trace()
    
    print("==> processing the eval data ...")
    processed_input_list = []
    for subfolder_path in tqdm(all_subfolders):  
        text_file = os.path.join(subfolder_path, "data_text.json")
        with open(text_file, "r") as f:
            text_data = json.load(f)
        paper_id = text_data["ID"]
        input_text = process_input_text(text_data["input"])
        gt_output = text_data["output"]
        
        # read all the image
        image_dir = os.path.join(subfolder_path, "images")
        all_images = get_all_used_images(image_dir, max_image_num=args.max_image_num, select_method="random", tables=args.tables, figures=args.figures)
        
        # if split, then cut the input text into pieces, each pieces should be less than `max_word_len`
        if args.split:
            input_text_length = word_num(input_text)
            num_pieces = input_text_length // args.max_word_len + 1
            all_pieces = cut_word_by_piece(input_text, num_pieces)
            for piece in all_pieces:
                processed_input_list.append({"id": paper_id, "input": str(piece), "output": gt_output, "whole_input_length": input_text_length, "images": all_images})
        else:
            input_text_length = word_num(input_text)
            input_text_cut = cut_word(input_text, args.max_word_len)
            processed_input_list.append({"id": paper_id, "input": str(input_text_cut), "output": gt_output, "whole_input_length": input_text_length, "images": all_images})
    if args.pick_num is not None:
        # sort the processed_input_list by the input length, then id
        raise ValueError("Don not use this feature, we dont pick up during inference.")
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
    
    
    print("==>  constructing prompts ...")
    all_input_prompts = []
    all_paper_ids = set()
    all_paper_ids_list = []
    for input_piece in tqdm(processed_input_list):
        all_paper_ids.add(input_piece["id"])
        all_paper_ids_list.append(input_piece["id"])
        fill_in_dict = {
            "context_input": input_piece["input"]  # either the cutted text (not split) or the piece (split)
        }
        input_prompt = template.query_prompt.format_map(fill_in_dict)
        all_images = input_piece["images"]
        if len(all_images) > 1:
            # smt like "<image_1>\n<image_2>\n", append it before the original prompt
            img_prompt = "\n".join([f"<image_{i+1}>" for i in range(len(all_images))]) + "\n"
            input_prompt = img_prompt + input_prompt
            input_prompt = "<|user|>\n" + input_prompt + "<|end|>\n<|assistant|>\n"  # Phi's prompt format
            ins_dict = {
                "prompt": str(input_prompt),
                "multi_modal_data": {"image": all_images},
            }
        elif len(all_images) == 1:
            # if the image length is only 1, then the prompt should be different
            img_prompt = "<image>\n"
            input_prompt = img_prompt + input_prompt
            input_prompt = "<|user|>\n" + input_prompt + "<|end|>\n<|assistant|>\n"  # Phi's prompt format
            ins_dict = {
                "prompt": str(input_prompt),
                "multi_modal_data": {"image": all_images[0]},
            }
        else:
            raise ValueError(f"==> the image length is 0, pls check.")
        all_input_prompts.append(ins_dict)
    
    print("==> start the prediction ...")
    st_time = datetime.now()
    all_responses = []
    for input_prompt in all_input_prompts:
        try:
            out = llm.generate(input_prompt, sampling_params) 
            # import pdb; pdb.set_trace()
            all_responses.append(out[0].outputs[0].text)  
        except Exception as e:  # TODO: dont know why it will cause the TypeError (not a string)
            print("Error: ", e, " just use empty string as the response.")
            all_responses.append("")
            
    # all_responses = llm.generate(all_input_prompts, sampling_params)  
    # all_responses_text = process_output(all_responses)
    all_responses_text = all_responses
    all_responses_list = [process_text_to_list(out) for out in all_responses_text]
    ed_time = datetime.now()
    
    res_dict = dict()  # {xxxx-id: [pred_1, pred_2, ...]}
    assert len(all_responses_list) == len(all_paper_ids_list), f"len(all_responses_list): {len(all_responses_list)} vs len(all_paper_ids_list): {len(all_paper_ids_list)}"
    for id, weakness_list in zip(all_paper_ids_list, all_responses_list):
        if id not in res_dict:
            res_dict[id] = [weakness_list]
        else:
            res_dict[id].append(weakness_list)
    
    
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
    api_name_save = api_name_save + "-tables" if args.tables else api_name_save
    api_name_save = api_name_save + "-figures" if args.figures else api_name_save
    api_name_save = api_name_save + "-" + f"num_{str(args.max_image_num)}"
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
        "figures": args.figures,
        "tables": args.tables,
        "max_image_num": args.max_image_num,
        "max_word_len": args.max_word_len,
        "seed": args.seed
    }
    with open(os.path.join(save_dir, "stat.json"), "w") as f:
        json.dump(state_dict, f, indent=4)
        
    # save all the paper ids
    all_paper_ids = list(all_paper_ids)
    with open(os.path.join(save_dir, "all_paper_ids.txt"), "w") as f:
        for item in all_paper_ids:
            f.write(f"{item}\n")
    
    
    print("="*20)
    print(f"Model: {args.api_name}")
    print(f"Total instances: {len(all_subfolders)}")
    print(f"Pick instances: {args.pick_num}")
    print(f"Split: {args.split}")
    print(f"Figures: {args.figures}")
    print(f"Tables: {args.tables}")
    print(f"Max image num: {args.max_image_num}")
    print(f"Input max word len: {args.max_word_len}")
    print(f"Time cost: {(ed_time - st_time).seconds / 60} minutes")
    print("="*20)
    print(f"Results saved to: {save_dir}")
    
    
if __name__ == "__main__":
    main()

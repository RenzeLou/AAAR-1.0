'''
random baseline
for experiment, randomly pick 5 sentences from input context
for explanation, directly copy the input experiment
'''
import random
from datetime import datetime
import os
import json
import argparse
from tqdm import tqdm

from prompt_templates import Exp_eval, Exp_explanation_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-4-1106-preview", help="the name of the openai model to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask2_experiment_human_anno/final_data", help="the directory save the data.")
    parser.add_argument("--save_dir", type=str, default="./subtask2_experiment_human_anno/eval_results", help="the directory to save the unified instances.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
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
        # pure text data
        eval_data.append(data_text)
    print(f"==> total instances: {len(eval_data)}\n")
    
    
    # sometimes the api_name will be like a path (for the open source LLM), e.g., `mistralai/Mistral-7B-Instruct-v0.1`
    # replace `/` with `_` to avoid the error of creating a directory with the name of the path
    # api_name_save = args.api_name.replace("/", "_")
    # api_name_save = api_name_save + "-" + str(args.max_word_len)
    # api_name_save = api_name_save + "-oracle" if args.oracle else api_name_save
    # api_name_save = api_name_save + "-images" if args.images else api_name_save
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    # api_name_save = api_name_save + "---" + current_time 
    api_name_save = "random_baseline"
    for instance in tqdm(eval_data):
        eval_ins, all_images_str = instance, None
        paper_id = eval_ins["id"]
        target_dir = os.path.join(args.save_dir, f"{api_name_save}", paper_id)
        os.makedirs(target_dir, exist_ok=True)
        
        input_text = eval_ins["input"]
        gt_experiment = eval_ins["output"]["What experiments do you suggest doing?"]
        gt_explanation = eval_ins["output"]["Why do you suggest these experiments?"]
        
        pick_num = min(5, len(input_text))
        pred_experiment = random.sample(input_text, pick_num)
        assert isinstance(pred_experiment, list), f"expect a list, but got: {pred_experiment}"
        
        pred_explanation = gt_experiment
        assert isinstance(pred_explanation, list), f"expect a string, but got: {pred_explanation}"
        
        # save all the prediction list (will calculate metrics later)
        save_result_dict = {
            "id": paper_id,
            "output": eval_ins["output"],
            "oracle": True,
            "predicton": {
                "What experiments do you suggest doing?": pred_experiment,
                "Why do you suggest these experiments?": pred_explanation
            },
            "time": current_time,
            "input": input_text
        }
        with open(os.path.join(target_dir, "eval_results.json"), "w") as f:
            json.dump(save_result_dict, f, indent=4)
    
    

    print(f"Results saved to: {os.path.join(args.save_dir, api_name_save)}")
    
    
if __name__ == "__main__":
    main()

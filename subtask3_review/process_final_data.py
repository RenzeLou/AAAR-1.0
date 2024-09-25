import json
import os
import argparse
import random

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./all_others/NeurIPS_2021")  # 
    parser.add_argument("--target_dir", type=str, default="./subtask3_review_final")
    parser.add_argument("--text_data_path", type=str, default="./subtask3_review_processed")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    conf_name = args.root_dir.split("/")[-1]
    target_dir = os.path.join(args.target_dir, conf_name)
    os.makedirs(target_dir, exist_ok=True)
    text_data_path = os.path.join(args.text_data_path, conf_name)
    if not os.path.exists(text_data_path):
        raise FileNotFoundError(f"{text_data_path} not found")
    
    # for loop each subfolder the root_dir, not the subsubfolders
    all_subfolders = os.listdir(args.root_dir)
    print(f"==> {len(all_subfolders)} subfolders found under {args.root_dir}")
    cnt = 0
    for subfolder in tqdm(all_subfolders):
        paper_id = subfolder
        paper_dir = os.path.join(args.root_dir, subfolder)
        # if the paper dir is a file instead of a folder, then skip
        if not os.path.isdir(paper_dir):
            continue
        # move the paper dir, including all the files, subfolders under it to the target_dir
        target_paper_dir = os.path.join(target_dir, paper_id)
        os.makedirs(target_paper_dir, exist_ok=True)
        os.system(f"mv {paper_dir}/* {target_paper_dir}")
        
        text_paper_dir = os.path.join(text_data_path, subfolder)
        text_data_file = os.path.join(text_paper_dir, "data_text.json")
        # copy the text data file to the target dir
        os.system(f"cp {text_data_file} {target_paper_dir}")
        
        text_paper_dir_images = os.path.join(text_paper_dir, "images")  # old images
        text_paper_dir_tables = os.path.join(text_paper_dir, "tables")  # old tables
        if os.path.exists(text_paper_dir_images):
            taerget_paper_dir_arxiv = os.path.join(target_paper_dir, "arxiv_source")
            taerget_paper_dir_arxiv_images = os.path.join(taerget_paper_dir_arxiv, "images")
            os.makedirs(taerget_paper_dir_arxiv_images, exist_ok=True)
            os.system(f"mv {text_paper_dir_images}/* {taerget_paper_dir_arxiv_images}")
        if os.path.exists(text_paper_dir_tables):
            taerget_paper_dir_arxiv = os.path.join(target_paper_dir, "arxiv_source")
            taerget_paper_dir_arxiv_tables = os.path.join(taerget_paper_dir_arxiv, "tables")
            os.makedirs(taerget_paper_dir_arxiv_tables, exist_ok=True)
            os.system(f"mv {text_paper_dir_tables}/* {taerget_paper_dir_arxiv_tables}")
        
        cnt += 1
    
    print(f"==> {cnt} papers are moved from {args.root_dir} to {target_dir}")
        
        

if __name__ == "__main__":
    main()
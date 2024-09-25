

import argparse
import os
import random

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./NeurIPS_2022")  
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    all_subfolders = os.listdir(args.root_dir)
    print(f"==> {len(all_subfolders)} subfolders found under {args.root_dir}")
    cnt = 0
    arxiv_cnt = 0
    for subfolder in tqdm(all_subfolders):
        paper_id = subfolder
        paper_dir = os.path.join(args.root_dir, subfolder)
        # if the paper dir is a file instead of a folder, then skip
        if not os.path.isdir(paper_dir):
            continue
        # remove the pdf file under the paper dir, only paper dir, not the subfolders under paper dir
        for file in os.listdir(paper_dir):
            if file.endswith(".pdf"):
                os.remove(os.path.join(paper_dir, file))
                cnt += 1
            # also remoove the subfolder named 'arxiv_source'
            if file == "arxiv_source":
                if os.path.isdir(os.path.join(paper_dir, file)):
                    os.system(f"rm -r {os.path.join(paper_dir, file)}")
                    arxiv_cnt += 1 
    
    print(f"==> {cnt} pdf files are removed from {args.root_dir}")
    print(f"==> {arxiv_cnt} arxiv_source folders are removed from {args.root_dir}")
    

if __name__ == "__main__":
    main()
    
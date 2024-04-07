'''
use this script for seperating the context before and after the "experiment" section of a paper
'''
import random
import re
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


def red_tex(file):
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


def rsplit_by_pattern(text, pattern, maxsplit=1):
    """
    Splits 'text' by the last occurrences of 'pattern' based on 'maxsplit'.
    
    Args:
        text (str): The text to be split.
        pattern (str): The regex pattern to split on.
        maxsplit (int): The maximum number of splits; must be at least 1.
        
    Returns:
        list: Parts of the string after splitting by 'pattern'.
    """
    # Check if maxsplit is valid
    if maxsplit < 1:
        raise ValueError("maxsplit must be at least 1")
    
    # Find all matches of the pattern
    matches = list(re.finditer(pattern, text))
    num_matches = len(matches)
    
    # If no matches or maxsplit=0, return the original text as a single-element list
    if not matches or maxsplit == 0:
        return [text]
    
    # Calculate the number of splits, which cannot exceed the number of matches
    num_splits = min(num_matches, maxsplit)
    
    # Initialize an empty list to store the results
    parts = []
    
    # Start index for slicing is the end of the string
    start_index = len(text)
    
    # Iterate through the required number of matches from the end
    for match in matches[-num_splits:]:
        parts.append(text[match.end():start_index])
        start_index = match.start()
    
    # Add the remaining part of the string (if any)
    parts.append(text[:start_index])
    
    # Reverse to maintain the original order
    parts.reverse()
    
    return parts

def search_last(pattern, text):
    """
    Searches for the last occurrence of a regex pattern in a given text.
    
    Args:
        pattern (str): The regex pattern to search for.
        text (str): The text to search within.
        
    Returns:
        A re.Match object of the last match if found, else None.
    """
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return matches[-1]       

def find_expriment_location(tex_content):
    '''
    find the line number of the "experiment" section in the tex contents
    
    return a list of line number
    note that, there could be multiple experiment sections in one paper, we just return the first one
    '''
    
    locations = []
    # to check if there is a "experiment" section in this line
    # any section format, e.g., \section{Experiment}, \subsection{Experiment}, \subsubsection{Experiment}
    # can have more characters after "experiment", e.g., \section{Experiment and Results}
    # case insensitive, no matter "experiment" or "Experiment"
    exp_sec_pattern = r"\\(sub)*section\{.*experiment.*\}"
    for i, line in enumerate(tex_content):
        if re.search(exp_sec_pattern, line, re.IGNORECASE):
            locations.append(i)
        
    return locations


def insert_other_sections(main_tex_content, content_list):
    combined_tex = []
    for line in main_tex_content:
        if re.search(r"\\input\{.*\}", line):
            # find the file name in this line
            # for example, sections/5ExperimentalSetup or 5ExperimentalSetup
            # i.e., the string inside the \input{}
            f_name = re.search(r"\\input\{(.*)\}", line).group(1)
            # for sections/5ExperimentalSetup, we need to remove the "sections/" part
            f_name = f_name.split("/")[-1]
            # make sure f_name is a .tex file
            f_name = f_name + ".tex" if not f_name.endswith(".tex") else f_name
            
            # find this file in the content_list
            found = False
            for file_name, content in content_list:
                if file_name == f_name:
                    combined_tex.extend(content)
                    found = True
                    break
                
            if not found:
                # just add the oringinal line
                combined_tex.append(line)
        else:
            combined_tex.append(line)

    return combined_tex
    
    
def combine_tex(tex_files:list):
    '''
    combine multiple tex files into one
    '''
    # first find the main.tex (usually has "\documentclass")
    content_list = []
    main_idx = -1
    main_tex_content = None
    main_tex_name = None
    for i, file in enumerate(tex_files):
        content_lines = red_tex(file)
        if content_lines is None:
            continue
        file_name = os.path.basename(file)
        content_list.append((file_name, content_lines))  # file_name, file_content
        for line in content_lines:
            # if this line contains "\documentclass", then this is the main.tex
            if re.search(r"\\documentclass", line):
                if main_tex_content is not None:
                    # if we already find the main.tex, but there is another main.tex, then raise an error
                    return None, "multiple main.tex files (have \\documentclass) founded in one project (you can try to manually delete those duplicated files): {} <==> {}".format(main_tex_name, file_name)
                main_idx = i
                main_tex_content = content_lines
                main_tex_name = file_name
                # then delete this file from the content_list
                # delete the last element
                # content_list.pop()
                break
    if main_tex_content is None:
        # if there is no main.tex file, then raise an error
        return None, "no main.tex file founded in this project"
    
    
    # since the other tex files might also have "\input{}" to refer to other tex files
    # we iterate all the tex files in the content_list, try to make sure all the other tex files are also self-complete
    ITER_NUM = 3  # empirically set to 3
    for _ in range(ITER_NUM):
        # since only do it once will still result in some missing files, we try multiple times
        new_content_list = []
        for current_tex in content_list:
            current_tex_name, current_tex_content = current_tex
            combined_current_tex = insert_other_sections(current_tex_content, content_list)
            # replace the current_tex_content with the combined_current_tex
            new_content_list.append((current_tex_name, combined_current_tex))
        content_list = new_content_list
    
    # combine all the tex files into one, based on the main.tex
    # for example, if the main.tex has a line "\input{sections/5ExperimentalSetup}" or "\input{5ExperimentalSetup}", then try to find this file in the content_list
    # if not found, just skip this line
    combined_tex = insert_other_sections(main_tex_content, content_list)
            
    return combined_tex, "success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./acl_papers", help="the directory to save the downloaded papers.")
    parser.add_argument("--target_dir", type=str, default="./subtask2_expriment", help="the directory to save the processed paper data.")
    # parser.add_argument("--ins_per_paper", type=int, default=1, help="the number of instances to generate per paper. for an eval set, we can use one paper for multiple instances, but not for training set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    total_paper, final_used_paper = 0, 0
    suitable_paper_num = 0  # note not every crawled paper is suitable for this task, for example, we omit those papers with complex latex structure and has no explicit "experiment" section

    # for each subdir under root_dir
    for subfolder in tqdm(os.listdir(args.root_dir)):
        total_paper += 1
        paper_id = subfolder
        meta_path = os.path.join(args.root_dir, subfolder, f"{subfolder}_metadata.json")
        source_package_path = os.path.join(args.root_dir, subfolder, f"{subfolder}_source.tar.gz")
        assert os.path.exists(source_package_path), f"Source package path does not exist: {source_package_path}"
        # extract the .tar.gz file under the subdir
        source_dir = os.path.join(args.root_dir, subfolder, f"{subfolder}_source")
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            # os.system(f"tar -xvf {source_package_path} -C {source_dir}")
            subprocess.run(['tar', '-xvf', source_package_path, '-C', source_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        
        assert os.path.exists(source_dir), f"running time error, there should be a source directory: {source_dir}"
        
        # find the main.text under the source_dir
        # first iterate all the files (including under subfolders) to see if there are multiple .tex files
        tex_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".tex"):
                    tex_files.append(os.path.join(root, file))
        
        tex_content = None
        
        if len(tex_files) == 0:
            # TODO: why there is no .tex file in this paper?
            print(f"==> {subfolder} has no tex file, skip.")  ## Human annotation needed, delete it when doing large-scale automatic annotation
            continue
        elif len(tex_files) > 1:
            # try to combine multiple tex files into one
            combined_tex, e_message = combine_tex(tex_files)
            if combined_tex is None:
                print(f"==> {subfolder} {e_message}, skip.")  ## Human annotation needed, delete it when doing large-scale automatic annotation
                continue
            else:
                # print("==> {} paper need to combine multiple tex files".format(subfolder))
                tex_content = combined_tex  # combined_tex is a list of lines
        else:
            # only one tex file
            main_tex = tex_files[0]
            # read this tex file
            # with open(main_tex, "r") as f:
            #     tex_content = f.readlines()
            tex_content = red_tex(main_tex)
            if tex_content is None:
                # if the tex file cannot be read, skip this paper
                print(f"==> {subfolder} source tex file cannot be read, skip.")  ## Human annotation needed, delete it when doing large-scale automatic annotation
                continue
        
        clean_tex = tex_cleaning(tex_content)
        locations = find_expriment_location(clean_tex)
        if len(locations) == 0:
            # mean there is no "experiment" section in this paper, skip this paper
            print(f"==> {subfolder} has no experiment section, skip.") ## Human annotation needed, delete it when doing large-scale automatic annotation 
            continue
        suitable_paper_num += 1
        
        # seperate the context before and after the "experiment" section
        loc = locations[0]  # TODO: currently only use the first "experiment" section
        context_before_exp = clean_tex[:loc]
        context_after_exp = clean_tex[loc:]
        save_content = {
            "context_before_exp": context_before_exp,
            "context_after_exp": context_after_exp
        }
        
        target_path = os.path.join(args.target_dir, subfolder)
        os.makedirs(target_path, exist_ok=True)
        # save the extracted equations of this paper
        with open(os.path.join(target_path, "context_before_after_exp.json"), "w") as f:
            json.dump(save_content, f, indent=4)
        # meanwhile, copy the meta file to the target directory, if meta file exists
        if os.path.exists(meta_path):
            shutil.copy(meta_path, os.path.join(target_path, f"{subfolder}_metadata.json"))
        # also copy the pdf file to the target directory, if pdf file exists
        pdf_file = os.path.join(args.root_dir, subfolder, f"{subfolder}.pdf")
        if os.path.exists(pdf_file):
            shutil.copy(pdf_file, os.path.join(target_path, f"{subfolder}.pdf"))
        # also save the cleaned tex content
        with open(os.path.join(target_path, "cleaned_tex.tex"), "w") as f:
            f.writelines(clean_tex)
        
        final_used_paper += 1
            
    
    print("="*50)
    print("Totally {} papers under {}".format(total_paper, args.root_dir))
    print("Among them, {} papers are suitable for this task (exclude no tex paper)".format(suitable_paper_num))
    print("Finally used {} papers to generate instances (some has no experiments section)".format(final_used_paper))
    print("="*50)
    print("Data are saved under {}".format(args.target_dir))
        
    
    
if __name__ == "__main__":
    main()
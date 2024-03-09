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

START_PATTERN = r"\\\[|\\begin{equation.*?}|\\begin{align.*?}|\\begin{gather.*?}|\\begin{multline.*?}|\\begin{flalign.*?}|\\begin{split}|\\begin{cases}|\\begin{array}"
END_PATTERN = r"\\\]|\\end{equation.*?}|\\end{align.*?}|\\end{gather.*?}|\\end{multline.*?}|\\end{flalign.*?}|\\end{split}|\\end{cases}|\\end{array}"


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

def check_equation_start_end_match(tex_content, locations):
    # double check whether all the (start, end) pairs are matched, e.g., "\[" should be matched with "\]"
    error_case = []  # save the error cases: for debugging
    for start_line,end_line in locations:
        start = tex_content[start_line]
        end = tex_content[end_line]
        # extract the start and end equation code
        start_code = re.search(START_PATTERN, start).group()
        end_code = re.search(END_PATTERN, end).group()
        # check whether the start and end code are matched
        if start_code == r"\[":
            if end_code != r"\]":
                error_case.append((start_code, end_code))
                return False, error_case
        elif start_code == r"\begin{equation}":
            if end_code != r"\end{equation}":
                error_case.append((start_code, end_code))
                return False, error_case
        elif start_code == r"\begin{align}":
            if end_code != r"\end{align}":
                error_case.append((start_code, end_code))
                return False, error_case
        elif start_code == r"\begin{gather}":
            if end_code != r"\end{gather}":
                error_case.append((start_code, end_code))
                return False, error_case
        
    return True, error_case


# def find_equation_location(tex_content):
#     '''
#     for a given tex content, find the location (start, end line number) of all the equations
#     return a list of tuples, each tuple is (start_line, end_line)
#     '''   
#     # a function that can return True if this line contains any "equation begining code"
#     # equation begin code is: "\[", "\begin{equation}", "\begin{align}", "\begin{gather}", and so on
#     check_equation_begin = lambda x: bool(re.search(START_PATTERN, x))
#     check_equation_end = lambda x: bool(re.search(END_PATTERN, x))
#     start_line, end_line = -1, -1
#     locations = []
#     for i, line in enumerate(tex_content):
#         if check_equation_begin(line):
#             start_line = i
#         elif check_equation_end(line):
#             end_line = i
#             if start_line != -1:
#                 locations.append((start_line, end_line))
#                 start_line, end_line = -1, -1
#             else:
#                 # raise RuntimeError(f"end_line: {end_line} does not have a start line")
#                 print(f"end_line: {tex_content[end_line]} does not have a start line")
#                 return [], False
    
#     check_flag, error_cases = check_equation_start_end_match(tex_content, locations)  
#     try:   
#         assert check_flag
#     except AssertionError:
#         print(f"Error: start and end equation code does not match, error cases: {error_cases}")
        
#     return locations, check_flag

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

'''
v2 version to handle nested equations
'''
def find_equation_location(tex_content):
    check_equation_begin = lambda x: bool(re.search(START_PATTERN, x))
    check_equation_end = lambda x: bool(re.search(END_PATTERN, x))
    
    stack = []
    locations = []
    for i, line in enumerate(tex_content):
        if check_equation_begin(line):
            stack.append(i)
        elif check_equation_end(line) and stack:
            start_line = stack.pop(0)  # Assume the first in is the start of the outermost equation.
            if not stack:  # Stack empty means all nested starts have been closed.
                locations.append((start_line, i))
        elif check_equation_end(line) and not stack:
            print(f"end_line at {i} does not have a start line")
            return [], False
    
    # In case there are unmatched beginnings after processing all lines
    if stack:
        # print("Warning: Some equations did not have an end.")
        return [], False

    check_flag = True   
    # check_flag, error_cases = check_equation_start_end_match(tex_content, locations)
    # try:
    #     assert check_flag
    # except AssertionError:
    #     print(f"Error: start and end equation code does not match, error cases: {error_cases}")
        
    return locations, check_flag

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


# def if_nested(equation: str):
#     '''
#     Check whether the equation is a nested equation, i.e., has multiple equation environments
#     '''
#     # Find all start and end tags
#     start_tags = list(re.finditer(START_PATTERN, equation))
#     end_tags = list(re.finditer(END_PATTERN, equation))
    
#     # Early exit if no tags are found
#     if not start_tags or not end_tags:
#         return False
    
#     # Counters for open environments
#     open_envs = 0
    
#     # Combined list of all tags sorted by their occurrence in the string
#     all_tags = sorted(start_tags + end_tags, key=lambda m: m.start())
    
#     for tag in all_tags:
#         if re.match(START_PATTERN, tag.group(0)):
#             open_envs += 1
#         else:
#             open_envs -= 1
            
#             # Found a nested equation if open environments exceed 1
#             if open_envs > 1:
#                 return True
    
#     return False
    
def how_many_nested(equation: str):
    '''
    Check whether the equation is a nested equation, i.e., has multiple equation environments
    return how many nests this equation has
    '''
    # if there are multiple matched start patterns or end patterns, then it's a nested equation
    number_of_start = len(re.findall(START_PATTERN, equation))
    number_of_end = len(re.findall(END_PATTERN, equation))
    return min(number_of_start, number_of_end) - 1  # minus 1 because the outermost equation is not considered as a nested equation


def process_equation(tex_content, locations):
    '''
    for each location, get one instance (input: the context before and after the equation; output: the equation)
    '''
    ins_list = []  # the instances list for this paper
    for start_line, end_line in locations:
        start = tex_content[start_line]
        end = tex_content[end_line]
        # get the context before and after the equation
        context_before = "".join(tex_content[:start_line])
        context_after = "".join(tex_content[end_line+1:])
        equation = "".join(tex_content[start_line:end_line+1])

        # consider the nested equations, for start line, split only the first match
        start_context, start_equation_part = re.split(START_PATTERN, start, maxsplit=1)
        start_notation = re.search(START_PATTERN, start).group()
        # start_equation_part = start_notation + start_equation_part  # TODO: whether it's needed to add equation environment?
        
        # consider the nested equations, for end line, split only the latest match (the most right one)
        end_equation_part, end_context = rsplit_by_pattern(end, END_PATTERN, maxsplit=1)
        end_notation = search_last(END_PATTERN, end).group()
        # end_equation_part = end_equation_part + end_notation  # TODO: whether it's needed to add equation environment?
        
        context_before += start_context
        context_after = end_context + context_after
        equation = start_equation_part + equation + end_equation_part
        # equation = r"\[" + equation + r"\]"  # TODO: whether it's needed to add equation environment?
        equation = equation.strip() # remove the leading and trailing spaces
        
        ins = {
            "context_before": context_before,
            "context_after": context_after,
            "equation": equation,
            "location": (start_line, end_line),
            "nest_num": how_many_nested(equation)
        }
        
        ins_list.append(ins)
    
    return ins_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./acl_papers", help="the directory to save the downloaded papers.")
    parser.add_argument("--target_dir", type=str, default="./subtask1_equation", help="the directory to save the processed paper data.")
    parser.add_argument("--ins_per_paper", type=int, default=1, help="the number of instances to generate per paper. for an eval set, we can use one paper for multiple instances, but not for training set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    total_paper, final_used_paper, equation_num = 0, 0, 0
    suitable_paper_num = 0  # note not every crawled paper is suitable for this task, for example, we omit those papers with complex latex structure
    nest_eq_num_list =  []  # save the number of nested equations for each paper
    
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
        
        if len(tex_files) > 1:
            # TODO: here, simply skip those papers with multiple .tex files ==> only process those simple structure paper projects
            continue
        elif len(tex_files) == 0:
            # TODO: why there is no .tex file in this paper?
            continue
        else:
            main_tex = tex_files[0]
            # read this tex file
            # with open(main_tex, "r") as f:
            #     tex_content = f.readlines()
            tex_content = red_tex(main_tex)
            if tex_content is None:
                # if the tex file cannot be read, skip this paper
                continue
            
            suitable_paper_num += 1
            
            clean_tex = tex_cleaning(tex_content)
            
            locations, check_flag = find_equation_location(clean_tex)
            
            
            if not check_flag:
                # if there is a latex grammar error:
                # e.g., the start and end equation code does not match, just skip this paper to avoid further data errors (e.g., wrong equation extracted)
                # TODO: why some equation have no end environment?
                continue
            if len(locations) == 0:
                # mean there is no equation in this paper, skip this paper
                continue
            
            random.shuffle(locations)
            # get the top args.ins_per_paper equations
            target_locations = locations[:min(args.ins_per_paper, len(locations))]
            
            ins_list = process_equation(clean_tex, target_locations)
            # count how many nested eq this paper's instances have
            nest_eq_num_list.extend([ins["nest_num"] for ins in ins_list])
           
            target_path = os.path.join(args.target_dir, subfolder)
            os.makedirs(target_path, exist_ok=True)
            # save the extracted equations of this paper
            with open(os.path.join(target_path, "equations.json"), "w") as f:
                json.dump(ins_list, f, indent=4)
            # meanwhile, copy the meta file to the target directory, if meta file exists
            if os.path.exists(meta_path):
                shutil.copy(meta_path, os.path.join(target_path, f"{subfolder}_metadata.json"))
                # os.rename(meta_path, os.path.join(target_path, f"{subfolder}_metadata.json"))
            # also save the cleaned tex content
            with open(os.path.join(target_path, "cleaned_tex.tex"), "w") as f:
                f.writelines(clean_tex)
            
            final_used_paper += 1
            equation_num += len(ins_list)
            
    
    print("="*50)
    print("Totally {} papers under {}".format(total_paper, args.root_dir))
    print("Among them, {} papers are suitable for this task (exclude no tex paper and multi tex paper)".format(suitable_paper_num))
    print("Finally used {} papers to extract equations and generate instances (some has no equations or wrong equation formats)".format(final_used_paper))
    print("="*50)
    print("Extract at most {} equations per paper".format(args.ins_per_paper))
    print("Totally {} equations are extracted".format(equation_num))
    print("="*50)
    print("Totally {} instances are nested (nest_num > 0)".format(len([i for i in nest_eq_num_list if i > 0])))
    print("AVG nest num per instance: {:.2f}".format(sum(nest_eq_num_list) / len(nest_eq_num_list)))
    print("="*50)
    print("Extracted equations are saved under {}".format(args.target_dir))
        
    
    
if __name__ == "__main__":
    main()
    # pattern = r'\\end{equation}|\\end{align}|\\end{gather}'
    # text = "This is an equation \\begin{align} ssssss \\begin{equation}E=mc^2\\end{equation} tttt \\end{align} and here is text after."
    # print(if_nested(text))
    # text = "This is an equation \\begin{align} ssssss \\end{align} and here is text after."
    # print(if_nested(text))
    # # end_equation_part, end_context = rsplit_by_pattern(text, pattern)
    # end_notation_1 = re.search(pattern, text).group()
    # end_notation = search_last(END_PATTERN, text).group()

    # print("Before the last pattern match:\n", end_equation_part)
    # print("After the last pattern match:\n", end_context)
    # print("The last pattern match (wrong):\n", end_notation_1)
    # print("The last pattern match:\n", end_notation)
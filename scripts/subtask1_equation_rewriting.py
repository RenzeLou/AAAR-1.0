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

from chat_completion import openai_chat_completion
from prompt_templates import Equation_Filtering, EquationRewrite_Difficult, EquationRewrite_Easy, Equation_Generation


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 3600
    temperature: float = 1.2  # TODO: is it good to encourage diverse wrong equations?
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    # stop: Optional[List[str]] = dataclasses.field(default_factory=default_stop)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

def generate_new_variable(exclusions):
    # Generate a new variable that is not in the exclusions list
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    new_var = random.choice(alphabet)
    while new_var in exclusions:
        new_var += random.choice(alphabet)
    return new_var

def random_rewriting(ori_equation, option=1):
    '''
    given an equation, randomly modify one part of this equation, including:
    1. change the operator
    2. change the variable or notations
    3. exchange the position of two variables or notations
    4. change the constant (if has)
    
    select one from the above 4 options, and return the modified equation
    '''
    new_equation = copy.deepcopy(ori_equation)
    if option == 1:
        # Define three groups of operators
        group1 = [r'\+', '-', r'\*', '/', r'\\times', r'\\div']  # Arithmetic operators
        group2 = [r'\\pm', r'\\mp', r'\\cup', r'\\cap', r'\\prod', r'\\sum']  # Misc operators
        group3 = [r'>', r'<', r'\\geq', r'\\leq', r'\\neq', r'==']  # Boolean operators

        # Create a list of all operators and define a pattern to match them
        all_operators = group1 + group2 + group3
        operators_pattern = r'(?<!\\)(?:' + '|'.join(all_operators) + ')'

        # Find all operators in the equation
        matches = re.findall(operators_pattern, new_equation)

        # If there are matches, proceed to randomly replace one
        if matches:
            chosen_operator = random.choice(matches)

            # Determine which group the chosen operator belongs to
            group_selected = None
            for group in [group1, group2, group3]:
                if chosen_operator in group:
                    group_selected = group
                    break

            # Pick a new operator from the same group (excluding the chosen one)
            if group_selected is not None:
                # Adjust pattern if it is a LaTeX special character
                if chosen_operator.startswith('\\'):
                    chosen_operator_pattern = re.escape(chosen_operator)
                else:
                    chosen_operator_pattern = chosen_operator

                replacement_operators = list(set(group_selected) - set([chosen_operator]))
                replacement_operator = random.choice(replacement_operators)

                # Replace the chosen operator with the new one in the equation, just once
                new_equation = re.sub(chosen_operator_pattern, replacement_operator, new_equation, count=1)
        
    elif option == 2:
        # change the variable or notations
        # Define a pattern that matches variables like "w_i" but not functions or LaTeX commands
        var_pattern = r'\b[a-zA-Z]+_?{?[a-zA-Z0-9]*}?\b'
        matches = re.findall(var_pattern, new_equation)
        
        # Exclude LaTeX commands and environments
        excluded_keywords = set(['equation', 'begin', 'end', 'leq', 'geq', 'prod', 'sum', 'times', 'div', 'pm', 'mp'])
        variables = [match for match in matches if all([not match.startswith(cmd) for cmd in excluded_keywords]) and len(match) > 1]

        if variables:
            chosen_variable = random.choice(variables)
            # Extract all variables to avoid collision while naming
            all_variables = set(variables)
            unused_variable = generate_new_variable(all_variables)
            # Replace chosen variable with a new one
            new_equation = re.sub(r'\b' + re.escape(chosen_variable) + r'\b', unused_variable, new_equation)
        
    elif option == 3:
        # exchange the position of two variables or notations
        var_pattern = r'\b[a-zA-Z]+_?{?[a-zA-Z0-9]*}?\b'
        matches = re.findall(var_pattern, new_equation)

        # Exclude LaTeX commands and environments
        excluded_keywords = {'equation', 'begin', 'end', 'leq', 'geq', 'prod', 'sum', 'times', 'div', 'pm', 'mp'}
        variables = [match for match in matches if all(not match.startswith(cmd) for cmd in excluded_keywords)]
        
        if len(variables) >= 2:
            var1, var2 = random.sample(variables, 2)
            
            def swap_variables(match):
                # Check which group matched and get the appropriate subgroups
                groups = match.groups()
                if groups[0] is not None:  # var1 is first
                    before, var1_match, between, var2_match, after = groups[:5]
                else:  # var2 is first
                    before, var2_match, between, var1_match, after = groups[5:]
                
                # Swap the variables
                return f"{before}{var2_match}{between}{var1_match}{after}"
            
            # Build pattern by considering variables may have text between them
            pattern = fr'(.*)\b({re.escape(var1)})\b(.*?)\b({re.escape(var2)})\b(.*)|' \
                      fr'(.*)\b({re.escape(var2)})\b(.*?)\b({re.escape(var1)})\b(.*)'
            
            new_equation, num_subs = re.subn(pattern, swap_variables, new_equation, count=1)
    
    elif option == 4:
        # change the constant (if has)
        constant_pattern = r'\b\d+\b'
        matches = re.findall(constant_pattern, new_equation)
        if matches:
            chosen_constant = random.choice(matches)
            # to make subtle changes, the new constant should be close to the original one (+1, -1)
            new_constant = str(int(chosen_constant) + random.choice([-1, 1]))
            new_equation = re.sub(re.escape(chosen_constant), new_constant, new_equation, count=1)
    
    else:
        raise ValueError(f"Unknown option: {option}")
    
    return new_equation


def simplify_equation(equation:str):
    '''
    used for simplifying the equation:
    remove all the redundant commands that will not affect the final compiled equation
    such as the equation environment, equation number, label, and citation, etc.
    '''
    START_PATTERN = r"\\\[|\\begin{equation.*?}|\\begin{align.*?}|\\begin{gather.*?}|\\begin{multline.*?}|\\begin{flalign.*?}|\\begin{split}|\\begin{cases}|\\begin{array}"
    END_PATTERN = r"\\\]|\\end{equation.*?}|\\end{align.*?}|\\end{gather.*?}|\\end{multline.*?}|\\end{flalign.*?}|\\end{split}|\\end{cases}|\\end{array}"
    # Patterns to remove other common redundant commands
    REDUNDANT_COMMANDS = [
        r"\\label{.*?}",
        r"\\cite{.*?}",
        r"\\ref{.*?}",
        r"\\setlength{.*?}{.*?}",
        r"\\hspace{.*?}",
        r"\\vspace{.*?}",
        r"\\nonumber",
        r"\\notag",
        r"\\tag{.*?}",  # Tags for manual numbering/labeling
        r"\\allowdisplaybreaks",  # This does not affect rendering inside a single equation
        r"\\belowdisplayskip",
        r"\\abovedisplayskip",
        r"\\displaybreak",  
        r"\\intertext{.*?}",  
        r"\\shortintertext{.*?}",
    ]
    
    # Clean the equation environment around the equation
    equation = re.sub(START_PATTERN, "", equation)
    equation = re.sub(END_PATTERN, "", equation)

    # Clean all other redundant commands that do not affect the final compiled equation
    for pattern in REDUNDANT_COMMANDS:
        equation = re.sub(pattern, "", equation)
    equation = equation.strip()
    
    # remove some puctuation at the begining or end of the equation
    # equation = re.sub(r"^[^\w\\]+|[^\w\\]+$", "", equation)
    # equation = equation.strip()
    
    return equation


# def get_lhs_of_equation(latex_eq):
#     """
#     Get the left-hand side of the LaTeX equation up to the first equal sign that is not in a subscript or superscript.
#     """
#     # Define a regex pattern for matching nested curly braces
#     nested_braces_pattern = r"\{(?:[^{}]|(?R))*\}"

#     # Replace nested braces content with a placeholder to avoid matching '=' inside them
#     placeholder = "PLACEHOLDER"
#     while True:
#         new_latex_eq, n_subs = re.subn(nested_braces_pattern, placeholder, latex_eq)
#         if n_subs == 0:
#             break
#         latex_eq = new_latex_eq

#     # Find the index of the first '=' not inside braces
#     match = re.search(r"=", latex_eq)
#     if match:
#         # Get the index of the match
#         index_of_equal = match.start()
#         # Get the original left-hand side up to the index found
#         lhs_with_placeholders = latex_eq[:index_of_equal]
        
#         # Count the number of placeholders before the '=' sign
#         n_placeholders = lhs_with_placeholders.count(placeholder)
#         # Replace the placeholders with the original nested content
#         lhs_parts = re.findall(nested_braces_pattern, new_latex_eq)
#         for i in range(n_placeholders):
#             lhs_with_placeholders = lhs_with_placeholders.replace(placeholder, lhs_parts[i], 1)

#         return lhs_with_placeholders

#     # If no '=' symbol is found, return a empty string
#     return ""


def get_lhs_of_equation(latex_eq):
    """
    Get the left-hand side of the LaTeX equation up to the first equal sign that is not in a subscript or superscript.
    """
    # Track the number of open braces
    brace_count = 0
    
    # Iterate through each character in the string
    for i, char in enumerate(latex_eq):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '=' and brace_count == 0:
            # Return everything up to the `=` sign if we are not inside braces
            return latex_eq[:i]
    
    # If no '=' symbol is found, return a empty string
    return ""



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

def save_intermediate_results(all_items, save_file, save_path, message):
    file_name = os.path.basename(save_file)
    file_name = file_name.rsplit(".", 1)[0] + f".{message}.json"
    terminate_save_path = os.path.join(save_path, "terminated_results")
    os.makedirs(terminate_save_path, exist_ok=True)
    with open(os.path.join(terminate_save_path, file_name), "w") as f:
        json.dump(all_items, f, indent=2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, 
                        default=None, help="not recommended; better to set env varaible.")
    parser.add_argument("--api_name", type=str, default="gpt-3.5-turbo-1106", help="the name of the openai model to use.")
    parser.add_argument("--template", type=int, default=1, help="the type of prompt template to use.")
    parser.add_argument("--root_dir", type=str, default="./subtask1_equation", help="the directory to save the downloaded papers.")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the existing files.")
    parser.add_argument("--ins_per_paper", type=int, default=1, help="the number of instances to generate per paper. for an eval set, we can use one paper for multiple instances, but not for training set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--add_left_equation", action="store_true", help="whether to add the left part of the equation to the context.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    openai.api_key = os.getenv("OPENAI_API_KEY") if args.api_key is None else args.api_key
    client = openai.OpenAI()
    decoding_args = OpenAIDecodingArguments()
    if args.template == 1:
        template = EquationRewrite_Easy()
    elif args.template == 2:
        template = EquationRewrite_Difficult()
    elif args.template == 3:
        template = Equation_Generation()
    else:
        raise ValueError(f"Unknown template type: {args.template}")
    
    filter_template = Equation_Filtering()
    
    random.seed(args.seed)
    
    rewrite_ins_num = 0
    origin_total_ins_num = 0
    
    # for each subdir under root_dir
    for subfolder in tqdm(os.listdir(args.root_dir)):
        # paper_id = subfolder
        # meta_path = os.path.join(args.root_dir, subfolder, f"{subfolder}_metadata.json")
        # main_tex = read_tex(os.path.join(args.root_dir, subfolder, "cleaned_tex.tex"))
        # if main_tex is None:
        #     raise ValueError(f"Cannot read the tex file of {subfolder}")
        
        # read the equations.json file
        equation_file = os.path.join(args.root_dir, subfolder, "equations.json")
        with open(equation_file, "r") as f:
            ins_list = json.load(f)
        
        save_folder = os.path.join(args.root_dir, subfolder)    
        save_file = os.path.join(save_folder, "equations_cls.json")
        # check if there is already a file under the subfolder
        if os.path.exists(save_file) and not args.overwrite:
            continue
        
        origin_total_ins_num += len(ins_list)
        
        sample_ins_num = min(args.ins_per_paper, len(ins_list))
        
        ins_list = ins_list[:sample_ins_num]
        
        try:
            # for each instance in the ins_list, create a new instance 
            overall_retry_cnt_same_as_ori = 0
            overall_retry_cnt_gpt4_false = 0
            new_ins_list = []
            for ins in ins_list:
                context_before = ins["context_before"]
                context_after = ins["context_after"]
                equation = ins["equation"]
                gpt_query = {
                    "ori_equation": equation
                }
                ### 1. ask GPT to rewrite the equation
                # wrong_eq_1, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                # wrong_eq_2, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                # wrong_eq_3, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                
                
                
                ### 2. ask GPT to generate the equation from the context
                equation = simplify_equation(equation)  # simplify the equation, to avoid shortcut in the ori equation
                # get the "="'s left part of the equation
                if args.add_left_equation:
                    equation_left_part = get_lhs_of_equation(equation)
                else:
                    equation_left_part = ""
                # equation_left_part either is a string or an empty string
                
                MAX_LEN = 200
                # for context_before, use the last context_max_len words
                context_before = " ".join(context_before.split()[-MAX_LEN:])
                # for context_after, use the first context_max_len words
                context_after = " ".join(context_after.split()[:MAX_LEN])
                gpt_query = {
                    "context_before": context_before,
                    "context_after": context_after,
                    "equation_left_part": equation_left_part
                }
                
                retry_cnt = 0
                retry_cnt_gpt4_filter = 0
                wrong_eq_list = []
                wrong_eq_1 = copy.deepcopy(equation)
                cnt = 0 
                flag = "wrong"
                while wrong_eq_1 == equation or flag != "correct":  # or wrong_eq_1 in wrong_eq_list
                    # wrong equation generation
                    wrong_eq_1, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                    wrong_eq_1 = equation_left_part + wrong_eq_1
                    # count how may times the same equation is generated (which means the LLM do good at equation recovery)
                    if wrong_eq_1 == equation:
                        retry_cnt += 1
                    # let the gpt itself do the filtering
                    # TODO: currectly, I use the same API as the equation generation, as well as the decoding args
                    flag, cost = openai_chat_completion(client, {"equation": wrong_eq_1}, filter_template, decoding_args, model_name=args.api_name)  
                    flag = re.sub(r"[^\w\s]", "", flag) # remove punctuations
                    flag = flag.strip().lower()
                    # assert flag in ["correct", "wrong"], f"Unknown flag: {flag}"
                    print(f"flag: {flag}")
                    if flag == "wrong":
                        retry_cnt_gpt4_filter += 1
                    cnt += 1
                    if cnt > 10:
                        wrong_eq_1 = "None"
                        break
                wrong_eq_list.append(wrong_eq_1)
                
                wrong_eq_2 = copy.deepcopy(equation)
                cnt = 0
                flag = "wrong"
                while wrong_eq_2 == equation or flag != "correct":  # or wrong_eq_2 in wrong_eq_list
                    wrong_eq_2, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                    wrong_eq_2 = equation_left_part + wrong_eq_2
                    if wrong_eq_2 == equation:
                        retry_cnt += 1
                    flag, cost = openai_chat_completion(client, {"equation": wrong_eq_2}, filter_template, decoding_args, model_name=args.api_name)
                    flag = re.sub(r"[^\w\s]", "", flag) # remove punctuations
                    flag = flag.strip().lower()
                    # assert flag in ["correct", "wrong"], f"Unknown flag: {flag}"
                    print(f"flag: {flag}")
                    if flag == "wrong":
                        retry_cnt_gpt4_filter += 1
                    cnt += 1
                    if cnt > 10:
                        wrong_eq_2 = "None"
                        break
                wrong_eq_list.append(wrong_eq_2)
                
                wrong_eq_3 = copy.deepcopy(equation)
                cnt = 0
                flag = "wrong"
                while wrong_eq_3 == equation or flag != "correct":  # or wrong_eq_3 in wrong_eq_list
                    wrong_eq_3, cost = openai_chat_completion(client,gpt_query, template, decoding_args, model_name=args.api_name)
                    wrong_eq_3 = equation_left_part + wrong_eq_3
                    if wrong_eq_3 == equation:
                        retry_cnt += 1
                    flag, cost = openai_chat_completion(client, {"equation": wrong_eq_3}, filter_template, decoding_args, model_name=args.api_name)
                    flag = re.sub(r"[^\w\s]", "", flag) # remove punctuations
                    flag = flag.strip().lower()
                    # assert flag in ["correct", "wrong"], f"Unknown flag: {flag}"
                    print(f"flag: {flag}")
                    if flag == "wrong":
                        retry_cnt_gpt4_filter += 1
                    cnt += 1
                    if cnt > 10:
                        wrong_eq_3 = "None"
                        break
                wrong_eq_list.append(wrong_eq_3)
                
                # clean all the wrong equations
                wrong_eq_1 = simplify_equation(wrong_eq_1)
                wrong_eq_2 = simplify_equation(wrong_eq_2)
                wrong_eq_3 = simplify_equation(wrong_eq_3)
                
                # print("retry count (due to the same as the original equation):", retry_cnt)
                # print("retry count (due to the GPT-4 filtering):", retry_cnt_gpt4_filter)
                # exit()
                overall_retry_cnt_same_as_ori += retry_cnt
                overall_retry_cnt_gpt4_false += retry_cnt_gpt4_filter
                
                
                
                ### 3. randomly make subtle changes to the original equation
                # wrong_eq_1 = copy.deepcopy(equation)
                # cnt = 0
                # while wrong_eq_1 == equation:
                #     wrong_eq_1 = random_rewriting(equation, option=1)
                #     cnt += 1
                #     if cnt > 10:
                #         wrong_eq_1 = None
                #         break
                # wrong_eq_2 = copy.deepcopy(equation)
                # cnt = 0
                # while wrong_eq_2 == equation:
                #     wrong_eq_2 = random_rewriting(equation, option=2)
                #     cnt += 1
                #     if cnt > 10:
                #         wrong_eq_2 = None
                #         break
                # wrong_eq_3 = copy.deepcopy(equation)
                # cnt = 0
                # while wrong_eq_3 == equation:
                #     wrong_eq_3 = random_rewriting(equation, option=3)
                #     cnt += 1
                #     if cnt > 10:
                #         wrong_eq_3 = None
                #         break
                # wrong_eq_4 = copy.deepcopy(equation)
                # cnt = 0
                # while wrong_eq_4 == equation:
                #     wrong_eq_4 = random_rewriting(equation, option=4)
                #     cnt += 1
                #     if cnt > 10:
                #         wrong_eq_4 = None
                #         break
                # three_wrong_eqs = [wrong_eq_1, wrong_eq_2, wrong_eq_3, wrong_eq_4]
                # random.shuffle(three_wrong_eqs)
                # # not none 
                # filtered_wrong_eqs = [eq for eq in three_wrong_eqs if eq is not None]
                # # if less than 3 wrong equations, add the empty string to make it 3
                # filtered_wrong_eqs += [""] * (3 - len(filtered_wrong_eqs))
                # wrong_eq_1, wrong_eq_2, wrong_eq_3 = filtered_wrong_eqs[:3]
                
                
                
                # make a multi-choice question with these 4 options
                options = [equation, wrong_eq_1, wrong_eq_2, wrong_eq_3]
                random.shuffle(options)
                # find the idx of the correct option after shuffling
                correct_idx = options.index(equation)
                ordered_options = f"(A). `{options[0]}`;\n(B). `{options[1]}`;\n(C). `{options[2]}`;\n(D). `{options[3]}`"  # in our dataset, we provide the shuffled order of the options, to ensure others could reproduce the results
                correct_option = chr(ord('A') + correct_idx)
                
                new_ins = copy.deepcopy(ins)
                new_ins["wrong_equations"] = [wrong_eq_1, wrong_eq_2, wrong_eq_3]
                new_ins["options"] = ordered_options
                new_ins["answer"] = correct_option
                new_ins_list.append(new_ins)
        # except openai.error.RateLimitError as e:
        except tenacity.RetryError as e:
            print("==> Error: {}".format(e))
            print("\nOpenAI API rate limit reached. Please increase the waiting/retry times in the tenacity decorator.\n")
            # save_intermediate_results(outputs, args, "RateLimitError")
            # sys.exit(0)  # Exit the program gracefully
        
        # save the new_ins_list to the new file, under the same folder
        with open(save_file, "w") as f:
            json.dump(new_ins_list, f, indent=4)
        
        rewrite_ins_num += len(new_ins_list)
    
    
    print("="*50)
    print(f"Total instances before rewriting: {origin_total_ins_num}")
    print(f"Select the first (at most) {args.ins_per_paper} instances for each paper, and rewrite the equations")
    print(f"Total instances after rewriting: {rewrite_ins_num}")
    print("="*50)
    print(f"Overall retry count (due to the same as the original equation): {overall_retry_cnt_same_as_ori}")
    print(f"Overall retry count (due to the GPT-4 filtering): {overall_retry_cnt_gpt4_false}")
    print(f"AVG retry count (due to the same as the original equation): {overall_retry_cnt_same_as_ori / rewrite_ins_num:.2f}")
    print(f"AVG retry count (due to the GPT-4 filtering): {overall_retry_cnt_gpt4_false / rewrite_ins_num:.2f}")
    
    
    
if __name__ == "__main__":
    main()
    # ori_equation = "\\begin{equation}\n    p_s(s_t | u_{\\leq t}, s_{<t}) = \\prod^{m_{s_t}}_{i=1} P(w_i | w_{< i}, u_{\\leq t}, s_{<t})\n\\end{equation}"
    # print("### Original equation\n", ori_equation, "\n\n")
    # new_latex_equation_1 = random_rewriting(ori_equation, option=1)
    # print("### modify the operator\n", new_latex_equation_1, "\n\n") # Modified equation
    # new_latex_equation_2 = random_rewriting(ori_equation, option=2)
    # print("### modify the variable\n", new_latex_equation_2, "\n\n") # Modified equation
    # new_latex_equation_3 = random_rewriting(ori_equation, option=3)
    # print("### modify the position\n", new_latex_equation_3, "\n\n") # Modified equation
    # new_latex_equation_4 = random_rewriting(ori_equation, option=4)
    # print("### modify the constant\n", new_latex_equation_4, "\n\n") # Modified equation
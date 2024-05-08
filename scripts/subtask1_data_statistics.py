import os
from tqdm import tqdm
import json
import re
os.environ['NLTK_DATA'] = '/scratch/rml6079/nltk_data'
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from pylatexenc.latex2text import LatexNodes2Text


data_path = "./subtask1_equation_unified/1049.human_filter.json"
with open(data_path, "r") as f:
    data_list = json.load(f)
    
answer_eq_len = []
other_eq_len = []
context_before_word_len_list = []
context_before_sent_len_list = []
context_after_word_len_list = []
context_after_sent_len_list = []
option_distribution = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 0
}
for data in tqdm(data_list):
    context_before = data["context_before"]
    context_after = data["context_after"]
    answer = data["answer"]
    answer_idx = ord(answer) - ord('A')
    options_list = data["options_list"]
    
    # count the word and sentence number of context before, context after
    context_before_word_length = len(word_tokenize(context_before))
    context_before_sent_length = len(sent_tokenize(context_before))
    context_after_word_length = len(word_tokenize(context_after))
    context_after_sent_length = len(sent_tokenize(context_after))
    context_before_word_len_list.append(context_before_word_length)
    context_before_sent_len_list.append(context_before_sent_length)
    context_after_word_len_list.append(context_after_word_length)
    context_after_sent_len_list.append(context_after_sent_length)
    
    # get the answer distribution
    option_distribution[answer] += 1
    
    answer_eq = options_list[answer_idx]
    other_eqs = options_list[:answer_idx] + options_list[answer_idx+1:]
    # count the non-white character number of answer equation and other equations
    answer_eq_text = LatexNodes2Text().latex_to_text(answer_eq)
    answer_eq_text = re.sub(r'\s+', '', answer_eq_text)
    answer_eq_len.append(len(answer_eq_text))
    other_eqs_text = [LatexNodes2Text().latex_to_text(eq) for eq in other_eqs]
    other_eqs_text = [re.sub(r'\s+', '', eq) for eq in other_eqs_text]
    other_eq_len.extend([len(eq) for eq in other_eqs_text])
    

print("=== statistics of equation ===")
print("==> <context before> word length: min={}, max={}, mean={}".format(min(context_before_word_len_list), max(context_before_word_len_list), sum(context_before_word_len_list)/len(context_before_word_len_list)))
print("==> <context before> sentence length: min={}, max={}, mean={}".format(min(context_before_sent_len_list), max(context_before_sent_len_list), sum(context_before_sent_len_list)/len(context_before_sent_len_list)))
print("==> <context after> word length: min={}, max={}, mean={}".format(min(context_after_word_len_list), max(context_after_word_len_list), sum(context_after_word_len_list)/len(context_after_word_len_list)))
print("==> <context after> sentence length: min={}, max={}, mean={}".format(min(context_after_sent_len_list), max(context_after_sent_len_list), sum(context_after_sent_len_list)/len(context_after_sent_len_list)))

print("==> <ground truth answer> equation length (in non-white character): min={}, max={}, mean={}".format(min(answer_eq_len), max(answer_eq_len), sum(answer_eq_len)/len(answer_eq_len)))
print("==> <GPT-craft wrong> equation length (in non-white character): min={}, max={}, mean={}".format(min(other_eq_len), max(other_eq_len), sum(other_eq_len)/len(other_eq_len)))

print("==> <answer> distribution: {}".format(option_distribution))
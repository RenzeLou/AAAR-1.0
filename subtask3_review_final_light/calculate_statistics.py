'''
use this script to calculate statistics:

input statistics:
1. # of words of input
2. # of sentences of input
3. # of figures per paper
4. # of tables per paper

output statistics:
1. # of reviews per paper
2. # of item per review
3. # of words per item
4. # of sentences per item

'''
import dataclasses
import random
import re
import requests
import os
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk
from nltk import sent_tokenize, word_tokenize
nltk.download('punkt')

def sentence_num(input_text):
    sentences = sent_tokenize(input_text)
    number_of_sentences = len(sentences)
    return number_of_sentences

def word_num(input_text):
    words = word_tokenize(input_text)
    number_of_words = len(words)
    return number_of_words

def count_input_text(input_text:dict):
    title = input_text["title"]
    main_text = []
    for section in input_text["sections"]:
        main_text.append(str(section["heading"]) + " " + str(section["text"]))
    main_text = "\n".join(main_text)
    abs = input_text["abstractText"]
    
    # make sure every text is a string, avoid bytes-like object
    title = str(title)
    main_text = str(main_text)
    abs = str(abs)
    
    sn = sentence_num(title) + sentence_num(main_text) + sentence_num(abs)
    wn = word_num(title) + word_num(main_text) + word_num(abs)
    return sn, wn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./subtask3_review_final_light")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--target_dir", type=str, default="./subtask3_review_final_light")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)

    conf_list = ["ICLR_2022", "NeurIPS_2021", "NeurIPS_2022", "ICLR_2023"]
    all_subfolders = []
    for conf in conf_list:
        conf_dir = os.path.join(args.root_dir, conf)
        all_files_dirs = os.listdir(conf_dir)
        # make all subfolders the whole path
        all_files_dirs = [os.path.join(conf_dir, x) for x in all_files_dirs]
        all_subfolders.extend(all_files_dirs)
    print(f"==> {len(all_subfolders)} subfolders found under {args.root_dir}")
    cnt = 0
    all_sentence_num, all_word_num, all_image_num, all_table_num = [], [], [], []
    all_review_num, all_weakness_len, all_item_word_num, all_item_sentence_num = [], [], [], []
    for subfolder_path in tqdm(all_subfolders):
        if not os.path.isdir(subfolder_path):
            continue  
        # TODO: also notice the 'target' subfolder (this is produced by pdffigures), should skip or del
        cnt += 1 
        paper_id = subfolder_path.split("/")[-1]
        
        # count the statistics for input text and output text
        ## input
        text_file = os.path.join(subfolder_path, "data_text.json")
        with open(text_file, "r") as f:
            text_data = json.load(f)
        id = text_data["ID"]
        sen_num, w_num = count_input_text(text_data["input"])
        all_sentence_num.append(sen_num)
        all_word_num.append(w_num)
        ## output
        review_num = text_data["review_num"]
        all_review_num.append(review_num)
        item_num_list = text_data["item_num"]
        all_weakness_len.extend(item_num_list)  # avg per review how many items
        out = text_data["output"]
        all_items_this_paper = []
        for r in out:
            all_items_this_paper.extend(r)
        for item in all_items_this_paper:
            this_item_word_num = word_num(item)
            this_item_sentence_num = sentence_num(item)
            all_item_word_num.append(this_item_word_num)
            all_item_sentence_num.append(this_item_sentence_num)
        
        # count the statistics for input figures and tables
        image_dir = os.path.join(subfolder_path, "images")
        # get all the path of *.png files under the image_dir
        # all_images_this_paper = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(".png")]
        all_images_this_paper = [x for x in os.listdir(image_dir) if x.endswith(".png")]
        fg_num, tb_num = 0, 0
        for img in all_images_this_paper:
            if f"{paper_id}-Figure" in img:
                fg_num += 1
            if f"{paper_id}-Table" in img:
                tb_num += 1
        all_image_num.append(fg_num)
        all_table_num.append(tb_num)
        
        try:
            assert id == paper_id
            for img in all_images_this_paper:
                assert paper_id in img
        except:
            import pdb; pdb.set_trace()
            print(f"==> id: {id}, paper_id: {paper_id}, found id mismatch")
    
    
    print(f"==> total instances (papers): {cnt}\n")
    print("="*50)
    print("==> Input statistics")
    print(f"==> AVG length of the input (words num): {sum(all_word_num)/len(all_word_num)}")
    print(f"====> MAX length of the input (words num): {max(all_word_num)}")
    print(f"====> MIN length of the input (words num): {min(all_word_num)}")
    print(f"==> AVG length of the input (sentence num): {sum(all_sentence_num)/len(all_sentence_num)}")
    print(f"====> MAX length of the input (sentence num): {max(all_sentence_num)}")
    print(f"====> MIN length of the input (sentence num): {min(all_sentence_num)}")
    print("="*50)
    print("==> Image statistics")
    print(f"==> AVG number of figures per paper: {sum(all_image_num)/len(all_image_num)}")
    print(f"====> MAX number of figures per paper: {max(all_image_num)}")
    print(f"====> MIN number of figures per paper: {min(all_image_num)}")
    print(f"==> AVG number of tables per paper: {sum(all_table_num)/len(all_table_num)}")
    print(f"====> MAX number of tables per paper: {max(all_table_num)}")
    print(f"====> MIN number of tables per paper: {min(all_table_num)}")
    print("="*50)
    print("==> Output statistics")
    print(f"==> AVG number of reviews per paper: {sum(all_review_num)/len(all_review_num)}")
    print(f"====> MAX number of reviews per paper: {max(all_review_num)}")
    print(f"====> MIN number of reviews per paper: {min(all_review_num)}")
    print(f"==> AVG number of items per review: {sum(all_weakness_len)/len(all_weakness_len)}")
    print(f"====> MAX number of items per review: {max(all_weakness_len)}")
    print(f"====> MIN number of items per review: {min(all_weakness_len)}")
    print(f"==> AVG number of words per item: {sum(all_item_word_num)/len(all_item_word_num)}")
    print(f"====> MAX number of words per item: {max(all_item_word_num)}")
    print(f"====> MIN number of words per item: {min(all_item_word_num)}")
    print(f"==> AVG number of sentences per item: {sum(all_item_sentence_num)/len(all_item_sentence_num)}")
    print(f"====> MAX number of sentences per item: {max(all_item_sentence_num)}")
    print(f"====> MIN number of sentences per item: {min(all_item_sentence_num)}")
    
    with open(os.path.join(args.target_dir, "statistics.json"), "w") as f:
        json.dump({
            "total_instances": cnt,
            "input": {
                "avg_word_num": sum(all_word_num)/len(all_word_num),
                "max_word_num": max(all_word_num),
                "min_word_num": min(all_word_num),
                "avg_sentence_num": sum(all_sentence_num)/len(all_sentence_num),
                "max_sentence_num": max(all_sentence_num),
                "min_sentence_num": min(all_sentence_num),
            },
            "image": {
                "avg_figures_num": sum(all_image_num)/len(all_image_num),
                "max_figures_num": max(all_image_num),
                "min_figures_num": min(all_image_num),
                "avg_tables_num": sum(all_table_num)/len(all_table_num),
                "max_tables_num": max(all_table_num),
                "min_tables_num": min(all_table_num),
            },
            "output": {
                "avg_reviews_num": sum(all_review_num)/len(all_review_num),
                "max_reviews_num": max(all_review_num),
                "min_reviews_num": min(all_review_num),
                "avg_items_num": sum(all_weakness_len)/len(all_weakness_len),
                "max_items_num": max(all_weakness_len),
                "min_items_num": min(all_weakness_len),
                "avg_words_num_per_item": sum(all_item_word_num)/len(all_item_word_num),
                "max_words_num_per_item": max(all_item_word_num),
                "min_words_num_per_item": min(all_item_word_num),
                "avg_sentences_num_per_item": sum(all_item_sentence_num)/len(all_item_sentence_num),
                "max_sentences_num_per_item": max(all_item_sentence_num),
                "min_sentences_num_per_item": min(all_item_sentence_num),
            }
        }, f, indent=4)
    
    
if __name__ == "__main__":
    main()
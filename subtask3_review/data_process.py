'''
use this script to combine all the input and output (review weakness list) together as one file

each directory has a json file with all the input and output
also the extracted images, tables, and pdf files are saved in the same directory

filter out the papers that dont have the source package, or might hard to be used 
'''
import dataclasses
import random
import re
import subprocess
import requests
import os
import pandas as pd
os.environ['TORCH_HOME'] = '/data/rml6079/.torch'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import arxiv
from arxiv import Client
from papermage.recipes import CoreRecipe


def download_file(url, path):
    """
    Helper function to download a file from a given URL
    """
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)


def FindAllSuffix(path: str, suffix: str, verbose: bool = False) -> list:
    ''' find all files have specific suffix under the path

    :param path: target path
    :param suffix: file suffix. e.g. ".json"/"json"
    :param verbose: whether print the found path
    :return: a list contain all corresponding file path (relative path)
    '''
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root, dirs, files)
        for file in files:
            if suffix in file:
                file_path = os.path.join(root, file)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result

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

def extract_all_used_figures(tex_content:list):
    '''
    get all the figure names that used in this tex file
    
    first find the location of "\begin{figure" and "\end{figure}"
    then extract the figure name in between the two lines
    '''
    figure_suffix = [".jpg", ".jpeg", ".png", ".pdf", ".svg", ".eps", ".tif", ".tiff"]
    
    figure_name_set = set()
    for i, line in enumerate(tex_content):
        if "\\begin{figure" in line:
            # go for the end of this figure
            for j in range(i+1, len(tex_content)):
                # if any figure suffix in this line, then extract the figure name
                # if any([suffix in tex_content[j] for suffix in figure_suffix]):
                if "\\end{figure" in tex_content[j]:
                    break
                for suffix in figure_suffix:
                    if suffix in tex_content[j]:
                        # use re to extract the figure name
                        # for example, extract 'comment_barplot.pdf' from '\includegraphics[width=\textwidth]{figs/comment_barplot.pdf}\vspace{-0.5em}'
                        # import pdb; pdb.set_trace()
                        search_res = re.search(r'{(.*?)}', tex_content[j])
                        if search_res is not None:
                            figure_name = search_res.group(1)
                            figure_name = os.path.basename(figure_name)
                            figure_name_set.add(figure_name)
    
    # import pdb; pdb.set_trace()
    return list(figure_name_set)           
                    

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


def extract_all_figure_data(dir_path):
    '''
    given a path, iterate all the files under the path
    return a list of file path, where each file is a figure
    '''
    figure_suffix = ["jpg", "jpeg", "png", "pdf", "svg", "eps", "tif", "tiff"]
    figure_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.split(".")[-1].lower() in figure_suffix:
                figure_list.append(os.path.join(root, file))
    return figure_list


def extract_all_tables(tex_content:list):
    '''
    for loop every lines
    get all the table data, namely:
    lines between "\begin{table" and "\end{table"
    '''
    all_tables = []
    for i, line in enumerate(tex_content):
        if "\\begin{table" in line:
            # go for the end of this table
            table = []
            table.append(line)
            for j in range(i+1, len(tex_content)):
                table.append(tex_content[j])
                if "\\end{table" in tex_content[j]:
                    break
            # table_str = "".join(table)
            # import pdb; pdb.set_trace()
            all_tables.append(table)
    # import pdb; pdb.set_trace()
    return all_tables
            


def process_text_data(input_text_file: str):
    # with open(input_text_file, "r") as f:
    #     data = json.load(f)
    with open(input_text_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["metadata"]

def process_images(title:str, pdf_file: str, temp_save_dir: str):
    '''
    first use papaermage to extract the title of this paper
    
    then search on the arxiv to get the source package
    
    clean the .tex in the source package, to get
        1. all the used figure name
        2. all the table latex source code
    
    then go for all the images under this package, and save only those used figures
    '''
    
    # recipe = CoreRecipe()
    # doc = recipe.run(pdf_file)
    
    # get tables 
    # all_tables = []
    # for page in doc.pages:
    #     # print(f'\n=== PAGE: {page.id} ===\n\n')
    #     for tb in page.tables:
    #         tb = tb.strip()
    #         if tb:
    #             all_tables.append(tb)
    
    # print(f"==> {pdf_file} has {len(all_tables)} tables")
    
    # get titles
    # title = doc.pages[0].titles[0].text
    if title.strip() == "":
        return None, None
    query = f'title:{title}'
    client = Client()  # Initialize the Client
    results = client.results(arxiv.Search(
        query=query,
        max_results=10 # fetch more papers to ensure the paper is found  
    ))
    if not results:
        # no paper searched
        return None, None
    else:
        # has some paper results found, then check if there is an exact matched paper (case-insensitive)
        exact_match = None
        for result in results:
            # Perform case-insensitive comparison, ignore spaces
            if re.sub(r"\s+", "", str(result.title).lower()) == re.sub(r"\s+", "", str(title).lower()):
                exact_match = result
                break
    if not exact_match:
        # no papers are exactly matched the paper title 
        return None, None
    else:
        # found a paper from arxiv that has the same title as the paper we want
        paper_id = exact_match.get_short_id()
        pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
        source_url = f'https://arxiv.org/e-print/{paper_id}'
        
        # paper id may be like "hep-ph/0605047v2.pdf", try to remove all the "/"
        paper_id_name = paper_id.replace("/", "_") 
        temp_paper_dir = f'{temp_save_dir}/{paper_id_name}'
        os.makedirs(temp_paper_dir, exist_ok=True)
        
        pdf_path = os.path.join(temp_paper_dir, f'{paper_id_name}.pdf')
        download_file(pdf_url, pdf_path)
        
        try:
            source_file_path = os.path.join(temp_paper_dir, f'{paper_id_name}_source.tar.gz')
            download_file(source_url, source_file_path)
        except Exception as e:
            # cannot download the source package
            return None, None
        
        assert os.path.exists(source_file_path), f"Source package path does not exist: {source_file_path}"  # shouldn't happen cuz if download failed, we will return None
        # extract the .tar.gz file under the subdir
        source_dir = os.path.join(temp_paper_dir, f"{paper_id_name}_source")
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            # os.system(f"tar -xvf {source_package_path} -C {source_dir}")
            subprocess.run(['tar', '-xvf', source_file_path, '-C', source_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        assert os.path.exists(source_dir), f"running time error, there should be a source directory: {source_dir}"
        
        # find the main.text under the source_dir
        # first iterate all the files (including under subfolders) to see if there are multiple .tex files
        tex_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".tex"):
                    tex_files.append(os.path.join(root, file))
        
        if len(tex_files) > 1:
            # if multiple .tex files, which means a complex tex structure, then we dont wanna use this paper
            return None, None
        elif len(tex_files) == 0:
            return None, None
        else:
            main_tex = tex_files[0]
            # read this tex file
            # with open(main_tex, "r") as f:
            #     tex_content = f.readlines()
            tex_content = red_tex(main_tex)
            if tex_content is None:
                # if the tex file cannot be read, skip this paper
                return None, None
            
            clean_tex = tex_cleaning(tex_content)
            all_used_figures = extract_all_used_figures(clean_tex)
            all_existing_figures_path = extract_all_figure_data(source_dir)
            
            res_figures_path = []
            for figure_path in all_existing_figures_path:
                figure_name = os.path.basename(figure_path)
                if figure_name in all_used_figures:
                    res_figures_path.append(figure_path)
            # import pdb; pdb.set_trace()
            
            # now go for extracting the table data from the cleaned_tex
            all_tables = extract_all_tables(clean_tex)
            return all_tables, res_figures_path

def get_title_by_id(df, input_id):
    result = df[df['ID'] == input_id]['Title']
    if not result.empty:
        # import pdb; pdb.set_trace()
        return result.values[0]
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./subtask3_review")
    parser.add_argument("--target_dir", type=str, default="./subtask3_review_processed")
    parser.add_argument("--num_per_conf", type=int, default=None, help="how many papers to select per conference; if None, then select all")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--temp_source_download_dir", type=str, default="./subtask3_review_temp_source_download", help="temp directory to save the downloaded source package")
    parser.add_argument("--keep_references", action="store_true", help="whether to keep the references in the input text data")
    # add blacklist to skip, such as "ICLR_2022_all_weakness.json,ICLR_2023_all_weakness.json"
    parser.add_argument("--blacklist", type=str, default=None, help="blacklist of the json files to skip")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    random.seed(args.seed)
    
    os.makedirs(args.target_dir, exist_ok=True)
    # if there are already subfolders in the target_dir, then throw an error
    if len(os.listdir(args.target_dir)) > 0:
        raise ValueError(f"target_dir {args.target_dir} is not empty, and you are trying to process new more files into it; please clear / rename the target_dir first")

    black_list = args.blacklist.split(",") if args.blacklist is not None else []
    print(f"==> black_list: {black_list}\n")
    # get all the json files, named with "XXX_all_weakness.json" in the root_dir
    all_conf_json_data_files = FindAllSuffix(args.root_dir, ".json")
    all_conf_json_data_files = [file for file in all_conf_json_data_files if "all_weakness" in file]
    all_conf_json_data_files = [file for file in all_conf_json_data_files if os.path.basename(file) not in black_list]
    for conf_json_file in all_conf_json_data_files:
        print("==> processing conf_json_file: {}".format(conf_json_file))
        with open(conf_json_file, "r") as f:
            conf_json_data = json.load(f)
        conf_name = conf_json_data[0]["Conferece"]
        # ./subtask3_review/our_dataset/ICLR_2022/ICLR_2022_paper
        input_text_dir = os.path.join(args.root_dir, "our_dataset", conf_name, f"{conf_name}_paper")
        # ./subtask3_review/our_dataset/ICLR_2022/ICLR_2022_pdf
        pdf_dir = os.path.join(args.root_dir, "our_dataset", conf_name, f"{conf_name}_pdf")
        meta_csv_dir = os.path.join(args.root_dir, "our_dataset", conf_name, f"{conf_name}_draft_comment.csv")
        df = pd.read_csv(meta_csv_dir)
        # num_per_conf = len(conf_json_data) if args.num_per_conf is None else min(args.num_per_conf, len(conf_json_data))
        # picked_conf_data = random.sample(conf_json_data, num_per_conf)
        
        # shuffle the data
        random.shuffle(conf_json_data)
        conf_paper_num = 0
        os.makedirs(args.temp_source_download_dir, exist_ok=True)
        for paper_data in tqdm(conf_json_data):
            paper_id = paper_data["ID"]
        
            input_text_file = os.path.join(input_text_dir, f"{paper_id}.pdf.json")
            pdf_file = os.path.join(pdf_dir, f"{paper_id}.pdf")
            paper_title = get_title_by_id(df, paper_id)
            
            if not os.path.exists(input_text_file):
                print(f"==> {input_text_file} not exists, skip")
                continue
            if not os.path.exists(pdf_file):
                print(f"==> {pdf_file} not exists, skip")
                continue
            if paper_title is None:
                print(f"==> {paper_id} title not found, skip")
                continue
            
            # 1. process the input text data and combine it with the output wekaness list
            input_text_data = process_text_data(input_text_file)
            if not args.keep_references:
                input_text_data.pop("references") # references of a paper is noisy and not useful
                input_text_data.pop("referenceMentions")
            if input_text_data["title"] is None:
                input_text_data["title"] = paper_title
            paper_data["input"] = input_text_data
            paper_data["output"] = paper_data.pop("weakness_filed")
            paper_data["review_num"] = paper_data.pop("review_num")
            paper_data["item_num"] = paper_data.pop("item_num")
            
            
            # 2. process the pdf, extract the figures and tables
            all_tables, all_figure_path = process_images(paper_title, pdf_file, temp_save_dir=args.temp_source_download_dir)
            if all_tables is None or all_figure_path is None:
                print(f"==> {conf_name}: {paper_id}, source pakcage error, such as downloading fail or multiple-tex, skip")
                continue
                
            save_dir = os.path.join(args.target_dir, conf_name, paper_id)
            save_dir_images = os.path.join(save_dir, "images")
            save_dir_tables = os.path.join(save_dir, "tables")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(save_dir_images, exist_ok=True)
            os.makedirs(save_dir_tables, exist_ok=True)
            # save input and output data
            text_save_file = os.path.join(save_dir, "data_text.json")
            with open(text_save_file, "w") as f:
                json.dump(paper_data, f, indent=2)
            # save the figures
            for figure_path in all_figure_path:
                figure_name = os.path.basename(figure_path)
                save_figure_path = os.path.join(save_dir_images, figure_name)
                subprocess.run(['cp', figure_path, save_figure_path])
            # save the tables
            for i, table in enumerate(all_tables):
                table_save_file = os.path.join(save_dir_tables, f"table_{i}.tex")
                with open(table_save_file, "w") as f:
                    f.writelines(table)
            # also move the pdf file to the save_dir
            pdf_save_file = os.path.join(save_dir, f"{paper_id}.pdf")
            subprocess.run(['cp', pdf_file, pdf_save_file])
            
            conf_paper_num += 1
            if args.num_per_conf is not None and conf_paper_num >= args.num_per_conf:
                break
        
        print(f"==> {conf_name} processed {conf_paper_num} papers out of {len(conf_json_data)}, saved to {args.target_dir}/{conf_name}\n")
        # remove the temp dir
        subprocess.run(['rm', '-rf', args.temp_source_download_dir])

if __name__ == "__main__":
    main()
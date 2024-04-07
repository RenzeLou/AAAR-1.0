import re
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

ACL_DATA = "/data/rml6079/projects/scientific_doc/acl-anthology"

top_tier_confs = ["naacl", "eacl", "tacl", "acl", "emnlp"] #


def download_file(url, path):
    """
    Helper function to download a file from a given URL
    """
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)
        
def extract_id(url):
    '''
    given a arxiv link, such as https://arxiv.org/abs/2106.01754 or https://arxiv.org/pdf/2106.01754.pdf
    get the id of the paper, e.g., 2106.01754
    '''
    id = url.split("/")[-1]
    id = id.replace(".pdf", "")
    return id

def crawl_acl():
    """
    Crawls papers within a given subject and primary category from a specific year
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_urls", type=str, default="https://arxiv.org/abs/2106.01754,https://arxiv.org/abs/2106.01754", help="a string of comma-separated paper urls.")
    parser.add_argument("--annotator", type=str, default="rz", help="the annotator name.")
    parser.add_argument("--save_path", type=str, default="subtask2_experiment_human_anno", help="the directory to save the downloaded papers.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    os.makedirs(args.save_path, exist_ok=True)
    all_paper_ids = os.path.join(args.save_path, "all_paper_ids.txt")
    if not os.path.exists(all_paper_ids):
        # just make an empty file
        with open(all_paper_ids, 'w') as f:
            f.write("")
    
    all_id_list = []
    with open(all_paper_ids, 'r') as f:
        for line in f:
            all_id_list.append(line.strip())
    
    print("==> already have {} papers.".format(len(all_id_list)))
    
    
    target_urls = args.paper_urls.split(",")
    print("==> for the annotator: {}, planning to download {} papers.".format(args.annotator, len(target_urls)))
    
    start_time = time.time()
    
    error_messages = []
    
    for url in target_urls:
        paper_id= extract_id(url)
        if paper_id in all_id_list:
            print(f"==> paper {paper_id} duplicated, skip.")
            error_messages.append((paper_id, "duplicated"))
            continue
        all_id_list.append(paper_id)
        
        pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
        source_url = f'https://arxiv.org/e-print/{paper_id}'
        
        paper_id_name = paper_id.replace("/", "_") 
        paper_dir = f'{args.save_path}/{args.annotator}/{paper_id_name}'
        os.makedirs(paper_dir, exist_ok=True)
        
        # check if there are already files (all the three files) downloaded under this directory
        if os.path.exists(os.path.join(paper_dir, f'{paper_id_name}_metadata.json')) \
        and os.path.exists(os.path.join(paper_dir, f'{paper_id_name}.pdf')) \
        and os.path.exists(os.path.join(paper_dir, f'{paper_id_name}_source.tar.gz')):
            print(f"Paper {paper_id} already downloaded.")
            continue
        
        pdf_path = os.path.join(paper_dir, f'{paper_id_name}.pdf')
        download_file(pdf_url, pdf_path)
                        
        try:
            source_path = os.path.join(paper_dir, f'{paper_id_name}_source.tar.gz')
            download_file(source_url, source_path)
        except Exception as e:
            error_messages.append((paper_id, e))
            print(f"Failed to download source for {paper_id}: {e}")
        
        # use arxiv package to get the title of the paper
        client = Client()  # Initialize the Client
        search = arxiv.Search(
            query=f'id:{paper_id}',
            max_results=1
        )
        try:
            paper = next(client.results(search), None)
        except StopIteration:
            paper = None
        
        if paper is None:
            # this error does not matter, cuz the meta data is not necessary for the task
            # error_messages.append((paper_id, "no arxiv result can be found"))
            print(f"Failed to get metadata for {paper_id} from arxiv. It's werid, becuase if we can successfully download the pdf, the metadata should be there. Double check it!")
        else:
            metadata = {
                    'title': paper.title,
                    'authors': [str(author) for author in paper.authors],  # Converting authors to strings
                    'abstract': paper.summary,
                    'comments': paper.comment,
                    'conference': "arxiv",  ## see our annotation sheet for the conference name
                }
            meta_path = os.path.join(paper_dir, f'{paper_id_name}_metadata.json')
            with open(meta_path, 'w') as meta_file:
                json.dump(metadata, meta_file, indent=4)
    
    # save the all_paper_ids
    with open(all_paper_ids, 'w') as f:
        for paper_id in all_id_list:
            f.write(paper_id + "\n")  
    
    end_time = time.time()
    
    print("=" * 50)
    # print all the error messages
    print("==> for the annotator: {}, found {} errors.".format(args.annotator, len(error_messages)))
    for paper_id, error in error_messages:
        print(f"==> paper {paper_id}: {error}")

# Example usage
if __name__ == '__main__':
    crawl_acl()
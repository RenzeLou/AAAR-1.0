'''
do a further filter on the papers crawled from the ACL anthology to ensure all papers are accepted by top-tier NLP conferences 
'''

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

top_tier_confs = ["acl", "emnlp","naacl", "eacl", "tacl", "iclr"] #  
conf_pattern = r"|".join(top_tier_confs)

root_path = "./acl_papers"
target_path = "./filter_accepted"
# for each subfolder in the root path
source_paper_num = len(os.listdir(root_path))
target_paper_num = 0
for subfolder in tqdm(os.listdir(root_path)):
    # read the *_metadata.json file
    metadata_path = os.path.join(root_path, subfolder, f"{subfolder}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    comments = metadata["comments"] 
    comments = "null" if comments is None else comments
    # use re packege to match: if the comments contain "accepted", "accept", or "appear in", case insensitive
    # and the paper is accepted by top-tier NLP conferences (if any top-tier conference is in the comments, then the paper is accepted by top-tier NLP conferences)
    if re.search(r"accepted|accept|appear in", comments, re.IGNORECASE) and re.search(conf_pattern, comments, re.IGNORECASE):
        # if so, move this subfolder to the target path
        os.makedirs(target_path, exist_ok=True)
        os.rename(os.path.join(root_path, subfolder), os.path.join(target_path, subfolder))
        target_paper_num += 1

print(f"source_paper_num: {source_paper_num}, target_paper_num: {target_paper_num}")
        
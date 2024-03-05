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

def crawl_acl():
    """
    Crawls papers within a given subject and primary category from a specific year
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_year", type=int, default=2024, help="the beginning year of the papers to crawl.")
    parser.add_argument("--end_year", type=int, default=2024, help="the end year of the papers to crawl.")
    parser.add_argument("--save_path", type=str, default="acl_papers", help="the directory to save the downloaded papers.")
    parser.add_argument("--max_per_conf", type=int, default=None, help="the maximum number of papers to download per conference.")

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    begin_year = args.begin_year
    end_year = args.end_year
    
    # load all the data from the ACL anthology
    anthology = Anthology.from_repo(path=ACL_DATA)
    
    year_range = [str(year) for year in range(begin_year, end_year + 1)]
    
    founded = 0
    total_downloaded = 0
    no_source_downloaded = 0
    conf_dist = dict([(conf+"-"+year, 0) for conf in top_tier_confs for year in year_range] + [("Others", 0)])
    conf_dist = dict(sorted(conf_dist.items()))
    
    
    print("="*50)
    print(f"Fetching papers from {begin_year} to {end_year}")
    print(f"Conferences: {top_tier_confs}")
    
    start_time = time.time()
    
    for conf in top_tier_confs:
        conf_num = 0
        for year in year_range:
            # got id, e.g., naacl-2022
            conf_id = f"{conf}-{year}"
            event = anthology.get_event(conf_id)
            # check if the event is NoneType
            if event is None:
                continue
            for volume in event.volumes():
                # skip the workshop
                if "workshop" in str(volume.title).lower():
                    continue
                # for the other venues, including the main conference, findings, etc. get all the papers
                for paper in volume.papers():
                    founded += 1
                    # search this paper on arxiv according to the title
                    try:
                        query = f'title:{paper.title}'
                        client = Client()  # Initialize the Client
                        results = client.results(arxiv.Search(
                            query=query,
                            max_results=10 # fetch more papers to ensure the paper is found  
                        ))
                        
                        if not results:
                            # no paper searched, then continue
                            # print(f"No papers found for the year {year} in {conf_id}.")
                            continue
                        else:
                            # has some paper results found, then check if there is an exact matched paper (case-insensitive)
                            exact_match = None
                            for result in results:
                                # Perform case-insensitive comparison, ignore spaces
                                if re.sub(r"\s+", "", str(result.title).lower()) == re.sub(r"\s+", "", str(paper.title).lower()):
                                    exact_match = result
                                    break
                                
                        if not exact_match:
                            # no papers are exactly matched the paper title from ACL anthology, then continue
                            continue
                        else:
                            # found a paper from arxiv that has the same title as the paper from ACL anthology, which is the paper we want
                            print(f"Downloading: {paper.title} from {conf_id}")
                        
                        
                        paper_id = exact_match.get_short_id()
                        pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
                        source_url = f'https://arxiv.org/e-print/{paper_id}'
                        
                        # paper id may be like "hep-ph/0605047v2.pdf", try to remove all the "/"
                        paper_id_name = paper_id.replace("/", "_") 
                        paper_dir = f'{args.save_path}/{paper_id_name}'
                        os.makedirs(paper_dir, exist_ok=True)
                        
                        # check if there are already files (all the three files) downloaded under this directory
                        if os.path.exists(os.path.join(paper_dir, f'{paper_id_name}_metadata.json')) \
                        and os.path.exists(os.path.join(paper_dir, f'{paper_id_name}.pdf')) \
                        and os.path.exists(os.path.join(paper_dir, f'{paper_id_name}_source.tar.gz')):
                            # print(f"Paper {paper.title} already downloaded.")
                            continue
                        
                        pdf_path = os.path.join(paper_dir, f'{paper_id_name}.pdf')
                        download_file(pdf_url, pdf_path)
                        
                        try:
                            source_path = os.path.join(paper_dir, f'{paper_id_name}_source.tar.gz')
                            download_file(source_url, source_path)
                        except Exception as e:
                            no_source_downloaded += 1
                            print(f"Failed to download source for {paper.title}: {e}")
                        
                        metadata = {
                            'title': exact_match.title,
                            'authors': [str(author) for author in exact_match.authors],
                            'abstract': exact_match.summary,
                            'comments': exact_match.comment,
                            'conference': conf_id
                        }
                        meta_path = os.path.join(paper_dir, f'{paper_id_name}_metadata.json')
                        with open(meta_path, 'w') as meta_file:
                            json.dump(metadata, meta_file, indent=4)
                        
                        total_downloaded += 1
                        if conf_id:
                            conf_dist[conf_id] += 1
                        else:
                            conf_dist["Others"] += 1
                        
                        conf_num += 1
                        if args.max_per_conf and conf_num >= args.max_per_conf:
                            break
                    except arxiv.UnexpectedEmptyPageError as e:    
                        print(f"Error: {str(e)}. Cannot find paper {paper.title} in arxiv.")
    
    end_time = time.time()
    
    print("="*50)
    print(f"Found {founded} papers.")
    print(f"Downloaded {total_downloaded} papers.")
    print(f"Failed to download source for {no_source_downloaded} papers.")
    print("="*50)
    print("Conference distribution:")
    for conf, count in conf_dist.items():
        print(f"{conf}: {count}")
    print("="*50)
    print(f"Time elapsed: {(end_time - start_time) / 3600} hours")
    print("AVG time per paper: ", (end_time - start_time) / founded, "seconds")
    
    
    # write the following summary to a log file
    # name the dir as the time of the crawl (the system time)
    log_dir = "./logs/{time}".format(time=time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"crawl_acl_{begin_year}-{end_year}.log")
    with open(log_file, 'w') as log:
        log.write(f"Found {founded} papers.\n")
        log.write(f"Downloaded {total_downloaded} papers.\n")
        log.write(f"Failed to download source for {no_source_downloaded} papers.\n")
        log.write("Conference distribution:\n")
        for conf, count in conf_dist.items():
            log.write(f"{conf}: {count}\n")
        log.write(f"Time elapsed: {(end_time - start_time) / 3600} hours\n")
        log.write("AVG time per paper: {}\n".format((end_time - start_time) / founded))
        log.write("Log file: {}\n".format(log_file))


# Example usage
if __name__ == '__main__':
    crawl_acl()
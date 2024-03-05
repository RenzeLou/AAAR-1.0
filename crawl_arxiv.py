import time
import arxiv
from arxiv import Client
import requests
import os
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm

top_tier_confs = ["ACL", "EMNLP", "NAACL", "EACL", "TACL", "Findings"]


def download_file(url, path):
    """
    Helper function to download a file from a given URL
    """
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)

# Warning: this function is deprecated, shoudl use acl-anthology package instead
def find_in_acl_anthology(title):
    """
    Checks if a given paper title is part of the top-tier ACL conferences.
    Returns the conference name if found, otherwise None.
    """
    search_url = f"https://aclanthology.org/search/?q={requests.utils.quote(title)}"
    page = requests.get(search_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    for conf in top_tier_confs:
        if soup.find("strong", string=conf):
            return conf
    return None

def crawl_arxiv():
    """
    Crawls arXiv for papers within a given subject and primary category from a specific year
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_category", type=str, default="cs", help="the subject category of the papers to crawl.")
    parser.add_argument("--primary_category", type=str, default="cs.CL", help="the primary category of the papers to crawl.")
    parser.add_argument("--begin_year", type=int, default=2024, help="the beginning year of the papers to crawl.")
    parser.add_argument("--end_year", type=int, default=2024, help="the end year of the papers to crawl.")
    parser.add_argument("--force_acl", action="store_true", help="force the crawler to only download papers from ACL conferences.")
    parser.add_argument("--save_path", type=str, default="arxiv_papers", help="the directory to save the downloaded papers.")
    parser.add_argument("--max_results", type=int, default=500, help="the maximum number of papers to download.")
    # parser.add_argument("--max_chunk_results", type=int, default=100, help="the maximum number of papers to fetch per request.")
    
    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)
    
    subject_cat = args.subject_category
    primary_cat = args.primary_category
    begin_year = args.begin_year
    end_year = args.end_year
    max_results = args.max_results
    # max_chunk_results = args.max_chunk_results
    
    print("="*50)
    print(f"Fetching papers in {primary_cat} from {begin_year} to {end_year}")
    print("Download at most", max_results, "papers")
    print("="*50)
    
    try:
        query = f'cat:{primary_cat} AND submittedDate:[{begin_year}01010000 TO {end_year}12312359]'
        client = Client()  # Initialize the Client
        results = client.results(arxiv.Search(
            query=query,
            #   max_chunk_results=max_chunk_results,  # adjust this to fetch more or fewer papers per request
            max_results=max_results  # adjust this to the maximum number of results you want
        ))
        
        if not results:
            print(f"No papers found for the year range {begin_year} to {end_year} in category {primary_cat}.")
            return
        
        founded = 0
        total_downloaded = 0
        no_source_downloaded = 0
        conf_dist = dict([(conf, 0) for conf in top_tier_confs] + [("Others", 0)])
        
        start_time = time.time()

        for paper in results:
            founded += 1
            conference = find_in_acl_anthology(paper.title)
            # print(f"Paper: {paper.title} from {conference}")
            # exit()
            if conference or not args.force_acl:
                # try:
                    print(f"Downloading: {paper.title} from {conference}")
                    paper_id = paper.get_short_id()
                    pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
                    source_url = f'https://arxiv.org/e-print/{paper_id}'

                    paper_dir = f'{args.save_path}/{paper_id}'
                    os.makedirs(paper_dir, exist_ok=True)

                    pdf_path = os.path.join(paper_dir, f'{paper_id}.pdf')
                    download_file(pdf_url, pdf_path)

                    try:
                        source_path = os.path.join(paper_dir, f'{paper_id}_source.tar.gz')
                        download_file(source_url, source_path)
                    except Exception as e:
                        print(f"Failed to download source for {paper.title}: {e}")
                        no_source_downloaded += 1

                    metadata = {
                        'title': paper.title,
                        'authors': [str(author) for author in paper.authors],
                        'abstract': paper.summary,
                        'comments': paper.comment,
                        'conference': conference
                    }
                    meta_path = os.path.join(paper_dir, f'{paper_id}_metadata.json')
                    with open(meta_path, 'w') as meta_file:
                        json.dump(metadata, meta_file, indent=4)
                    
                    total_downloaded += 1
                    if conference:
                        conf_dist[conference] += 1
                    else:
                        conf_dist["Others"] += 1
                # except arxiv.UnexpectedEmptyPageError as e:
                #     print("Empty page error of paper: {}".format(pdf_url))
            else:
                # print(f"Skipping: {paper.title} as it's not found in top-tier ACL conferences.")
                continue    
    except arxiv.UnexpectedEmptyPageError as e:
        print(f"Error: {str(e)}. No results were found for the query.")
        
        
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
    log_file = os.path.join(log_dir, f"crawl_arxiv_{primary_cat}_{begin_year}-{end_year}_{args.force_acl}.log")
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
    crawl_arxiv()
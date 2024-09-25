import os

from tqdm import tqdm
import random
random.seed(1234)

num = 500
source_dir = "/data/rml6079/projects/scientific_doc/subtask3_review_processed/NeurIPS_2021"
target_dir = "/data/rml6079/projects/scientific_doc/NeurIPS_2021"

os.makedirs(target_dir, exist_ok=True)

# copy all the pdf files under the source_dir to the target_dir
cnt = 0
# only the pdf under the subfolders of the source_dir will be copied, those subsubfolders will be ignored
all_subfolders = os.listdir(source_dir)
# import pdb; pdb.set_trace()
for root in tqdm(all_subfolders):
    for file in os.listdir(os.path.join(source_dir, root)):
        if file == f"{root}.pdf":
            # import pdb; pdb.set_trace()
            source_file = os.path.join(source_dir, root, file)
            target_file = os.path.join(target_dir, file)
            # import pdb; pdb.set_trace()
            os.system(f"cp {source_file} {target_file}")
            cnt += 1
            # print(f"copy {source_file} to {target_file}")
            if num is not None and cnt > num:
                print(f"==> {cnt} pdf files are copied from {source_dir} to {target_dir}")
                exit()

print(f"==> {cnt} pdf files are copied from {source_dir} to {target_dir}")
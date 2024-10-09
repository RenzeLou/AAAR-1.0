## data crwaling
```bash
python crawl_acl.py --begin_year 2021 --end_year 2023
```

Will crwal papers from the top-tier conferences in NLP (2021,2022,2023). Those papers are found in the ACL Anthology. Crawl the source package of each paper from Arxiv (save to `./acl_papers`).

04/02: I have crawled more papers. currently, there are papers from 2019 to 2023 (2019, 2020, 2021, 2022, 2023).


## SubTask 1 --- equation classification

### 1. Data Preparation

```bash
python scripts/subtask1_equation_extraction.py --root_dir './acl_papers' --target_dir './subtask1_equation' --ins_per_paper 10
```

Process the latex source code and extract the equations from the papers. Each paper randomly extract **at most** 10 equations. The extracted equations are saved in `./subtask1_equation`.


- optional
```bash
python scripts/subtask1_data_unified.py --root_dir './subtask1_equation' --save_file 'equation_completion_1_per_paper_gpt4-generate-alot-none.json' --ins_per_paper 1
```

unify the data into one single json file, saved to `./subtask1_equation_unified`. This is optional, cuz if using 
`/subtask1_equation_generation_with_filtering.py`, then the saved data is already unified.

### 2. Wrong equation crafting

- optional
There are three crafting methods in `subtask1_equation_rewriting.py`, here is one example usage:
```bash
python scripts/subtask1_equation_rewriting.py --api_name 'gpt-4-1106-preview' --template 3 --root_dir './subtask1_equation' --ins_per_paper 1 --add_left_equation --overwrite
```

this is optional, if using LLM to generate wrong eq and do the filtering as well. If wanna filtering, do the this step directly:

```bash
python scripts/subtask1_equation_generation_with_filtering.py --api_name gpt-4-1106-preview --template 3 --root_dir './subtask1_equation' --save_dir './subtask1_equation_unified' --save_file 'equation_gpt-gen-wrong-eq.json' --add_left_equation --total_ins_num 1449
```

Will ask GPT-4 to craft the equation based on the contetx before and after the equation. The generated equations are also filtered by the GPT-4 (save a classification instance, only if there is at least one wrong equation is not filtered out). The generated equations are directly saved to `./subtask1_equation_unified` (no data unifying needed).

### 3. Model Evaluation

- close source API LLMs (e.g., Openai GPT)

Use `litellm` to unify the api call of all the close-source LLMs. First, install the `litellm` package:

```bash
conda activate litellm
```

then run the following command to evaluate the model:

```bash
python scripts/subtask1_equation_model_eval.py --api_name 'o1-preview' --root_dir './subtask1_equation_unified' --eval_data_file '1049.human_filter.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 1000

python scripts/subtask1_equation_model_eval.py --api_name 'gpt-4-turbo' --root_dir './subtask1_equation_unified' --eval_data_file '1049.human_filter.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 700
```

For `Gemini`:
```bash
conda activate google_genmini
python scripts/subtask1_equation_model_eval.py --api_name 'gemini/gemini-1.5-pro-latest' --root_dir './subtask1_equation_unified' --eval_data_file '1049.human_filter.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 1000
```

<!-- - classification Accuracy (use --context_max_len to control the context length):
```bash
python scripts/subtask1_equation_model_eval.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 100
python scripts/subtask1_equation_model_eval.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results_no-context' --context_max_len 0
```

- three negative equation "None" classification accuracy:
```bash
python scripts/subtask1_equation_model_eval-none_prediction.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 100
python scripts/subtask1_equation_model_eval-none_prediction.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results_no-context' --context_max_len 0
```
-->

- open source API LLMs (e.g., Llama-3)

```bash
conda activate vllm

sh scripts/run_subtask1.sh 6 mistralai/Mistral-7B-Instruct-v0.3 1000 8192
sh scripts/run_subtask1.sh 6,7 meta-llama/Meta-Llama-3.1-70B-Instruct 1000 10000
sh scripts/run_subtask1.sh 4,5,6,7 mistralai/Mixtral-8x22B-Instruct-v0.1 1000 8192
sh scripts/run_subtask1.sh 4,5,6,7 Qwen/Qwen2.5-72B-Instruct 1000 8192
sh scripts/run_subtask1.sh 4,5,6,7 google/gemma-2-27b 1000 8192
sh scripts/run_subtask1.sh 4,5,6,7 tiiuae/falcon-40b 500 8192
sh scripts/run_subtask1.sh 4,5,6,7 allenai/OLMo-7B-0724-Instruct-hf 500 8192
```

All the evaluation results are saved to `./subtask1_equation_unified/eval_results` 


### 4. final data:

- `equation_gpt-gen-wrong-eq_gpt-filtered_1449.json`


### 5. human filtering

I used `python scripts/subtask1_equation_data_seperate_combine.py` seperated the data into 3 parts, and then ask the annotator to filter the data.

Then, each annotator run `python scripts/subtask1_equation_human_filter.py`, to do the filtering in the terminal.

### 6. final data (after human filtering):

- `1049.human_filter.json`

```
==> totoal data length:  1449
==> filtered data length:  1049
==> percentage:  0.7239475500345065
```

### 7. data statistics

```bash
python scripts/subtask1_data_statistics.py

=== statistics of equation ===
==> <context before> word length: min=711, max=24849, mean=4376.7550047664445
==> <context before> sentence length: min=12, max=540, mean=112.40610104861773
==> <context after> word length: min=8, max=32948, mean=6362.4795042897995
==> <context after> sentence length: min=1, max=608, mean=154.02573879885605
==> <ground truth answer> equation length (in non-white character): min=1, max=1039, mean=55.2745471877979
==> <GPT-craft wrong> equation length (in non-white character): min=1, max=306, mean=48.25293930727677
==> <answer> distribution: {'A': 261, 'B': 266, 'C': 261, 'D': 261}
```


## SubTask 2 --- experiments design

<!-- ### 1. Data seperation

use the following command to seperate the context before and after "Experiment" section in the tex.

```bash
python scripts/subtask2_context_seperate.py --root_dir './acl_papers' --target_dir './subtask2_expriment'
``` -->

### 1. Check paper validity

If we are using human to annotaote (100 papers), then first is to check if the annotator's paper choice is valid, by running `scripts/subtask2_download_process_verify.py`.

For example:

```bash
python scripts/subtask2_download_process_verify.py --paper_urls https://arxiv.org/abs/2109.01247,https://arxiv.org/abs/2104.08773,https://arxiv.org/abs/2204.07705,https://arxiv.org/abs/2202.12837,https://arxiv.org/abs/2109.07830,https://arxiv.org/abs/2212.10560,https://arxiv.org/abs/2212.09689,https://arxiv.org/abs/2306.04751,https://arxiv.org/abs/2305.14327,https://arxiv.org/abs/2312.02436 --annotator "rz"
```

It will download all the papers and print any errors on the screen, if there is any errors in the paper choices (e.g., duplicated paper, paper not found, etc.), let this annotator to choose another paper.



Then try to process those downloaded papers, and seperate the context before and after the "Experiment" section. For example:

```bash
python scripts/subtask2_context_seperate.py --root_dir './subtask2_experiment_human_anno/rz' --target_dir './subtask2_experiment_human_anno/rz_seperate'
```

If there are some `multiple main.tex files (have \documentclass) founded in one project, skip.` warnings, you should mannually delete those older versions of the main.tex files. (or simply ask the annotator to choose another paper)

For all the other warnings, ask the annotator to change the paper choice.

The final data is saved in `./subtask2_experiment_human_anno`, each subfolder is named by the annotator's name. and the `XXX_seperate` contains the final processed latex source code data (context before and after the "Experiment" section), that should have at least 10 papers under each annotator's folder. Those paper dir named with "_dep" or "_deperacated" are omitted.

### 2. Manual annotation

Have asked 10 expertise to annotate totaly 100 papers. The raw human annotation list can be found in this [google sheet](https://docs.google.com/spreadsheets/d/1GDNJyXWMrnYzQKQVMe8ITShbpF-jsSxhMrsuYIWJFDI/edit?gid=0#gid=0).

I have manually copied all the annotation exp list in to each paper subfolder. For example:
```
/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/renze_seperate/2104.08773/what.txt
/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/renze_seperate/2104.08773/why.txt
```

and ran `python process_and_combine_annotation.py` can combine both `what` and `why` into one single json file, namely the `annotation.json` under each papaer folder.

### 3. del leaking sentence

since we use the `context_beforw_exp` as the model's input, we have to delete those sentences in `context_beforw_exp` that might leak the output (i.e., experiment idea list).

here we provide gpt with the ground-truth exp list (from `annotation.json`), along with the `context_beforw_exp` to let gpt to do a binary classification. If the sentence is classified as leaking the experiment ideas, then we delete it.

```bash
python ./scripts/subtask2_del_sentence_leak.py
```
All the delete version of `context_beforw_exp` will be saved as a new filed named `context_before_exp_cleaned`, in the original paper folder's context file.

For example:
```
./subtask2_experiment_human_anno/jianxie_seperate/2303.11366/context_before_after_exp.json
```

within this file, there should be four keys: `context_before_exp`, `context_after_exp`, `context_before_exp_cleaned`, `del_percentage`.

### 4. final data

then run `python combine_single_data_file.py` to get the final data, under `./subtask2_experiment_human_anno/final_data`.

Where each paper folder contains the following files:
```
data_text.json: the pure text data (input context and output experiment idea list)
images: all the images in this paper (suplementary data)
XX.source.tar.gz: the source code of the paper (suplementary data; raw data)
```

The final statistics (run `python calculate_statistics.py`):

```
==================================================
==> Input statistics
Number of sentences: AVG: 345.15, MAX: 952, MIN: 53
Number of words: AVG: 4288.3, MAX: 9799, MIN: 698
==================================================
==> Output statistics
Number of items in output: AVG: 5.7, MAX: 13, MIN: 2
Number of words per item (what): AVG: 34.333333333333336, MAX: 135, MIN: 9
Number of words per item (why): AVG: 27.145614035087718, MAX: 89, MIN: 9
==================================================
==> Image statistics
Number of images per paper: AVG: 16.01, MAX: 91, MIN: 0
```

while for the images, since all the image files under the `images` are from the arxiv source pakcage, not all of them are used in the paper. I used the following cmd to i. keep only those images used in the input context; ii. unified all the images format into `png`:

```bash
conda activate crawl_arxiv
cd subtask2_experiment_human_anno
python process_all_images.py
```

After processing, there is a new subfolder `images/used`, contains all the used figures in this paper (used in the input context, remember that we only provide model with the context before the experiment section as the input). That's why, the final used images number is much less than the total image number in the source package.

```
==> total instances: 100
==> AVG images (in latex source):  16.01
==> MAX images (in latex source):  91
==> MIN images (in latex source):  0
==> AVG used images (used in the input context):  2.57  # much lower than the # of images in the source package
==> MAX used images (used in the input context):  16
==> MIN used images (used in the input context):  0
```


### 5. run experiment

#### 5.1 close source model (e.g., Openai GPT)

- pure text

use following command to get the eval prediction of close source model:

```bash
conda activate litellm

sh scripts/run_subtask2.sh [GPUs] [model_name] [max_word_len for input] [max_model_len]

# v1 is deprecated, use v2 instead
# python scripts/subtask2_experiment_model_prediction.close_source.py --api_name "gpt-4" --root_dir "./subtask2_experiment_human_anno/final_data" --save_dir "./subtask2_experiment_human_anno/eval_results" --max_word_len 5000 --oracle
# 2024/09/29: but found the v1 for openai gpt performance is better, v2 is better for open source models

python scripts/subtask2_experiment_model_prediction.close_source.v2.py --api_name "gpt-4o" --root_dir "./subtask2_experiment_human_anno/final_data" --save_dir "./subtask2_experiment_human_anno/eval_results" --max_word_len 3000 --oracle

python scripts/subtask2_experiment_model_prediction.close_source.v2.py --api_name "gpt-4" --root_dir "./subtask2_experiment_human_anno/final_data" --save_dir "./subtask2_experiment_human_anno/eval_results" --max_word_len 3000 --oracle

python scripts/subtask2_experiment_model_prediction.close_source.v2.py --api_name "o1-preview" --root_dir "./subtask2_experiment_human_anno/final_data" --save_dir "./subtask2_experiment_human_anno/eval_results" --max_word_len 3000 --oracle
```

- multi-modal (using the images in the input context)

```bash
python scripts/subtask2_experiment_model_prediction.close_source.py --api_name "gpt-4o" --root_dir "./subtask2_experiment_human_anno/final_data" --save_dir "./subtask2_experiment_human_anno/eval_results" --max_word_len 3000 --oracle --images
```

#### 5.2 open source model (e.g., Llama-3)

use following command to get the eval prediction of open source model:

```bash
conda activate vllm

sh scripts/run_subtask2.v2.sh 2,3,4,5 Qwen/Qwen2.5-72B-Instruct 3000 8192
sh scripts/run_subtask2.v2.sh 2,3,4,5 meta-llama/Meta-Llama-3.1-70B-Instruct 3000 10000
sh scripts/run_subtask2.v2.sh 2,3,4,5 mistralai/Mixtral-8x22B-Instruct-v0.1 3000 8192
sh scripts/run_subtask2.v2.sh 2,3,4,5 mistralai/Mistral-7B-Instruct-v0.3 3000 8192
sh scripts/run_subtask2.v2.sh 2,3,4,5 google/gemma-2-27b 3000 8192 
sh scripts/run_subtask2.v2.sh 2,3,4,5 tiiuae/falcon-40b 2000 8192
sh scripts/run_subtask2.v2.sh 2,3,4,5 allenai/OLMo-7B-0724-Instruct-hf 2000 4096
```

- multi-modal (using the images in the input context)

```bash
conda activate vllm_mm
sh scripts/run_subtask2.multi_modal.v2.sh 4,5,6,7 OpenGVLab/InternVL2-26B 2000 12000 1  # 1 images
sh scripts/run_subtask2.multi_modal.v2.sh 4,5,6,7 OpenGVLab/InternVL2-26B 2000 12000 0  # 0 images for comparison
```

### 6. metrics calculation

then, use the following command to get the final evaluation metrics, for both close source and open source models:
```bash
conda activate crawl_arxiv

python scripts/subtask2_metric.py --root_dir './subtask2_experiment_human_anno/eval_results/xxx'
```

for example:
```bash
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/tiiuae_falcon-40b-2000-oracle---202409211523'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/mistralai_Mistral-7B-Instruct-v0.3-2000-oracle---202409210421'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/meta-llama_Meta-Llama-3.1-70B-Instruct-2000-oracle---202409210239'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/allenai_OLMo-7B-0724-Instruct-hf-2000-oracle---202409211510'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/google_gemma-2-27b-2000-oracle---202409210156'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/gemini-1.5-pro-3000-oracle---202409211345/gemini-1.5-pro-3000-oracle---202409211345'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/claude-3-5-sonnet-20240620-3000-oracle---202409211939'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/gpt-4o-3200-oracle---202409152244'
python scripts/subtask2_metric.py --root_dir '/data/rml6079/projects/scientific_doc/subtask2_experiment_human_anno/eval_results/gpt-4-3000-oracle---202409162134'
```

the eval performance will be saved in `--root_dir` folder.

## SubTask 3 --- review weakness generation

### 1. preprocess the data

Download the raw data from [google drive](https://drive.google.com/drive/folders/1KKpMFj3S5CLUWF_vIo7vaBpXnAYrIxtv)

```bash
# for v1
cd subtask3_review

python step3.5_weakness_extraction.py --csv '/data/rml6079/projects/scientific_doc/subtask3_review/our_dataset/ICLR_2022/ICLR_2022_draft_comment.csv'
python step3.5_weakness_extraction.py --csv '/data/rml6079/projects/scientific_doc/subtask3_review/our_dataset/ICLR_2023/ICLR_2023_draft_comment.csv'
python step3.5_weakness_extraction.py --csv '/data/rml6079/projects/scientific_doc/subtask3_review/our_dataset/NeurIPS_2021/NeurIPS_2021_draft_comment.csv'
python step3.5_weakness_extraction.py --csv '/data/rml6079/projects/scientific_doc/subtask3_review/our_dataset/NeurIPS_2022/NeurIPS_2022_draft_comment.csv'

# for v2, only use ICLR 2023
cd subtask3_review_v2 
python extract_weakness_list.py --csv './ICLR_2023_draft_comment_meta.csv'
# use the following script to select 1000 paper, and plot the track and score distribution
python score_distribution.py
```

The above commands will process all the csv files to json files, by **using GPT4 to extract the weakness list** from the raw review comments. The extracted weakess list for each paper are saved at `./subtask3_review/all_weakness.json`, such as `./subtask3_review/ICLR_2023_all_weakness.json`

Then use the following command to combine all the output (weakness list) with the input, and extract download the figure and table in the paper.

```bash
conda activate papermage

# v1.1 that download the source package and extract all the source pic, tables (really time-consuming)
python ./subtask3_review/data_process.py --root_dir "./subtask3_review" --target_dir "./subtask3_review_processed"
# v1.2 that don't download the source package, will use pdffigures to extract the figures and tables from the draft later (really fast)
python ./subtask3_review/data_process.v2.py --root_dir "./subtask3_review" --target_dir "./subtask3_review_processed_v2" --blacklist 'NeurIPS_2021_all_weakness.json' --num_per_conf 500

# v2
python subtask3_review_v2/data_process.py
```

the processed data will be saved in `./subtask3_review_processed` folder, where each paper is the subfolder, and there are three type of files under each paper folder: 1. `data_text.json`, 2. `images`, 3. `tables`

while `./subtask3_review_processed_v2` folder contains only the `data_text.json` file.

### 2. extract the image and table

since the arxiv table and images might not be under-review version, we use the `pdffigures` to extract the figures and tables from the draft version of the paper.

I run the pdffigures on my local machine.

for v1, after run `python subtask3_review/process_final_data.py`, get the finald data under `subtask3_review_final_light` (simply combine the `data_text.json` and `images`, del useless files such as arxiv source)

for v2, after run `python subtask3_review_v2/process_final_data.py`. get final data at `./subtask3_review_final_v2/ICLR_2023`, totally 995 subfolders (papers).

### 3. statistics

run `python subtask3_review_final_light/calculate_statistics.py`


```
==> total instances (papers): 1925

==================================================
==> Input statistics
==> AVG length of the input (words num): 9152.732467532467
====> MAX length of the input (words num): 36776
====> MIN length of the input (words num): 23
==> AVG length of the input (sentence num): 339.24363636363637
====> MAX length of the input (sentence num): 1635
====> MIN length of the input (sentence num): 3
==================================================
==> Image statistics
==> AVG number of figures per paper: 5.484675324675325
====> MAX number of figures per paper: 45
====> MIN number of figures per paper: 0
==> AVG number of tables per paper: 3.2306493506493505
====> MAX number of tables per paper: 53
====> MIN number of tables per paper: 0
==================================================
==> Output statistics
==> AVG number of reviews per paper: 3.802077922077922
====> MAX number of reviews per paper: 7
====> MIN number of reviews per paper: 2
==> AVG number of items per review: 5.313567427244159
====> MAX number of items per review: 54
====> MIN number of items per review: 0
==> AVG number of words per item: 40.06847518642324
====> MAX number of words per item: 372
====> MIN number of words per item: 1
==> AVG number of sentences per item: 2.89154024170738
====> MAX number of sentences per item: 18
====> MIN number of sentences per item: 1
```

`python subtask3_review_final_v2/calculate_statistics.py`

I mannualy deleted two papaers without any input text extracted by hanzi. Finally 993 papers are left.

```
==> total instances (papers): 993

==================================================
==> Input statistics
==> AVG length of the input (words num): 9811.710976837865
====> MAX length of the input (words num): 49195
====> MIN length of the input (words num): 24
==> AVG length of the input (sentence num): 368.58710976837864
====> MAX length of the input (sentence num): 2635
====> MIN length of the input (sentence num): 3
==================================================
==> Image statistics
==> AVG number of figures per paper: 6.997985901309164
====> MAX number of figures per paper: 37
====> MIN number of figures per paper: 0
==> AVG number of tables per paper: 4.295065458207453
====> MAX number of tables per paper: 53
====> MIN number of tables per paper: 0
==================================================
==> Output statistics
==> AVG number of reviews per paper: 3.785498489425982
====> MAX number of reviews per paper: 9
====> MIN number of reviews per paper: 3
==> AVG number of items per review: 4.84463953179037
====> MAX number of items per review: 39
====> MIN number of items per review: 0
==> AVG number of words per item: 39.08956125418703
====> MAX number of words per item: 371
====> MIN number of words per item: 1
==> AVG number of sentences per item: 2.8793586293998135
====> MAX number of sentences per item: 16
====> MIN number of sentences per item: 1
==================================================
==> 95% of the papers have word length below: 20943
==> 90% of the papers have word length below: 16532
==> 80% of the papers have word length below: 12367
```

### 4. run experiment

- close source model (e.g., Openai GPT)

```bash
# v1
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_light' --save_dir './subtask3_review_final_light/eval_results' --split --max_word_len 3500 --pick_num 1000

# v2
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --split --max_word_len 3000 
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'o1-preview' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --split --max_word_len 3000
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4-turbo' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --split --max_word_len 3000
```

run multi-modal (using the images):
```bash
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --max_word_len 3000 --split --tables --figures

python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --max_word_len 3000 --split --figures

python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --max_word_len 3000 --split --tables
```

- open source model (e.g., Llama-3)

```bash
# v1
sh scripts/run_subtask3.sh 4,5,6,7 Qwen/Qwen2.5-72B-Instruct 2500 8192 1000 1
sh scripts/run_subtask3.sh 4,5,6,7 meta-llama/Meta-Llama-3.1-70B-Instruct 2500 10000 1000 1
sh scripts/run_subtask3.sh 4,5,6,7 mistralai/Mixtral-8x22B-Instruct-v0.1 2500 8192 1000 1
sh scripts/run_subtask3.sh 4,5,6,7 mistralai/Mistral-7B-Instruct-v0.3 2500 8192 1000 1
sh scripts/run_subtask3.sh 4,5,6,7 google/gemma-2-27b 2500 8192 1000 1
sh scripts/run_subtask3.sh 4,5,6,7 allenai/OLMo-7B-0724-Instruct-hf 1000 8192 1000 1
sh scripts/run_subtask3.sh 4,5,6,7 tiiuae/falcon-40b 1000 8192 1000 1

# v2, no pick_num args
sh scripts/run_subtask3.sh 4,5,6,7 Qwen/Qwen2.5-72B-Instruct 3000 8192 1
sh scripts/run_subtask3.sh 4,5,6,7 meta-llama/Meta-Llama-3.1-70B-Instruct 3000 10000 1
sh scripts/run_subtask3.sh 4,5,6,7 mistralai/Mixtral-8x22B-Instruct-v0.1 3000 8192 1
sh scripts/run_subtask3.sh 4,5,6,7 mistralai/Mistral-7B-Instruct-v0.3 3000 8192 1
sh scripts/run_subtask3.sh 4,5,6,7 google/gemma-2-27b 2500 8192 1
sh scripts/run_subtask3.sh 4,5,6,7 allenai/OLMo-7B-0724-Instruct-hf 1000 8192 1
sh scripts/run_subtask3.sh 4,5,6,7 tiiuae/falcon-40b 1000 8192 1
```

run multi-modal (using the images)

```bash
sh scripts/run_subtask3.multi_modal.sh 4,5,6,7 OpenGVLab/InternVL2-26B 2000 12000 2
sh scripts/run_subtask3.multi_modal.sh 4,5,6,7 OpenGVLab/InternVL2-26B 2000 12000 1
sh scripts/run_subtask3.multi_modal.sh 4,5,6,7 OpenGVLab/InternVL2-26B 2000 12000 0
```

- Agent Framework (AI-Scientist)

```bash
conda activate ai_scientist

python scripts/subtask3_review_model_prediction.agent.py --api_name 'gpt-4o-2024-05-13' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --max_word_len 3000 --split
```

### 5. metrics calculation

```bash
conda activate crawl_arxiv

# soft score
python scripts/subtask3_metric.py
# cross diversity
python scripts/subtask3_metric_cross_diversity.py --batch_size 512 --papaer_top_k 2 --track_top_k 20 --threshold 0.5
python scripts/subtask3_metric_cross_diversity.py --batch_size 512 --papaer_top_k 2 --track_top_k 20 --only_human_score --threshold 0.5 --pick_choice 3 # human's weakness list score
```
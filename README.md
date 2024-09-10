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

TODO

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
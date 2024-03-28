## data crwaling
```bash
python crawl_acl.py --begin_year 2021 --end_year 2023
```

Will crwal papers from the top-tier conferences in NLP (2021,2022,2023). Those papers are found in the ACL Anthology. Crawl the source package of each paper from Arxiv (save to `./acl_papers`).




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

- classification Accuracy (use --context_max_len to control the context length):
```bash
python scripts/subtask1_equation_model_eval.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 100
python scripts/subtask1_equation_model_eval.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results_no-context' --context_max_len 0
```

- three negative equation "None" classification accuracy:
```bash
python scripts/subtask1_equation_model_eval-none_prediction.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 100
python scripts/subtask1_equation_model_eval-none_prediction.py --api_name 'gpt-4-1106-preview' --root_dir './subtask1_equation_unified' --eval_data_file 'equation_gpt-gen-wrong-eq_gpt-filtered_1449.json' --save_dir './subtask1_equation_unified/eval_results_no-context' --context_max_len 0
```

All the evaluation results are saved to `./subtask1_equation_unified/eval_results`
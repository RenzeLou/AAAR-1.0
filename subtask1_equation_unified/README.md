## Gemini Experiments

#### 1. Environment Setup

```bash
conda create -n google_genmini python==3.10
conda activate google_genmini
```

Then install all the required packages:

```bash
pip install -q google-generativeai
pip install vllm
pip install tenacity
pip install bs4 requests
pip install litellm
pip install absl-py
pip install nltk
pip install six
```

#### 2. Model Evaluation (put in the main table)

```bash
python scripts/subtask1_equation_model_eval.py --api_name 'gemini/gemini-1.5-pro-latest' --root_dir './subtask1_equation_unified' --eval_data_file '1049.human_filter.json' --save_dir './subtask1_equation_unified/eval_results' --context_max_len 1000
```
After finishing running the above command, the evaluation results will be saved to `./subtask1_equation_unified/eval_results/1049.human_filter.json/gemini_gemini-1.5-pro-latest`. 

Please sent me the json files under the above directory.
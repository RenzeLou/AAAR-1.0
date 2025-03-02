
<h1 align="center"> <img src="./figures/robot_expert_2.png" width="45" height="45">AAAR-1.0: Assessing AI's Potential to Assist Research </h1>

<p align="center">
<a href="https://renzelou.github.io/AAAR-1.0/"><img src="https://img.shields.io/badge/Website-red" alt="website" /></a>
<a href="https://arxiv.org/abs/2410.22394"><img src="https://img.shields.io/badge/Paper-orange" alt="paper" /></a>
  <a href="https://github.com/RenzeLou/AAAR-1.0/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-blue" alt="license" /></a>
  <!-- <a href="https://github.com/RenzeLou/Muffin"><img src="https://img.shields.io/badge/Python-3.8-green" alt="python" /></a> -->
</p>

This repository contains the source code for running the LLMs' performance on the [AAAR-1.0 benchmark](https://renzelou.github.io/AAAR-1.0/).

<p align="center" width="100%">
<a ><img src="./figures/2.png" alt="paradigm" style="width: 90%; min-width: 300px; display: block; margin: auto;"></a>
</p>

We dfined four tasks in the AAAR-1.0 benchmark:
- (i) 𝙀𝙦𝙪𝙖𝙩𝙞𝙤𝙣 𝙄𝙣𝙛𝙚𝙧𝙚𝙣𝙘𝙚 🌟: Based on the context of the related paper, such as the description and necessary symbols of an AI/ML algorithm, infer the correct mathematical equation for the algorithm.

- (ii) 𝙀𝙭𝙥𝙚𝙧𝙞𝙢𝙚𝙣𝙩 𝘿𝙚𝙨𝙞𝙜𝙣 🧪: Given a partial research paper containing the research idea or proposal (primarily the "Abstract" or "Introduction" sections), design appropriate experiments and explain their necessity.

- (iii) 𝙋𝙖𝙥𝙚𝙧 𝙒𝙚𝙖𝙠𝙣𝙚𝙨𝙨 🔍: Given a paper draft, write the review (weaknesses) of this work, i.e., LLMs act as reviewers.

- (iv) 𝙍𝙚𝙫𝙞𝙚𝙬 𝘾𝙧𝙞𝙩𝙞𝙦𝙪𝙚 ✍️: Given a paper draft along with its peer review, identify any unreliable or deficient viewpoints, i.e., LLMs act as meta reviewers.


---


## Benchmark Download

Please download AAAR-1.0 from 🤗 HuggingFace: [https://huggingface.co/datasets/Reza8848/AAAR-1.0](https://huggingface.co/datasets/Reza8848/AAAR-1.0)

You can use the following command:
```bash
git lfs install  # make sure you have git-lfs installed (https://git-lfs.com)
git clone git@hf.co:datasets/Reza8848/AAAR-1.0  # clone all the large data files

mv AAAR-1.0/Equation_Inference ./ 
mv AAAR-1.0/Experiment_Design ./
mv AAAR-1.0/Paper_Weakness ./
```


## Environment Setup

For running closed-source LLMs (e.g., OpenAI GPT), we use [litellm](https://github.com/BerriAI/litellm) to unify various model calling APIs, please setup the following environment:

<!-- ```bash
conda env create -f environment.litellm.yml
conda activate litellm
``` -->
```bash
conda create -n litellm python=3.9
conda activate litellm
pip install -r env_source/requirements.litellm.txt
```

while for running open-source LLMs (e.g., Llama), we mainly use [vllm](https://github.com/vllm-project/vllm), please setup the following environment:

<!-- ```bash
conda env create -f environment.vllm.yml
conda activate vllm
``` -->
```bash
conda create -n vllm python=3.10
conda activate vllm
pip install -r env_source/requirements.vllm.txt
```

** If you wanna run open-source LLMs with multi-modal inputs, please use `requirements.vllm_mm.txt`


## API Tokens

When running closed-source commercial LLMs, you can set the API tokens in the environment variables, for example: 
```bash
export OPENAI_API_KEY='your-api-key-here'
export ANTHROPIC_API_KEY='your-api-key-here'
```

or write them in the `~/.bashrc` or `~/.zshrc` file.

While for running open-source LLMs from HuggingFace, you have to write a `huggingface_key.txt` file in this project root directory, and put your Huggingface Access Token in it.

## Running the Benchmark

### 1. Equation Inference 🌟:

- For **closed-source** LLMs, please using the following command:

```bash
conda activate litellm
python scripts/subtask1_equation_model_eval_binary.py --root_dir './Equation_Inference' --eval_data_file 'equation.1049.json' --save_dir './Equation_Inference/eval_results_binary' --context_max_len [max_context_len] --api_name [model_name]

# for example
python scripts/subtask1_equation_model_eval_binary.py --root_dir './Equation_Inference' --eval_data_file 'equation.1049.json' --save_dir './Equation_Inference/eval_results_binary' --context_max_len 1000 --api_name 'o1-preview'
```

- For **open-source** LLMs (such as Llama), please using the following command:

```bash
conda activate vllm
sh scripts/run_subtask1_binary.sh [GPU_IDs] [model_name] [max_context_len] [max_model_len]

# for example
sh scripts/run_subtask1_binary.sh 6,7 meta-llama/Meta-Llama-3.1-70B-Instruct 1000 10000
```

All the evaluation results are saved to `./Equation_Inference/eval_results_binary` directory.


> Note that, in our paper, we treat the equation inference task as a binary classification task, i.e., do binary decision on each equation. However, since there are 4 candidate equations (1 correct and 3 incorrect), we can also regard it as a **multi-class classification task (QA)**:

```bash
# for closed-source LLMs
python scripts/subtask1_equation_model_eval.py --root_dir './Equation_Inference' --eval_data_file 'equation.1049.json' --save_dir './Equation_Inference/eval_results' --context_max_len [max_context_len] --api_name [model_name]

# for open-source LLMs
sh scripts/run_subtask1.sh [GPU_IDs] [model_name] [max_context_len] [max_model_len]
```

### 2. Experiment Design 🧪:

- For **closed-source** LLMs, please using the following command:

```bash
conda activate litellm
python scripts/subtask2_experiment_model_prediction.close_source.v2.py --root_dir "./Experiment_Design" --save_dir "./Experiment_Design/eval_results" --oracle --max_word_len [max_context_len] --api_name [model_name]

# for example
python scripts/subtask2_experiment_model_prediction.close_source.v2.py --root_dir "./Experiment_Design" --save_dir "./Experiment_Design/eval_results" --max_word_len 3000 --api_name "gpt-4o" --oracle
```

- For **open-source** LLMs, please using the following command:

```bash
conda activate vllm
sh scripts/run_subtask2.v2.sh [GPU_IDs] [model_name] [max_context_len] [max_model_len]

# for example
sh scripts/run_subtask2.v2.sh 2,3,4,5 Qwen/Qwen2.5-72B-Instruct 3000 8192
```

All the evaluation results are saved to `./Experiment_Design/eval_results` directory.


- **Evaluation Metrics**:

For experiment design task, we use LLM-as-judge to measure the matching degree between the generated experiments and the ground-truth experiments. Use the following scripts:
```bash
python scripts/calculate_metrics_subtask2_exp_entailment.v2.py --api_name 'gpt-4o' --paper_selection_path none --model_prediction_path 'xxx' ## use the specific model prediction output directory
python scripts/subtask2_metric_llm_as_judge.py --root_dir './Experiment_Design/eval_results/'
```

For the generated explanations, we use SentenceBERT to evaluate the semantic similarity between the generated explanations and the ground-truth explanations. Use the following command to run SentenceBERT on your local machine:

```bash
python scripts/subtask2_metric.py --root_dir './Experiment_Design/eval_results/xxx'  ## use the specific model results directory
```


### 3. Paper Weakness 🔍:

- For **closed-source** LLMs, please using the following command:

```bash
conda activate litellm
python scripts/subtask3_review_model_prediction.close_source.py --root_dir './Paper_Weakness' --save_dir './Paper_Weakness/eval_results' --split --max_word_len [max_context_len] --api_name [model_name]

# for example
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './Paper_Weakness' --save_dir './Paper_Weakness/eval_results' --split --max_word_len 3000
```

- For **open-source** LLMs, please using the following command:

```bash
conda activate vllm
sh scripts/run_subtask3.sh [GPU_IDs] [model_name] [max_context_len] [max_model_len] [split_context]

# for example
sh scripts/run_subtask3.sh 4,5,6,7 Qwen/Qwen2.5-72B-Instruct 3000 8192 1  # "1" means split context into multiple parts, and combine the results afterwards
```

- **Evaluation Metrics**:

```bash
python scripts/subtask3_metric.py  # soft score
python scripts/subtask3_metric_cross_diversity.py --batch_size 512 --papaer_top_k 2 --track_top_k 20 --threshold 0.5 # weakness diversity
```

It will calculate the metrics for all the model's results in the `./Paper_Weakness/eval_results` directory.

### 4. Review Critique ✍️:

Please refer to [this repository](https://github.com/jiangshdd/ReviewCritique) for more details on running the review critique task.

---

## 🥳 Citation

Please kindly cite our paper if you use any resources of AAAR-1.0:


```bibtex
@article{Lou2024AAAR,
  title={{AAAR-1.0}: Assessing AI's Potential to Assist Research},
  author={Renze Lou and Hanzi Xu and Sijia Wang and Jiangshu Du and Ryo Kamoi and Xiaoxin Lu and Jian Xie and Yuxuan Sun and Yusen Zhang and Jihyun Janice Ahn and Hongchao Fang and Zhuoyang Zou and Wenchao Ma and Xi Li and Kai Zhang and Congying Xia and Lifu Huang and Wenpeng Yin},
  journal={arXiv preprint arXiv:2410.22394},
  year={2024}
}
```

<!-- As well as the following paper:

```bibtex
@inproceedings{du2024llms,
  title={Llms assist nlp researchers: Critique paper (meta-) reviewing},
  author={Du, Jiangshu and Wang, Yibo and Zhao, Wenting and Deng, Zhongfen and Liu, Shuaiqi and Lou, Renze and Zou, Henry Peng and Venkit, Pranav Narayanan and Zhang, Nan and Srinath, Mukund and others},
  journal={Proceedings of {EMNLP} },
  year={2024}
}
``` -->

---

<!-- omit in toc -->
## ⭐ Star History


[![Star History Chart](https://api.star-history.com/svg?repos=RenzeLou/AAAR-1.0&type=Date)](https://star-history.com/#RenzeLou/AAAR-1.0&Date)

## data

download data from [here](https://drive.google.com/file/d/1917SLXerD3qlrQHNkRac_71Tl4WYiV16/view?usp=sharing)

unzip and put the data under this folder, finally the folder structure should be like this:

```
subtask3_review_final_light
    - ICLR_2022
    - ICLR_2023
    - NeurIPS_2021
    - NeurIPS_2022
```

## run experiment

I used the following cmd to run the OpenAI GPT model:
```bash
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_light' --save_dir './subtask3_review_final_light/eval_results' --split --max_word_len 3500 --pick_num 1000
```

## Data

Download the version 2 data from [here](https://drive.google.com/file/d/1cPh_z_yMAjIpC6XIJn1xnLas39lSzGQd/view?usp=sharing)

Put the `ICLR_2023` folder under the `./subtask3_review_final_v2`

Now, there is only one folder `ICLR_2023`:
```
subtask3_review_final_v2
    - ICLR_2023
    - README.md
```


## run the experiments:

I use the following cmd to run openai gpt:
```bash
python scripts/subtask3_review_model_prediction.close_source.py --api_name 'gpt-4o' --root_dir './subtask3_review_final_v2' --save_dir './subtask3_review_final_v2/eval_results' --split --max_word_len 3000 
```

Pls adjust the line 186 in file `subtask3_review_model_prediction.close_source.py` to use your claude and gemini query function.
import argparse
import json
import re
import time
import pandas as pd
from tqdm import tqdm
import ast
import os
from typing import Optional, Sequence, Union, List
import dataclasses
import openai
from openai import OpenAI

import litellm
litellm.drop_params=True  # allow litellm to drop the parameters that are not supported by the model
# litellm.set_verbose=True  # for debugging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
MAX_RETRY = 3
# template = '''
# You are given a paragraph, which is a review comment made by a reviewer of a research paper. This review comment might contain the 'Strengths',  and 'Weakness' and some questions regarding this paper.

# Please help me extract and summarize the **Weakness** from this review comment, i.e., the reason why the reviewer wants to lower the score of this paper / reject this paper.

# The following is the review comment:
# ```
# {{review_comment}}
# ```

# Please give me a list of summarized weaknesses (in a "1. xxx; 2. xxx; 3. xxx" format). Each item in this list is an **individual** point that briefly mentions one weakness of this paper, which means you are encouraged to merge multiple similar points into a concise one.
# '''

template = '''You are given a paragraph, which is a review comment made by a reviewer of a research paper. This review comment might contain the 'Strengths', and 'Weakness' and some other questions regarding this paper.

Please help me extract the **Weaknesses** from this review comment, i.e., the reasons why the reviewer wants to lower the score of this paper / reject this paper.

The following is the review comment:
```
{review_comment}
```

Please give me a list of weaknesses (in a "1. xxx; 2. xxx; 3. xxx" format). Each item in this list is an **individual** point that mentions one weakness of this paper. For each item, please keep the original text of the review comment.
'''

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1600  # make sue to set this value smaller, otherwise, the repeated reduce will be very slow
    temperature: float = 0.9
    top_p: float = 0.99
    n: int = 1
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(16))
def completion_with_backoff(model_name,messages,decoding_args):
    '''
    # Retry with exponential backoff
    # See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    '''
    try:
        result = litellm.completion(model=model_name, messages=messages, **decoding_args)
    except Exception as e:
        if "context_length_exceeded" in str(e):
            # if the context length is exceeded, reduce the max output tokens
            print(f"==> context_length_exceeded, reduce the max output tokens")
            decoding_args["max_tokens"] = max(decoding_args["max_tokens"] // 2, 1)  # TODO: "max_tokens" might only work for GPT, for other models, param name might be different
            time.sleep(3)  # avoid too frequent recursive calls
            return completion_with_backoff(model_name,messages,decoding_args)
        else:
            raise e
    
    return result


def main(args):
    df = pd.read_csv(args.csv)
    # extract "NeurIPS_2021" from NeurIPS_2021_draft_comment.csv
    conf_name = os.path.basename(args.csv).split("_")[:2]
    conf_name = "_".join(conf_name)
    # import pdb; pdb.set_trace()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    def GPT_completion(prompt):
        # completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt,
        #         }
        #     ],
        #     model="gpt-4o",
        #     max_tokens=1800,
        #     temperature=0.9,
        # )
        messages = [
                {"role": "system", "content": "You are an expert in Machine Learning and Natural Language Processing (NLP). Your responsibility is to help the user read the review comments of a research paper and extract the weaknesses of the paper."},
                {"role": "user", "content": prompt},
            ]
        decoding_args = OpenAIDecodingArguments()
        # convert decoding_args to a dictionary
        decoding_args = dataclasses.asdict(decoding_args)
        completion = completion_with_backoff("gpt-4o", messages, decoding_args)

        return completion.choices[0].message.content
    
    # if args.weakness_type == "strength_and_weakness":
    #     prefix = "The following reviews include both strengths and weaknesses of a paper. Extract the weaknesses and keep the original text.\n"
    # elif args.weakness_type == "main_review":
    #     prefix = "The following reviews include comments regarding a paper. Extract the weaknesses comments and keep the original text.\n"

    final_data = []
    for i in tqdm(range(len(df))):

        review_with_weakness_list = ast.literal_eval(df["weakness_filed"][i])

        # if args.seperate_weakness:
        weakness_list = []
        for review in review_with_weakness_list:
            prompt = template.format_map({"review_comment": review})
            # import pdb; pdb.set_trace()
            retry_flag = True
            retry_cnt = 0
            while retry_flag:
                response = GPT_completion(prompt)
                response = response.strip()
                response_list = re.findall(r"\d+\..*", response)
                # import pdb; pdb.set_trace()
                if len(response_list) > 0:
                    retry_flag = False
                else:
                    print(f"empty list, retrying...")
                    retry_cnt += 1
                    if retry_cnt > MAX_RETRY:
                        print(f"retry failed, max retry count reached")
                        response_list = []
                        retry_flag = False
            weakness_list.append(response_list)

        title = df["Title"][i]
        draft_url = df["paper_draft_url"][i]
        id = df["ID"][i]
        url = df["URL"][i]
        
        # import pdb; pdb.set_trace()
        # if all the list in the weakness_list are empty, then skip this paper
        if all([len(weakness) == 0 for weakness in weakness_list]):
            continue
        final_data.append({
            "ID": id,
            "Title": title,
            "URL": url,
            "paper_draft_url": draft_url,
            "Conferece": conf_name,
            "weakness_filed": weakness_list,
            "review_num": len(weakness_list),  # how many reviews this paper has
            "item_num": [len(weakness) for weakness in weakness_list]  # how many weakness items in each review
        })
        # else:
        #     prompt = f"{prefix}:\n{' '.join(review_with_weakness_list)}"
        #     weakness = GPT_completion(prompt)

    # df['extracted_weakness'] = weakness_extrated
    # df.to_csv(f"{args.csv.split('csv')[0]}_weakness.csv")
    target_file = f"{conf_name}_{args.target_file}"
    with open(target_file, "w") as f:
        json.dump(final_data, f, indent=2)
    
    print("==> Conferece: ", conf_name)
    print(f"totally {len(final_data)} papers have extracted weaknesses, and the results are saved to {target_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='csv path')
    # parser.add_argument('--weakness_type', type=str, required=True, help='weakness type: strength_and_weakness, main_review')
    # parser.add_argument('--seperate_weakness', type=bool, default=False,  help='whether the comments will be processed one by one under one paper')
    parser.add_argument('--target_file', type=str, default='all_weakness.json', help='the file to save the extracted weaknesses')
    
    args = parser.parse_args()
    main(args)









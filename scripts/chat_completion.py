
import argparse
import copy
import json
import os
import openai
from openai import OpenAI
import dataclasses
import logging
import tiktoken
from tqdm import tqdm
from typing import Optional, Sequence, Union, List

import litellm
litellm.drop_params=True  # allow litellm to drop the parameters that are not supported by the model

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from prompt_templates import ConversationPrompt
# from generate_attributes import OpenAIDecodingArguments


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """
    Returns the number of tokens used by a list of messages.
    See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-1106-preview",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def construct_prompt_gpt35(input_dic: dict, template: ConversationPrompt, max_tokens=2048, model="gpt-3.5-turbo-0301"):
    '''
    # cut long completion
    # assert the max length of chatgpt is 4096
    # therefore, 4096 = completion (max_tokens) + messages
    '''
    if "16k" in model:
        raise NotImplementedError("we haven't test the 16k version yet, which may result in unexpected errors.")  
    user_content = template.query_prompt.format_map(input_dic)
    messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": user_content}
        ]
    message_tok_num = num_tokens_from_messages(messages=messages, model=model)
    # the sum of tokens of messages and completion should be less than 4096
    if message_tok_num + max_tokens > 4096:
        max_tokens = max(4096 - message_tok_num - 100, 0) # 100 is a buffer
        logging.warning("since the message is too long ({}), reduce the max_tokens of completion to {}".format(message_tok_num, max_tokens))

    return messages, max_tokens


def construct_prompt_gpt4(input_dic: dict, template: ConversationPrompt, max_tokens=2048, model="gpt-4"):
    '''
    # cut long completion
    # assert the max length of gpt-4 is 8192
    # therefore, 8192 = completion (max_tokens) + messages
    '''
    if "32k" in model:
        raise NotImplementedError("we haven't test the 32k version yet, which may result in unexpected errors.")
    user_content = template.query_prompt.format_map(input_dic)
    messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": user_content}
        ]
    message_tok_num = num_tokens_from_messages(messages=messages, model=model)
    # the sum of tokens of messages and completion should be less than 4096
    if message_tok_num + max_tokens > 8192:
        max_tokens = max(8192 - message_tok_num - 100, 0) # 100 is a buffer
        logging.warning("since the message is too long ({}), reduce the max_tokens of completion to {}".format(message_tok_num, max_tokens))

    return messages, max_tokens


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(16))
def completion_with_backoff(model_name,messages,decoding_args):
    '''
    # Retry with exponential backoff
    # See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    '''
    result = litellm.completion(model=model_name, messages=messages, **decoding_args)
    
    return result


def openai_chat_completion(
    client: OpenAI,  # useless
    input_dic: dict,
    template: ConversationPrompt,
    decoding_args,
    model_name="gpt-3.5-turbo-0301",  # TODO: 0301 will be deprecated in the future
    **decoding_kwargs,
):
    '''
    For each input x, do single-turn chat completion
    
    args:
        - input_dic: a dictionary of the input.
        - template: a string template that is waiting for filling in the values in the input_dic.
    return:
        - content: the content of the response
        - cost: the number of tokens used by this completion
        
    return (None, None) if the input is too long (exceeds the max length of ChatGPT)
    '''
    
    user_content = template.query_prompt.format_map(input_dic)
    messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": user_content}
        ]
    
    # convert decoding_args to a dictionary
    decoding_args = dataclasses.asdict(decoding_args)
    # res = litellm.completion(model_name, messages, **decoding_args)
    res = completion_with_backoff(model_name=model_name,messages=messages,decoding_args=decoding_args)
    response = res.choices[0].message.content
    cost = res.usage.total_tokens

    # extract the contents from the response
    content = template.extract_content(response)
    
    # print("==")
    # print(res)
    # print(content)
    # exit()

    return content, cost
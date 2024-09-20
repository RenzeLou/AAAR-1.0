## data

Pls download data from [google drive](https://drive.google.com/file/d/1yuYo8REqwnHuksqdfaoga8GTxlYcYXd6/view?usp=sharing)

unzip data to the `./subtask2_experiment_human_anno`

Each foler within this zip package is the data of a paper (totally 100 papers). Each paper has three types of data:
```
- data_text.json: the cleaned text of the paper
- images: folder of the image of the paper
- XXX.tar.gz: the source Latex package of this paper 
```

## input for the model

Now we are only using the `data_text.json`, where there are two filed you will use:
```
# the input for the model
input:[
        "\\documentclass{article}\n",
        "\\usepackage{microtype}\n",
        "\\usepackage{graphicx}\n",
        "\\usepackage{subfigure}\n",
        ...
]

# the output for the model (two outputs, each is a list)
"output": {
        "What experiments do you suggest doing?": [
            "1. xxx",
            "2. xxx",
            ...
        ],
        "Why do you suggest these experiments?": [
            "1. xxx",
            "2. xxx",
            ...
        ]
    },
```
Pls use the above `input` in each `data_text.json` as the input for the model.

## Two list generation

Pls help me prompt the model for genrating two types of output:

### 1.Experiment generation

the prompt is stored at `./subtask2_experiment_human_anno/prompt_experiment.txt`


Fill in the `input` into this prompt, feed it to the model to generate a list of experiments.

Note that the `input` might be really long, so pls first convert the `input` from a python list to a string, then use the following code to cut the max length of the input:

```python
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')

def cut_word(input_context:str, max_word_len:int):
    words = word_tokenize(input_context)
    # import pdb; pdb.set_trace()
    words = words[:max_word_len]
    cutted_text = TreebankWordDetokenizer().detokenize(words)
    return cutted_text

# read the input from the above data_text.json
input = data["input"]
# first convert the list to the string
input_text = "".join(input)
# then cut the input, in our experiment, we set the max length to 3000
input_text_cut = cut_word(input_text, 3000)
```

Then, since the model's response is a string, pls use the following code to convert the string to a python list:

```python
import re
response = response.strip()
response = re.findall(r"\d+\..*", response)
return response
```

### 2. Explanation generation

the prompt is stored at `./subtask2_experiment_human_anno/prompt_explanation.txt`

here, the explanation list will be generated based on the oracle `output["What experiments do you suggest doing?"]`

Fill in the `input` and `output["What experiments do you suggest doing?"]` into this prompt

similarly, pls use the aforementioned code to cut the `input` (max 3000 words) and convert the response to a python list.


## save all the prediction list from the models

pls save the above two list of **each paper** as a json file, in the following structure:

```json
{
"output": {
        "What experiments do you suggest doing?": [
            "1. xxx",
            "2. xxx",
            ...
        ],
        "Why do you suggest these experiments?": [
            "1. xxx",
            "2. xxx",
            ...
        ]
    },
"predicton": {
        "What experiments do you suggest doing?": [
            "1. xxx",
            "2. xxx",
            ...
        ],
        "Why do you suggest these experiments?": [
            "1. xxx",
            "2. xxx",
            ...
        ]
    }
}
```

I.e., pls keep the `output` in your final prediction json file, and add the `prediction` to the json file, so that I can use them to calculate the metrics.

Then, save the each paper's prediction json file to the path as following

For example, here is your final saving structure:
```
- eval_results 
    - gemini-pro                 # use model name as the subfolder
        - 1902.00751             # paper id
            - eval_results.json  # your prediction json file
        - 1906.01502
            - eval_results.json
        - ...
    - calude-3.5
        - 1902.00751
            - eval_results.json
        - 1906.01502
            - eval_results.json
        - ...
```

namely, each paper's prediction json file are saved under the subfolder named by the paper id.

# This file contains the prompt templates used in chat completion.
import re

# parent class
class ConversationPrompt(object):
    def __init__(self):
        self.system = (
            "You are a creative assistant. " +
            "Your responsibility is to understand the user's instructions and help brainstorm novel ideas."
        )

    def extract_content(self, content:str):
        '''
        remove "```" and the empty space at the begining and end of the content
        '''
        content = content.replace("```", "")
        content = content.strip()
        return content
    
    
    
# used for equation rewritting (difficult option with logic errors)
class EquationRewrite_Difficult(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)." + 
            "Your responsibility is to help the user rewrite the latex equation of an NLP paper."
        )
        self.query_prompt = (
            "### Task:\n" +
            "You are given a latex source code of a displayed equation (i.e., original equation); please rewrite it into another version that contains some errors.\n\n" +
            "### Requirements:\n" +
            "1. Use all the defined notations in the original equation, but add some logic errors to break the original function.\n" +
            "2. Also, avoid adding new notations.\n" +
            "3. Make the wrong version as hard as possible to distinguish from the original equation, which demands more of the reader's knowledge.\n" +
            "4. Only generate the equation. Avoid any other explanations.\n\n" +
            "### Original Equation:\n" +
            "```\n" +
            "{ori_equation}\n" +
            "```\n\n" +
            "### Wrong Version:\n"
        )


# used for equation rewritting (easy option with trivial errors)
class EquationRewrite_Easy(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)." + 
            "Your responsibility is to help the user rewrite the latex equation of an NLP paper."
        )
        self.query_prompt = (
            "### Task:\n" +
            "You are given a latex source code of a displayed equation (i.e., original equation); please rewrite it into another version that contains some errors.\n\n" +
            "### Requirements:\n" +
            "1. Add some trivial errors to the original equation. For example, change the arithmetic operations or replace some functions.\n" +
            "2. Only generate the equation. Avoid any other explanations.\n\n" +
            "### Original Equation:\n" +
            "```\n" +
            "{ori_equation}\n" +
            "```\n\n" +
            "### Wrong Version:\n"
        )
        

# used for equation generation
# can be both used for model prediction (if we regard it as a generation task)
# or wrong equation crafting (if we regard it as a cls task)
class Equation_Generation(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)." + 
            "Your responsibility is to help the user generate the latex equation of an NLP paper."
        )
        self.query_prompt = (
            "### Task:\n" +
            "You are asked to complete the equation in an NLP paper. Given the context before and after an equation, where the equation is deleted, you should help me recover that equation.\n\n" +
            "### Requirements:\n" +
            "1. Give me the latex source code of the missed the equation.\n" +
            "2. Only give me the equation, avoid any other explanations.\n\n" +
            "### Context Before:\n" +
            "```\n" +
            "{context_before}\n" +
            "```\n\n" +
            "### Context After:\n" +
            "```\n" +
            "{context_after}\n" +
            "```\n\n" +
            "### Equation:\n"
            "```\n" +
            "{equation_left_part}"
        )
        
    def extract_content(self, content:str):
        '''
        1. remove "```" and the empty space at the begining and end of the content
        2. also remove "```latex"
        '''
        content = content.replace("```latex", "")
        content = content.replace("```", "")
        content = content.strip()
        return content
    

# equation filtering
# use GPT to filter wrong equations (without accessing contexts)
class Equation_Filtering(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)."
        )
        self.query_prompt = (
            "### Task:\n" +
            "You are given a source code of a latex equation. Based on your knowledge regarding the Machine Learning and NLP, you should help me identify if this equation has obvious flaw.\n\n" +
            "### Requirements:\n" +
            "1. If you think this equation has significant flaws, such as grammar errors, logical errors, or any other issues, please mark it as 'Wrong'.\n" +
            "2. Otherwise, please mark it as 'Correct'.\n" +
            "3. Please only give me either 'Correct' or 'Wrong'. Avoid any other explanations.\n\n" +
            "### Equation:\n"
            "```\n" +
            "{equation}\n" +
            "```\n\n" +
            "### Your Answer:\n"
        )
        
    def extract_content(self, content:str):
        content = content.strip()
        return content
    
    
        

# used for testing model's performance on subtask1
class Equation_eval(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)." + 
            "Your responsibility is to help the user write the correct latex equations."
        )
        self.query_prompt = (
            "### Task:\n" +
            "You are given the latex source code of the context before and after an equation in an NLP paper, while this equation is masked. Your task is to select a correct equation out of four options (A, B ,C ,D).\n\n" +
            "### Requirements:\n" +
            "Only provide the option ID (either A, B, C, or D). Avoid any explanations.\n\n" +
            "### Context Before:\n" +
            "```\n" +
            "{context_before}\n" +
            "```\n\n" +
            "### Context After:\n" +
            "```\n" +
            "{context_after}\n" +
            "```\n\n" +
            "### Options:\n" +
            "{options}\n\n" +
            "### Your Answer:\n"
        )
        # self.query_prompt = (
        #     "### Task:\n" +
        #     "You are given the latex source code of an equation in an NLP paper, while there are four options for this equation. Your task is to select a correct equation out of four options (A, B ,C ,D).\n\n" +
        #     "### Requirements:\n" +
        #     "Only provide the option ID (either A, B, C, or D). Avoid any explanations.\n\n" +
        #     "### Options:\n" +
        #     "{options}\n\n" +
        #     "### Your Answer:\n"
        # )
        

    def extract_content(self, content:str):
        '''
        simply remove the empty space at the begining and end of the content
        '''
        content = content.strip()
        return content
    
    

# used for testing model's performance on subtask1 (none-prediction)
class Equation_eval_none(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)." + 
            "Your responsibility is to help the user write the correct latex equations."
        )
        self.query_prompt = (
            "### Task:\n" +
            "You are given the latex source code of the context before and after an equation in an NLP paper, while this equation is masked. Your task is to select a correct equation out of three options (A, B ,C).\n\n" +
            "### Requirements:\n" +
            "1. Only provide the option ID (either A, B, or C). Avoid any explanations.\n" +
            "2. Note that, it's possible that all the three provided options are wrong. So, if you think all the options are wrong, please answer 'None'.\n\n" +
            "\n\n" +
            "### Context Before:\n" +
            "```\n" +
            "{context_before}\n" +
            "```\n\n" +
            "### Context After:\n" +
            "```\n" +
            "{context_after}\n" +
            "```\n\n" +
            "### Options:\n" +
            "{options}\n\n" +
            "### Your Answer:\n"
        )

    def extract_content(self, content:str):
        '''
        simply remove the empty space at the begining and end of the content
        '''
        content = content.strip()
        return content    


# used for prompting the LLM delete the sentence leaking the experiment ideas.
class Experiment_leak(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP)." + 
            "Your responsibility is to help the user read a scientific paper."
        )
        self.query_prompt = (
            "You are given a sentence (or a short paragraph) from an NLP paper, along with a list of the experiments from this paper; help me decide whether this sentence discusses any experiments in the list.\n\n" +
            "Let's say, if one sentence includes clues for coming up with any experiments in the list, we call this sentence a 'leaking sentence'; otherwise, if any experiment ideas cannot be inferred from the sentence, we call it a 'non-leak sentence'.\n\n" +
            "Please give me a '1' if this sentence is a 'leaking sentence'; otherwise, give me a '0'.\n\n" +
            "### Experiment List:\n" +
            "```\n" +
            "{experiment_list}\n" +
            "```\n\n" +
            "### Sentence:\n" +
            "```\n" +
            "{sentence}\n" +
            "```\n\n" +
            "Now, give me your decision (give me either '0' or '1', only the number, without any explanations):"
        )

    def extract_content(self, content:str):
        '''
        simply remove the empty space at the begining and end of the content
        '''
        content = content.strip()
        return content   
    
    
# used for prompting the model to generate experiment list in subtask2
class Exp_eval(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP). " +
            "Your responsibility is to help the user design experiments and develop new ideas."
        )
        self.query_prompt = (
            "You are partially given an NLP paper (in latex), including some useful sections (e.g., 'abstract' and 'introduction') having some basic introductions to the research of this paper, where all the 'experiment' related sections are deleted.\n\n" +
            "Please first help me carefully read these sections and try to understand the motivations of this research, such as 'what the authors are trying to propose/demonstrate?' and 'what are the main contributions/differences of this paper from others?'\n\n" +
            "Then, based on your in-depth understanding of this paper, imagine that you are the authors of this paper; what experiments do you have to conduct to prove your research? Namely, you have to **recover the deleted experiments** by providing me with **a list of experiment ideas**, where the list briefly summarizes the experiments the authors should conduct.\n\n" +
            "Here is an example:\n" +
            "```\n" +
            "1. Cross-label generalisation comparison with the previous works. Since this work proposes a new method for open-domain relation type discovery, the authors should test this idea on some widely used benchmarks, such as FewRel.\n" +
            "2. Semantic representation visualisation on the test set. The authors should conduct further visualisation by using some dimension-reduce methods (e.g., t-SNE) on the widely adopted cross-label test set (e.g., FewRel).\n" +
            "3. and so on ...\n" +
            "```\n\n" +
            "Here is the target NLP paper (partial content):\n" +
            "```\n" +
            "{context_input}\n" +
            "```\n\n" +
            "Now, based on this paper, give me a list of experiments the author has to do. Please only give me the list, without any other words.\n\n" +
            "### Your Experiment List:\n"+
            "```\n"
        )

    def extract_content(self, content:str):
        '''
        extract the list from the model's response
        
        for example, the ori response is:
        
        "
        ### Your Experiment List:
        ```  
        1. Performance comparison on standard NLP benchmarks (e.g., GLUE, SQuAD) between adapter-based tuning and fine-tuning.
        2. Parameter efficiency evaluation by measuring the number of parameters required per task for adapter-based tuning vs fine-tuning.
        ```
        "
        
        the final extracted content is a list:
        [
            "1. Performance comparison on standard NLP benchmarks (e.g., GLUE, SQuAD) between adapter-based tuning and fine-tuning.",
            "2. Parameter efficiency evaluation by measuring the number of parameters required per task for adapter-based tuning vs fine-tuning."
        ]
        
        use re to extract the list
        '''
        content = content.strip()
        content = re.findall(r"\d+\..*", content)
        return content



# used for prompting the model to generate explanation list in subtask2
class Exp_explanation_eval(ConversationPrompt):
    def __init__(self):
        super().__init__()
        self.system = (
            "You are an expert in Machine Learning and Natural Language Processing (NLP). " +
            "Your responsibility is to help the user undertand a paper."
        )
        self.query_prompt = (
            "You are partially given an NLP paper (in latex), including some useful sections (e.g., 'abstract' and 'introduction') having some basic introductions to this research, where all the 'experiment' related sections are deleted.\n\n" +
            "Meanwhile, you are also given a list of experiments that try to predict the missed experiments in this paper.\n\n" +
            "Now, imagine the experiment list you created; you have to explain **why you suggested these experiments**.\n\n" +
            "Here is an example experiment list:\n" +
            "```\n" +
            "1. Cross-label generalisation comparison with the previous works. Since this work proposes a new method for open-domain relation type discovery, the authors should test this idea on some widely used benchmark, such as FewRel.\n" +
            "2. Semantic representation visualisation on the test set. The authors should conduct further visualisation by using some dimension-reduce methods (e.g., t-SNE) on the widely adopted cross-label test set (e.g., FewRel).\n" +
            "```\n" +
            "Here is the example corresponding explanation list:\n" +
            "```\n" +
            "1. To support the effectiveness of deep metric learning compared with the unsupervised algorithm.\n" +
            "2. To demonstrate the explainability and robustness of the proposed semi-supervised algorithm.\n" +
            "```\n\n" +
            "Now, help me look at the following paper:\n" +
            "### Paper:\n" +
            "```\n" +
            "{context_input}\n" +
            "```\n\n" +
            "### Experiment List:\n" +
            "```\n" +
            "{experiment_list}\n" +
            "```\n\n" +
            "Please give me your explanation list, which should be the same length as the 'Experiment List'; the items of the two lists correspond one-to-one. Only give me the list without any other useless words.\n" +
            "### Explanation List:\n"
        )

    def extract_content(self, content:str):
        '''
        use re to extract the list
        '''
        content = content.strip()
        content = re.findall(r"\d+\..*", content)
        return content



if __name__ == "__main__":
    # test the prompt templates
    # eq_eval_template = Equation_eval()
    # value_dic = {
    #     "context_before": "The context before the equation.",
    #     "context_after": "The context after the equation.",
    #     "options": "The options for the equation."
    # }
    
    # print(eq_eval_template.query_prompt.format(**value_dic))
    
    # difficult_prompt = EquationRewrite_Difficult()
    # value_dic = {
    #     "ori_equation": "The original equation."
    # }
    # print(difficult_prompt.query_prompt.format(**value_dic))
    
    
    
    # equation_gen = Equation_Generation()
    # value_dic = {
    #     "context_before": "The context before the equation.",
    #     "context_after": "The context after the equation.",
    #     "equation_left_part": "P = "
    # }
    # print(equation_gen.query_prompt.format(**value_dic))
    
    
    # equation_identify = Equation_Filtering()
    # value_dic = {
    #     "equation": "The equation."
    # }
    # print(equation_identify.query_prompt.format(**value_dic))
    
    # exp_eval = Exp_eval()
    # value_dic = {
    #     "context_input": "The context input."
    # }
    # print(exp_eval.query_prompt.format(**value_dic))
    
    exp_explanation_eval = Exp_explanation_eval()
    value_dic = {
        "context_input": "The context input.",
        "experiment_list": "The experiment list."
    }
    print(exp_explanation_eval.query_prompt.format(**value_dic))
    # response = '''
    # ### Your Experiment List:
    # ```  
    # 1. Performance comparison on standard NLP benchmarks (e.g., GLUE, SQuAD) between adapter-based tuning and fine-tuning.
    # 2. Parameter efficiency evaluation by measuring the number of parameters required per task for adapter-based tuning vs fine-tuning.
    # ```
    # '''
    # print(exp_eval.extract_content(response))
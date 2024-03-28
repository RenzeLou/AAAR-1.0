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
    
    
    equation_identify = Equation_Filtering()
    value_dic = {
        "equation": "The equation."
    }
    print(equation_identify.query_prompt.format(**value_dic))
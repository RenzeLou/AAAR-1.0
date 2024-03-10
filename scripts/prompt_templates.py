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

    def extract_content(self, content:str):
        '''
        simply remove the empty space at the begining and end of the content
        '''
        content = content.strip()
        return content
import re
from typing import Callable


class AnswerTrigger:
    def __init__(self, trigger: str, split_index: int = 1):
        self.trigger = trigger
        self.split_index = split_index

    def __call__(self, text):
        if self.trigger and self.trigger in text:
            return text.split(self.trigger)[self.split_index]
        return text


class Matcher:
    def __init__(self, regrex: str = r'A|B|C|D|E', reverse: bool = False):
        self.regrex = regrex
        self.reverse = reverse

    def __call__(self, text):
        preds = re.findall(self.regrex, text)
        if len(preds) == 0:
            return ""
        if self.reverse:
            return preds[-1]
        return preds[0]


class AnswerCleaner:
    def __init__(self, trigger: Callable, matcher: Callable, remove_input: bool = True):
        self.trigger = trigger
        self.matcher = matcher
        self.remove_input = remove_input

    def __call__(self, text: str, input: str):
        if self.remove_input and input.strip() in text:
            text = text.replace(input.strip(), "")
        text = self.trigger(text)
        text = self.matcher(text)
        return text


# BIG-Bench Hard
bbh_regrex = {
    "boolean_expressions": "True|False",
    "data_understanding": "(A)|(B)|(C)|(D)|(E)|(F)",
    "disambiguation_qa": "(A)|(B)|(C)",
    "dyck_languages": "",
    "formal_fallacies": "valid|invalid",
    "geometric_shapes": "(A)|(B)|(C)|(D)|(E)|(F)|(G)|(H)|(I)|(J)|(K)|(L)|(M)|(N)|(O)|(P)|(Q)|(R)|(S)|(T)|(U)|(V)|(W)|(X)|(Y)|(Z)",

}

BBH_MULTIPLE_CHOICE_TASKS = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table',
    'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects',
    'movie_recommendation', 'salient_translation_error_detection', 'reasoning_about_colored_objects',
]

BBH_FREE_FORM_TASKS = [
    'dyck_languages', 'word_sorting',
]

BBH_OTHER_TASKS = {
    "navigate": "Yes|No|yes|no",
    "sports_understanding": "yes|no|Yes|No",
    "boolean_expressions": "True|False",
    "object_counting": r'-?\d+\.?\d*',  # number matching,
    "multistep_arithmetic_two": r'-?\d+\.?\d*',
    "formal_fallacies": "valid|invalid",
    "causal_judgement": "Yes|No|yes|no",
    "web_of_lies": "Yes|No|yes|no",
}


class BBHMatcher:
    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def __call__(self, text, mode):
        if mode in BBH_MULTIPLE_CHOICE_TASKS:
            # A to Z
            regrex = "A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z"
        elif mode in BBH_FREE_FORM_TASKS:
            regrex = ""
        elif mode in BBH_OTHER_TASKS:
            regrex = BBH_OTHER_TASKS[mode]
        else:
            raise ValueError(f"Mode {mode} not found in BBH tasks")

        if not regrex:
            return text

        preds = re.findall(regrex, text)
        if len(preds) == 0:
            return ""

        if self.reverse:
            return preds[-1]
        return preds[0]

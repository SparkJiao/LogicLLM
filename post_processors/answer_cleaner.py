import re
from typing import Callable


class AnswerTrigger:
    def __init__(self, trigger: str, split_index: int = 1):
        self.trigger = trigger
        self.split_index = split_index

    def __call__(self, text):
        if self.trigger in text:
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

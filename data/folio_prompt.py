import json
import random
from typing import List

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from general_util.tokenization_utils import expand_special_tokenizer
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

_template = {
    "base": "Premises:\n{}\n\nConclusion: {}\n\nLabel:",
    "value": "Premises:\n{}\n\nConclusion: {}\n\nThe truth value of the conclusion is: ",
    "mcqa": "Premises:\n{}\n\nConclusion: {}\n\nThe truth value options of the conclusion is:\nA. True\nB. False\nC. Uncertain\n\n"
            "The truth answer is:",
    "mcqa2": "Premises:\n{}\n\nConclusion: {}\n\nQuestion: What is the truth value of the conclusion?\n\n"
             "Options:\nA. True\nB. False\nC. Uncertain\n\nThe answer is: "
}


def get_template_by_name(name):
    if name in _template:
        return _template[name]
    else:
        raise NotImplementedError


_instruction = {
    "base": "Read following premises, select the correct truth values from {True, False, Uncertain} for the given conclusion,  "
            "where ``True`` means\n\n the conclusion can be logically inferred from the premises, "
            "``False`` means the conclusion cannot be logically inferred from the premises, "
            "and ``Uncertain`` means the premises are not sufficient to determine the truth value of the conclusion.\n\n",
    "base-1": "Read following premises, select the correct truth values from {True, False, Uncertain} for the given conclusion,  "
              "where True means\n\n the conclusion can be logically inferred from the premises, "
              "False means the conclusion cannot be logically inferred from the premises, "
              "and Uncertain means the premises are not sufficient to determine the truth value of the conclusion.\n\n",
    "simple": "Read the following premises and conclusion, decide the correct truth value of the conclusion.\n\n",
}


def get_instruction_by_name(name):
    if name in _instruction:
        return _instruction[name]
    else:
        raise NotImplementedError


_suffix = {
}


def get_suffix_by_name(name):
    if name in _suffix:
        return _suffix[name]
    else:
        raise NotImplementedError


class FolioPromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func,
                 prompt_template: str = _template["base"],
                 instruction: str = _instruction["base"],
                 suffix: str = "", ):
        all_premises, all_hypotheses, all_labels = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_premises)):
            self.inputs.append(prompt_template.format(all_premises[i], all_hypotheses[i]))
            self.indices.append(i)
            self.labels.append(all_labels[i])

        self.instruction = instruction
        self.suffix = suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input": self.instruction + self.inputs[index] + self.suffix,
            "index": self.indices[index],
            "label": self.labels[index],
            "prompt_index": "0",
        }

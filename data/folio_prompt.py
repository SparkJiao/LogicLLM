import json
import random
from typing import List

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from general_util.tokenization_utils import expand_special_tokenizer, is_seq2seq_tokenizer
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

_candidate_template = {
    "base": "Premises:\n{}\n\nConclusion: {}\n\nTruth value:{}",
    "value": "Premises:\n{}\n\nConclusion: {}\n\nThe truth value of the conclusion is: {}",
}

FOLIO_OPTIONS = ["True", "False", "Uncertain"]


def get_template_by_name(name):
    if name in _template:
        return _template[name]
    else:
        raise NotImplementedError


def get_candidate_template_by_name(name):
    if name in _candidate_template:
        return _candidate_template[name]
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


def get_exemplar_prompt(file_path: str, read_func, n_shot, prompt_template: str, sample: bool = False):
    all_premises, all_hypothesis, all_labels = read_func(file_path)

    if sample:
        indices = random.sample(range(len(all_premises)), n_shot)
    else:
        indices = list(range(n_shot))

    inputs = []
    for i in indices:
        inputs.append(prompt_template.format(all_premises[i], all_hypothesis[i], all_labels[i]))

    return "\n\n".join(inputs) + "\n\n"


class FolioPromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, exemplar: str = "",
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
        self.exemplar = exemplar
        self.suffix = suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input": self.instruction + self.exemplar + self.inputs[index] + self.suffix,
            "index": self.indices[index],
            "label": self.labels[index],
            "prompt_index": "0",
        }


class FolioCandidatePromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func,
                 prompt_template: str = _candidate_template["base"],
                 instruction: str = _instruction["base"],
                 suffix: str = "", ):
        all_premises, all_hypotheses, all_labels = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_premises)):
            self.inputs.append([
                prompt_template.format(all_premises[i], all_hypotheses[i], op) for op in FOLIO_OPTIONS
            ])
            self.indices.append(i)
            self.labels.append(all_labels[i])

        self.instruction = instruction
        self.suffix = suffix
        self.label2map = {label: i for i, label in enumerate(FOLIO_OPTIONS)}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input": [self.instruction + self.inputs[index][i] + self.suffix for i in range(len(self.inputs[index]))],
            "index": self.indices[index],
            "label": self.label2map[self.labels[index]],
            "prompt_index": "0",
        }


class CandidateGenerativeCollator:
    def __init__(self, tokenizer: str, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.is_seq2seq = is_seq2seq_tokenizer(self.tokenizer)

    def __call__(self, batch):
        inputs = [b.pop("input") for b in batch]
        outputs = [b.pop("output") for b in batch]
        batch = default_collate(batch)
        flat_inputs = []
        flat_outputs = []
        for i in range(len(inputs)):
            flat_inputs.extend(inputs[i])
            flat_outputs.extend(outputs[i])
        if self.is_seq2seq:
            model_inputs = self.tokenizer(flat_inputs, text_target=flat_outputs,
                                          padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        else:
            model_inputs = self.tokenizer(flat_inputs,
                                          padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if "token_type_ids" in model_inputs:
            model_inputs.pop("token_type_ids")
        model_inputs["meta_data"] = batch
        model_inputs["meta_data"]["input"] = inputs
        return model_inputs

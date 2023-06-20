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
    "base-2": "Premises:\n{}\nConclusion: {}\nTruth value:{}",
    "value": "Premises:\n{}\n\nConclusion: {}\n\nThe truth value of the conclusion is: {}",
    "logic-lm": "Context:\n{}\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? {}\n\n"
                "Options:\nA. True\nB. False\nC. Uncertain\n\nThe correct option is: {}",
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
    "simple-2": "Read the following premises and conclusion, decide the truth value of the conclusion is True, False, or Uncertain.\n\n",
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

    return inputs


def compose_prompts(prompts: List[str]) -> str:
    return "\n\n".join(prompts) + "\n\n"


def truncate_prompt(prompts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 2048,
                    instruction: str = "", op_input: str = "", suffix: str = "") -> str:
    while True:
        _input = instruction + compose_prompts(prompts) + op_input + suffix
        if len(tokenizer(_input)["input_ids"]) <= max_length:
            break
        prompts = prompts[:-1]
    return _input


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
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, exemplar: List[str] = None, max_seq_length: int = 2048,
                 prompt_template: str = _candidate_template["base"],
                 instruction: str = _instruction["base"],
                 suffix: str = "",
                 folio_options: List[str] = FOLIO_OPTIONS):
        all_premises, all_hypotheses, all_labels = read_func(file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.inputs = []
        self.outputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_premises)):
            self.inputs.append([
                prompt_template.format(all_premises[i], all_hypotheses[i], op) for op in folio_options
            ])
            self.outputs.append([op for op in folio_options])
            self.indices.append(i)
            self.labels.append(all_labels[i])

        self.instruction = instruction
        self.exemplar = exemplar
        self.suffix = suffix
        self.label2map = {label: i for i, label in enumerate(FOLIO_OPTIONS)}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.exemplar:
            _input = [truncate_prompt(self.exemplar, self.tokenizer, self.max_seq_length, self.instruction, self.inputs[index][i],
                                      self.suffix) for i in range(len(self.inputs[index]))]
        else:
            _input = [self.instruction + self.inputs[index][i] + self.suffix for i in range(len(self.inputs[index]))]
        return {
            "input": _input,
            "index": self.indices[index],
            "label": self.label2map[self.labels[index]],
            "output": self.outputs[index],
            "prompt_index": "0",
        }


# class FolioCandidatePromptGeneratorSingle(Dataset):
#     def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, exemplar: List[str] = None, max_seq_length: int = 2048,
#                  prompt_template: str = _candidate_template["base"],
#                  instruction: str = _instruction["base"],
#                  suffix: str = "", ):
#         all_premises, all_hypotheses, all_labels = read_func(file_path)
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#
#         choices = ["A", "B", "C", "D"]
#         label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
#
#         self.inputs = []
#         self.outputs = []
#         self.indices = []
#         self.labels = []
#         for i in range(len(all_premises)):
#             self.inputs.append([
#                 prompt_template.format(all_premises[i], all_hypotheses[i], op) for op in choices
#             ])
#             self.outputs.append([op for op in FOLIO_OPTIONS])
#             self.indices.append(i)
#             self.labels.append(all_labels[i])
#
#         self.instruction = instruction
#         self.exemplar = exemplar
#         self.suffix = suffix
#         self.label2map = {label: i for i, label in enumerate(FOLIO_OPTIONS)}
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, index):
#         if self.exemplar:
#             _input = [truncate_prompt(self.exemplar, self.tokenizer, self.max_seq_length, self.instruction, self.inputs[index][i],
#                                       self.suffix) for i in range(len(self.inputs[index]))]
#         else:
#             _input = [self.instruction + self.inputs[index][i] + self.suffix for i in range(len(self.inputs[index]))]
#         return {
#             "input": _input,
#             "index": self.indices[index],
#             "label": self.label2map[self.labels[index]],
#             "output": self.outputs[index],
#             "prompt_index": "0",
#         }


class CandidateGenerativeCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
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


class CandidateSelectionCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_size="left", **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.is_seq2seq = is_seq2seq_tokenizer(self.tokenizer)

    def __call__(self, batch):
        inputs = [b.pop("input") for b in batch]
        outputs = [b.pop("output") for b in batch]
        batch = default_collate(batch)
        if self.is_seq2seq:
            model_inputs = self.tokenizer(inputs, text_target=["" for _ in inputs],
                                          padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        else:
            model_inputs = self.tokenizer(inputs,
                                          padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if "token_type_ids" in model_inputs:
            model_inputs.pop("token_type_ids")
        model_inputs["meta_data"] = batch
        model_inputs["meta_data"]["input"] = inputs
        model_inputs["meta_data"]["output"] = outputs
        return model_inputs

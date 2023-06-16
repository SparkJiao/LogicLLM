import collections
import json
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from general_util.tokenization_utils import expand_special_tokenizer, is_seq2seq_tokenizer
from general_util.logger import get_child_logger
from glob import glob
import pandas as pd

logger = get_child_logger(__name__)


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


class MMLUReader:
    """
    Most are copied from https://github.com/hendrycks/test/blob/master/evaluate.py
    """
    label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    choices = ["A", "B", "C", "D"]

    # def format_example(self, df, idx, include_answer=True):
    #     prompt = df.iloc[idx, 0]
    #     k = df.shape[1] - 2
    #     for j in range(k):
    #         prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])
    #     prompt += "\nAnswer:"
    #     if include_answer:
    #         prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    #     return prompt

    def __call__(self, file_dir):
        all_questions = []
        all_option_list = []
        all_label = []
        all_subjects = []

        files = list(glob(file_dir + "/*.csv"))
        for file in files:
            tmp = file.split("/")[-1].split("_")[:-1]
            subject = " ".join(tmp)
            # df = pd.read_csv(file)
            df = pd.read_csv(file, header=None)  # FIXED in 0610. Previous results are all 4-shot.
            for i in range(df.shape[0]):
                all_questions.append(df.iloc[i, 0])
                k = df.shape[1] - 2
                options = []
                for j in range(k):
                    options.append("{}. {}".format(self.choices[j], df.iloc[i, j + 1]))
                all_option_list.append(options)
                all_label.append(df.iloc[i, k + 1])
                all_subjects.append(subject)

        return all_questions, all_option_list, all_label, all_subjects


instructions = {
    "base": "The following are multiple choice questions (with answers) about {}.\n\n",
}

candidate_templates = {
    "base": "{}\n\nOptions:\n{}\n\nAnswer: {}",
    "base_seq2seq": "{}\n\nOptions:\n{}\n\nAnswer:",
    "base1": "{}\n{}\nAnswer: {}",
    # check `lm_eval_harness` implementation here:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py
    "lm_eval_harness": "Question: {}\nChoices: {}\nAnswer: {}",
    # Check the prompt here:
    # https://github.com/FranxYao/chain-of-thought-hub/blob/main/MMLU/run_mmlu_llama.py
    "mmlu_official": "{}\n{}\nAnswer: {}",
    "mmlu_official_2": "{}\n{}\nThe answer is {}",
}


def get_instruction_by_name(name: str) -> str:
    if name in instructions:
        return instructions[name]
    else:
        raise ValueError("Instruction name {} not found.".format(name))


def get_template_by_name(name: str) -> str:
    if name in candidate_templates:
        return candidate_templates[name]
    else:
        raise ValueError("Template name {} not found.".format(name))


def read_direct_prompt(file_path, read_func=MMLUReader(), prompt_template: str = candidate_templates["base"], k: int = 5):
    all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
    # choices = ["A", "B", "C", "D"]
    # label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    all_prompts = collections.defaultdict(list)
    for i in range(len(all_questions)):
        prompt = prompt_template.format(all_questions[i], "\n".join(all_option_list[i]), all_labels[i])
        all_prompts[all_subjects[i]].append(prompt)

    prompts = {}
    for subject in all_prompts:
        prompts[subject] = all_prompts[subject][:k]

    from collections import Counter
    counter = Counter()
    for subject in prompts:
        counter[len(prompts[subject])] += 1
    print(counter)

    return prompts


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


class MMLUPromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func=MMLUReader(), max_seq_length: int = 2048,
                 prompt_template: str = candidate_templates["base"],
                 instruction: str = instructions["base"],
                 suffix: str = "",
                 exemplars: Dict[str, List[str]] = {}):
        all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
        choices = ["A", "B", "C", "D"]
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.inputs = []
        self.indices = []
        self.subjects = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_questions)):
            if is_seq2seq:
                self.inputs.append([prompt_template.format(all_questions[i], "\n".join(all_option_list[i]))] * len(all_option_list[i]))
            else:
                self.inputs.append([
                    prompt_template.format(all_questions[i], "\n".join(all_option_list[i]), choices[op_id])
                    for op_id in range(len(all_option_list[i]))
                ])
            self.outputs.append([choices[op_id] for op_id in range(len(all_option_list[i]))])
            self.indices.append(i)
            self.labels.append(label2id[all_labels[i]])
            self.subjects.append(all_subjects[i])

        self.instruction = instruction
        self.exemplars = exemplars
        self.suffix = suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.exemplars:
            _input = [
                # self.instruction.format(self.subjects[index]) + self.exemplars[self.subjects[index]] + op_input + self.suffix
                truncate_prompt(self.exemplars[self.subjects[index]], self.tokenizer, max_length=self.max_seq_length,
                                instruction=self.instruction.format(self.subjects[index]),
                                op_input=op_input, suffix=self.suffix)
                for op_input in self.inputs[index]
            ]
        else:
            _input = [
                self.instruction.format(self.subjects[index]) + op_input + self.suffix
                for op_input in self.inputs[index]
            ]
        return {
            "input": _input,
            "output": self.outputs[index],
            "index": self.indices[index],
            "label": self.labels[index],
            "category": self.subjects[index],
        }


class MMLUPromptGeneratorFlat(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func=MMLUReader(), max_seq_length: int = 2048,
                 prompt_template: str = candidate_templates["base"],
                 instruction: str = instructions["base"],
                 suffix: str = "",
                 exemplars: Dict[str, List[str]] = {}):
        all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
        choices = ["A", "B", "C", "D"]
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.inputs = []
        self.indices = []
        self.subjects = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_questions)):
            if is_seq2seq:
                self.inputs.extend([prompt_template.format(all_questions[i], "\n".join(all_option_list[i]))] * len(all_option_list[i]))
            else:
                self.inputs.extend([
                    prompt_template.format(all_questions[i], "\n".join(all_option_list[i]), choices[op_id])
                    for op_id in range(len(all_option_list[i]))
                ])
            self.outputs.extend([choices[op_id] for op_id in range(len(all_option_list[i]))])
            self.indices.extend([f"{i}_{op_id}" for op_id in range(len(all_option_list[i]))])
            self.labels.extend([label2id[all_labels[i]]] * len(all_option_list[i]))
            self.subjects.extend([all_subjects[i]] * len(all_option_list[i]))

        self.instruction = instruction
        self.exemplars = exemplars
        self.suffix = suffix

        assert len(self.inputs) == len(self.outputs) == len(self.indices) == len(self.labels) == len(self.subjects)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.exemplars:
            _input = truncate_prompt(self.exemplars[self.subjects[index]], self.tokenizer, max_length=self.max_seq_length,
                                     instruction=self.instruction.format(self.subjects[index]),
                                     op_input=self.inputs[index], suffix=self.suffix)
        else:
            _input = self.instruction.format(self.subjects[index]) + self.inputs[index] + self.suffix
        return {
            "input": _input,
            "output": self.outputs[index],
            "index": self.indices[index],
            "label": self.labels[index],
            "category": self.subjects[index],
        }


class MMLUPromptGeneratorSingle(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func=MMLUReader(), max_seq_length: int = 2048,
                 prompt_template: str = candidate_templates["base"],
                 instruction: str = instructions["base"],
                 suffix: str = "",
                 exemplars: Dict[str, List[str]] = {}):
        all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
        choices = ["A", "B", "C", "D"]
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if prompt_template[-3:] == " {}":
            prompt_template = prompt_template[:-3]

        self.inputs = []
        self.indices = []
        self.subjects = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_questions)):
            flat_options = "\n".join(all_option_list[i])
            self.inputs.append(prompt_template.format(all_questions[i], flat_options))
            _full_input = prompt_template + " {}"
            self.outputs.append([self.tokenizer.tokenize(_full_input.format(all_questions[i], flat_options, choices[op_id]))[-1]
                                 for op_id in range(len(all_option_list[i]))])
            self.indices.append(i)
            self.labels.append(label2id[all_labels[i]])
            self.subjects.append(all_subjects[i])

        self.instruction = instruction
        self.exemplars = exemplars
        self.suffix = suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.exemplars:
            _input = truncate_prompt(self.exemplars[self.subjects[index]], self.tokenizer, max_length=self.max_seq_length,
                                     instruction=self.instruction.format(self.subjects[index]),
                                     op_input=self.inputs[index], suffix=self.suffix)
        else:
            _input = self.instruction.format(self.subjects[index]) + self.inputs[index] + self.suffix

        return {
            "input": _input,
            "output": self.outputs[index],
            "index": self.indices[index],
            "label": self.labels[index],
            "category": self.subjects[index],
        }


class MMLUPromptGeneratorSingleV2(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func=MMLUReader(), max_seq_length: int = 2048,
                 prompt_template: str = candidate_templates["base"],
                 instruction: str = instructions["base"],
                 suffix: str = "",
                 exemplars: Dict[str, List[str]] = {}):
        all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
        choices = ["A", "B", "C", "D"]
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        is_seq2seq = is_seq2seq_tokenizer(tokenizer)
        logger.info("{} is seq2seq tokenizer: {}".format(tokenizer.__class__.__name__, is_seq2seq))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if prompt_template[-3:] == " {}":
            prompt_template = prompt_template[:-3]

        self.inputs = []
        self.indices = []
        self.subjects = []
        self.labels = []
        self.outputs = []
        for i in range(len(all_questions)):
            flat_options = "\n".join(all_option_list[i])
            self.inputs.append(prompt_template.format(all_questions[i], flat_options))
            # The following writing is ugly. Another way to implement this is use specific logits processor and call `generate` method.
            _full_input_1 = prompt_template + " {}"
            tmp = [(self.tokenizer.tokenize(_full_input_1.format(all_questions[i], flat_options, choices[op_id]))[-1], op_id)
                   for op_id in range(len(all_option_list[i]))]
            _full_input_2 = prompt_template + "{}"
            tmp += [(self.tokenizer.tokenize(_full_input_2.format(all_questions[i], flat_options, choices[op_id]))[-1], op_id)
                    for op_id in range(len(all_option_list[i]))]
            self.outputs.append(tmp)
            self.indices.append(i)
            self.labels.append(label2id[all_labels[i]])
            self.subjects.append(all_subjects[i])

            self.instruction = instruction
            self.exemplars = exemplars
            self.suffix = suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.exemplars:
            _input = truncate_prompt(self.exemplars[self.subjects[index]], self.tokenizer, max_length=self.max_seq_length,
                                     instruction=self.instruction.format(self.subjects[index]),
                                     op_input=self.inputs[index], suffix=self.suffix)
        else:
            _input = self.instruction.format(self.subjects[index]) + self.inputs[index] + self.suffix

        return {
            "input": _input,
            "output": self.outputs[index],
            "index": self.indices[index],
            "label": self.labels[index],
            "category": self.subjects[index],
        }

# class MMLUDirectPromptGenerator(Dataset):
#     def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func=MMLUReader(),
#                  prompt_template: str = candidate_templates["base"],
#                  instruction: str = instructions["base"],
#                  suffix: str = "", ):
#         all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
#         choices = ["A", "B", "C", "D"]
#         label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
#
#         self.inputs = []
#         self.indices = []
#         self.subjects = []
#         self.labels = []
#         for i in range(len(all_questions)):
#             self.inputs.append([
#                 prompt_template.format(all_questions[i], "\n".join(all_option_list[i]), choices[op_id])
#                 for op_id in range(len(all_option_list[i]))
#             ])
#             self.indices.append(i)
#             self.labels.append(label2id[all_labels[i]])
#             self.subjects.append(all_subjects[i])
#
#         self.instruction = instruction
#         self.suffix = suffix
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, index):
#         return {
#             "input": [self.instruction.format(self.subjects[index]) + op_input + self.suffix for op_input in self.inputs[index]],
#             "index": self.indices[index],
#             "label": self.labels[index],
#             "category": self.subjects[index],
#         }

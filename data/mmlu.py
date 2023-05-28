import json
import random
from typing import List

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from general_util.tokenization_utils import expand_special_tokenizer
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
            df = pd.read_csv(file)
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
    "base": "{}\n\nOptions:\n{}\n\nAnswer: {}"
}


class MMLUPromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func=MMLUReader(),
                 prompt_template: str = candidate_templates["base"],
                 instruction: str = instructions["base"],
                 suffix: str = "", ):
        all_questions, all_option_list, all_labels, all_subjects = read_func(file_path)
        choices = ["A", "B", "C", "D"]
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

        self.inputs = []
        self.indices = []
        self.subjects = []
        self.labels = []
        for i in range(len(all_questions)):
            self.inputs.append([
                prompt_template.format(all_questions[i], "\n".join(all_option_list[i]), choices[op_id])
                for op_id in range(len(all_option_list[i]))
            ])
            self.indices.append(i)
            self.labels.append(label2id[all_labels[i]])
            self.subjects.append(all_subjects[i])

        self.instruction = instruction
        self.suffix = suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input": [self.instruction.format(self.subjects[index]) + op_input + self.suffix for op_input in self.inputs[index]],
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


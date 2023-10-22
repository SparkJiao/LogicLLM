import json
import omegaconf
import json
import random
from typing import List, Dict, Tuple, Union, Any, Callable

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer

from data.collators.flan import convert_to_standard_inputs
from data.cot_critic import vanilla_seq2seq_convertor as vanilla_seq2seq_converter_text
from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer, is_seq2seq_tokenizer
from omegaconf.listconfig import ListConfig

logger = get_child_logger(__name__)


def read_example_files(file_list: ListConfig, split: str = "\n\n"):
    file_list = [str(file) for file in file_list]

    exemplars = [open(file, "r").read().strip() for file in file_list]
    return split.join(exemplars)


def read_single_file(file_path: str):
    return open(file_path, "r").read().strip()


templates = [
    "[Context]\n{}\n\n[Question]\n{}\n\n[Options]\n{}\n\nHere are the transformed ones in logic form:\n\n",
    "{}\n\nBased on the original description of the problem above and the corresponding logic form. What's the correct answer?\n",
    "{}\n\nThe answer is ",
    "[Context]\n{}\n\n[Question]\n{}\n\n[Options]\n{}\n\nPlease decompose the problem above into smaller ones so that we can solve it separately and reach the final answer by consideing each subproblem and merge the sub-conclusions.\n\n",
    "[Response]\n{}\n\n[Json]\n"
]

_rank2option = ["A", "B", "C", "D", "E"]


def _format_option_list(option_list: List[str]) -> str:
    res = ""
    for op_id, op in enumerate(option_list):
        res += f"{_rank2option[op_id]}. {op}\n"
    return res


class LogicFormPredictionReader:
    def __init__(self, few_shot: bool = True):
        self.few_shot = few_shot

    @staticmethod
    def extract_few_shot_prompt(prompt: str) -> str:
        inputs = prompt.split("[Context]")
        assert len(inputs) >= 2
        return "[Context]" + inputs[-1]

    def __call__(self, file_path: str) -> List[Dict[str, str]]:
        data = json.load(open(file_path, "r"))
        data = [
            {
                "index": item["id"],
                "label": item["label"],
                "text": item["text"] if not self.few_shot else self.extract_few_shot_prompt(item["text"]) + item["response"],
            }
            for item in data
        ]
        return data


class ChatReader:
    def __init__(self):
        pass

    def __call__(self, file_path: str) -> List[Dict[str, str]]:
        data = json.load(open(file_path, "r"))
        data = [
            {
                "index": item["id"],
                "label": item["label"],
                "text": item["text"] + item["response"],
                "response": item["response"],
            }
            for item in data
        ]
        return data


class LogicFormPromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, template_id: int = 0, instruction: str = "", few_shot_prompt: str = "",
                 max_data_num: int = -1, api_based: bool = False):
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            _input = ""
            if self.instruction:
                _input += self.instruction + "\n\n"
            if self.few_shot_prompt:
                _input += self.few_shot_prompt + "\n\n"
            _input += templates[template_id].format(all_context[i], all_question[i], _format_option_list(all_option_list[i]))

            self.inputs.append(_input)
            self.indices.append(i)
            self.labels.append(_rank2option[all_label[i]])

        self.tokenizer = tokenizer
        self.max_data_num = max_data_num
        self.api_based = api_based

    def __len__(self):
        if self.max_data_num:
            return min(self.max_data_num, len(self.inputs))
        return len(self.inputs)

    def api_getitem(self, index):
        return {
            "text": self.inputs[index],
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": self.inputs[index],
            }
        }

    def __getitem__(self, index):
        if self.api_based:
            return self.api_getitem(index)
        return {
            "input": self.inputs[index],
            "index": self.indices[index],
            "prompt_index": "",
            "label": self.labels[index],
        }


class ComposePromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, template_id: int = 0,
                 instruction: str = "", few_shot_prompt: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 max_data_num: int = -1,
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None):
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.compose_keys = compose_keys
        self.input_data: List[Dict[str, Any]] = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(self.input_data)):
            _input = ""
            if self.instruction:
                _input += self.instruction + "\n\n"
            if self.few_shot_prompt:
                _input += self.few_shot_prompt + "\n\n"

            params = [self.input_data[i][key] for key in self.compose_keys]
            _input += templates[template_id].format(*params)

            self.inputs.append(_input)
            self.indices.append(i)
            self.labels.append(self.input_data[i]["label"])

        self.tokenizer = tokenizer
        self.max_data_num = max_data_num
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.inputs))
        return len(self.inputs)

    def api_getitem(self, index):
        return {
            "text": self.inputs[index],
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": self.inputs[index],
            }
        }

    def service_getitem(self, index):
        prompt = self.inputs[index]
        response = self.service_processor(prompt)
        return {
            "input": prompt,
            "response": response,
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": self.inputs[index],
                "response": response,
            }
        }

    def __getitem__(self, index):
        if self.api_based:
            return self.api_getitem(index)
        if self.service_based:
            return self.service_getitem(index)
        return {
            "input": self.inputs[index],
            "index": self.indices[index],
            "prompt_index": "",
            "label": self.labels[index],
        }

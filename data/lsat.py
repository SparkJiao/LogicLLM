import json
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer

from data.collators.flan import convert_to_standard_inputs
from data.cot_critic import vanilla_seq2seq_convertor as vanilla_seq2seq_converter_text
from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer, is_seq2seq_tokenizer

logger = get_child_logger(__name__)

_default_instruct = "Answer the following question with the given context:"

_rank2option = ["A", "B", "C", "D", "E"]

instruction_list = {
    "default": _default_instruct,
    "vicuna_style": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "\n\n### Instruction: ",
}


def _format_option_list(option_list: List[str]) -> str:
    res = ""
    for op_id, op in enumerate(option_list):
        res += f"{_rank2option[op_id]}: {op}\n"
    return res


def load_prompt_from_file(file):
    with open(file) as f:
        tmp = f.read().strip()
    return tmp


lsat_qa_templates = [
    "{}\n\nQuestion: {}\n\nOptions:\n{}",
    "Passage:\n{}\n\nQuestion: {}\n\nOptions:\n{}"
]


class LSATPromptDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, template_id: int = 0, prompt: str = "", suffix: str = ""):
        self.prompt = prompt
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            _input = lsat_qa_templates[template_id].format(all_context[i], all_question[i], _format_option_list(all_option_list[i]))
            _input = prompt + "\n\n" + _input
            if suffix:
                _input += "\n\n" + suffix
            self.inputs.append(_input)
            self.indices.append(i)
            self.labels.append(_rank2option[all_label[i]])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input": self.inputs[index],
            "index": self.indices[index],
            "prompt_index": "",
            "label": self.labels[index],
        }

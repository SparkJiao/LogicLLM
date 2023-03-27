import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer
from typing import Callable
import hydra
from omegaconf import DictConfig
from typing import Union, Tuple

logger = get_child_logger("FLAN")


class FLANDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        self.data = torch.load(file_path, map_location="cpu")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WikiPathDatasetV5WFlan(Dataset):
    def __init__(self, raw_data: Union[Tuple, DictConfig], flan_file: str, file_path: str, tokenizer: PreTrainedTokenizer):
        # print(type(raw_data))
        if isinstance(raw_data, DictConfig):
            raw_data = hydra.utils.instantiate(raw_data, file_path=file_path, tokenizer=tokenizer)

        self.examples = raw_data[0]
        self.flan_data = torch.load(flan_file, map_location="cpu")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if index >= len(self.flan_data):
            flan = random.choice(self.flan_data)
        else:
            flan = self.flan_data[index]
        return {
            "example": example,
            "flan": flan,
            "index": index,
        }


def vanilla_seq2seq_convertor(examples, tokenizer, max_seq_length, decoder_only: bool = False):
    inputs = []
    outputs = []
    for exp in examples:
        inputs.append(exp["inputs"])
        if decoder_only:
            outputs.append(exp["inputs"] + " " + exp["targets"])
        else:
            outputs.append(exp["targets"])

    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding="longest",
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding="longest",
                                 truncation=True, return_tensors="pt")
        model_inputs["input_lens"] = input_lens

    return model_inputs


class FlanCollatorOverCollator:
    def __init__(self, collator, tokenizer: str, max_seq_length: int):
        self.collator = collator
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        flan_batch = []
        for item in batch:
            flan_batch.append(item.pop("flan"))

        if self.collator is not None:
            model_inputs = self.collator(batch)
            flan_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length)
            # if len(model_inputs["input_ids"].size()) == 2:
            #     for k, v in flan_inputs.items():
            #         tmp = torch.zeros(model_inputs[k].size(0) + v.size(0), max(model_inputs["input_ids"].size(1), v.size(1)),
            #                           dtype=v.dtype)
            #         if k != "attention_mask":
            #             tmp = tmp.fill_(self.tokenizer.pad_token_id)
            #         tmp[:model_inputs[k].size(0), :model_inputs[k].size(1)] = model_inputs[k]
            #         tmp[model_inputs[k].size(0):, :v.size(1)] = v
            #         model_inputs[k] = tmp
            # else:
            for k, v in flan_inputs.items():
                model_inputs[f"flan_{k}"] = v
        else:
            model_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length)

        return model_inputs

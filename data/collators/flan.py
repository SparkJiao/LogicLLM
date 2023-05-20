import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer
from typing import Callable
from datasets import load_dataset
import hydra
from omegaconf import DictConfig
from typing import Union, Tuple

logger = get_child_logger("FLAN")


def load_flan_data_w_filter(file_path: str):
    logger.info(f"Loading FLAN data from {file_path}...")
    data = torch.load(file_path, map_location="cpu")
    new_data = []
    cnt = 0
    for item in data:
        if item["inputs"].strip() == "":
            continue
        if item["targets"].strip() == "":
            cnt += 1
            continue
        new_data.append(item)
    logger.info(f"Removed {cnt} empty examples.")
    logger.info(f"Loaded {len(new_data)} examples.")
    return new_data


# def load_gpt4all_data():
#     return load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision='v1.2-jazzy')["train"]


class PromptDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, cfg: DictConfig):
        self.data = hydra.utils.instantiate(cfg, file_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "flan": {
                "inputs": self.data[idx]["prompt"],
                "targets": self.data[idx]["response"],
            }
        }


class FLANDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        self.data = load_flan_data_w_filter(file_path)
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
        self.flan_data = load_flan_data_w_filter(flan_file)

    def __len__(self):
        return max(len(self.examples), len(self.flan_data))

    def __getitem__(self, index):
        example = self.examples[index % len(self.examples)]
        flan = self.flan_data[index % len(self.flan_data)]
        # example = self.examples[index]
        # if index >= len(self.flan_data):
        # flan = random.choice(self.flan_data)
        # else:
        #     flan = self.flan_data[index]
        return {
            "example": example,
            "flan": flan,
            "index": index,
        }


class WikiPathDatasetV5WithDataset(Dataset):
    def __init__(self, raw_data: Union[Tuple, DictConfig], extra_data: Union[PromptDataset, DictConfig],
                 file_path: str, tokenizer: PreTrainedTokenizer, add_wiki_text: bool = False):
        if isinstance(raw_data, DictConfig):
            raw_data = hydra.utils.instantiate(raw_data, file_path=file_path, tokenizer=tokenizer)

        if isinstance(extra_data, DictConfig):
            extra_data = hydra.utils.instantiate(extra_data, tokenizer=tokenizer)

        self.examples = raw_data[0]
        self.extra_data = extra_data

        self.add_wiki_text = add_wiki_text
        if self.add_wiki_text:
            self.wiki_texts = raw_data[1]

    def __len__(self):
        return max(len(self.examples), len(self.extra_data))

    def __getitem__(self, index):
        example = self.examples[index % len(self.examples)]
        flan = self.extra_data[index % len(self.extra_data)]
        res = {
            "example": example,
            "index": index,
        }
        res.update(flan)
        if self.add_wiki_text:
            res["text"] = self.wiki_texts[index % len(self.wiki_texts)]
        return res


class FlanCollectionGroupDataset(Dataset):
    def __init__(self, file_path: str, tokenizer=None):
        super().__init__()
        logger.info(f"Loading FLAN data from {file_path}...")
        data = torch.load(file_path, map_location="cpu")
        self.data = []
        cnt = 0
        for item in data:
            if item["inputs"].strip() == "":
                continue
            if item["targets"].strip() == "":
                cnt += 1
                continue
            self.data.append(item)
        logger.info(f"Removed {cnt} empty examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


def vanilla_seq2seq_convertor(examples, tokenizer: PreTrainedTokenizer, max_seq_length, decoder_only: bool = False):
    inputs = []
    outputs = []
    for exp in examples:
        inputs.append(exp["inputs"])
        if decoder_only:
            outputs.append(exp["inputs"] + " " + exp["targets"] + tokenizer.eos_token)
        else:
            outputs.append(exp["targets"])

    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding="longest",
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding="longest",
                                 truncation=True, return_tensors="pt")
        new_input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        input_lens = input_lens - input_lens.eq(new_input_lens).to(input_lens.dtype) * (input_lens // 2)
        input_lens = input_lens.to(torch.long)
        model_inputs["input_lens"] = input_lens

    return model_inputs


def combine_tensor_on_length(a: torch.Tensor, b: torch.Tensor, pad_id: int):
    max_len = max(a.size(1), b.size(1))
    new_tensor = torch.zeros(a.size(0) + b.size(0), max_len, dtype=a.dtype, device=a.device).fill_(pad_id)
    new_tensor[:a.size(0), :a.size(1)] = a
    new_tensor[a.size(0):, :b.size(1)] = b
    return new_tensor


class FlanCollatorOverCollator:
    def __init__(self, collator, tokenizer: str, max_seq_length: int, decoder_only: bool = False):
        self.collator = collator
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only

    def __call__(self, batch):
        flan_batch = []
        for item in batch:
            flan_batch.append(item.pop("flan"))

        if self.collator is not None:
            model_inputs = self.collator(batch)
            flan_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only)
            for k, v in flan_inputs.items():
                if f"flan_{k}" in model_inputs:
                    model_inputs[f"flan_{k}"] = combine_tensor_on_length(model_inputs[f"flan_{k}"], v, self.tokenizer.pad_token_id)
                else:
                    model_inputs[f"flan_{k}"] = v
        else:
            model_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only)

        return model_inputs

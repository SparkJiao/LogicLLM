import json
from tqdm import tqdm
from data.readers import ReClorReader
import random
from typing import List
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer

_default_instruct = "Answer the following question with the given context:"

_rank2option = ["A", "B", "C", "D", "E"]


def _format_option_list(option_list: List[str]) -> str:
    res = ""
    for op_id, op in enumerate(option_list):
        res += f"{_rank2option[op_id]}: {op}\n"
    return res


class ReClorExemplarGenerator:
    def __init__(self, file_path, read_func, shot: int = 0, random_sampling: bool = False, instruct: str = _default_instruct):
        self.shot = shot
        self.random = random_sampling
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        index = list(range(len(all_context)))

        if self.random:
            random.shuffle(index)

        prompts = []
        for i in index[:self.shot]:
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\nThe answer is {_rank2option[all_label[i]]}"
            prompts.append(prompt)

        self.prompt = instruct + "\n\n" + "\n\n".join(prompts)
        self.indices = index[:self.shot]

    def __call__(self):
        return self.prompt, self.indices


class ReClorGenerativeDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is"):
        self.prompt_generator = prompt_generator
        # read_func = ReClorReader()
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix
            self.inputs.append(prompt)
            self.indices.append(i)
            self.labels.append(_rank2option[all_label[i]])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorGenerativeCollator:
    def __init__(self, tokenizer: str, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        batch = default_collate(batch)
        inputs = batch.pop("input")
        model_inputs = self.tokenizer(inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")

        model_inputs["meta_data"] = batch
        model_inputs["meta_data"]["input"] = inputs
        return model_inputs

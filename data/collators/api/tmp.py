import json
import random

import torch
from torch.utils.data import Dataset

from data.collators.api.wiki_utils import extract_ending_entity, extract_ending_entity_and_anonymization, extract_ending_entity_and_replace
from data.collators.wiki_seq2seq_collator import construct_seq2seq
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class XuangDescriptionDataset(Dataset):
    prompt_templates = [
        "Given a description, replace one or two the most significant words or phrases describing the emotions or idendity of human face with any other words or phrases sharing the same category:\n\n"
        "a man in a suit and tie poses outside, in the style of light silver and green, [philip treacy]( [tom chambers](\n\n"
        "For example, you can replace \"man\" with \"woman\" and \"white\" with \"\black\".\n\n"
        "Now, here is a new description:\n\n"
        "{}\n\n"
        "Please give the description after replacement:",
    ]

    def __init__(self, file_path: str, tokenizer=None):
        data = json.load(open(file_path))
        self.data = []
        for k, v in data.items():
            for desc_id, desc in enumerate(v["des"]):
                self.data.append((k, desc_id, desc))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        k, desc_id, desc = self.data[index]
        text = self.prompt_templates[0].format(desc)
        return {
            "text": text,
            "meta_data": {
                "text": text,
                "index": f"{k}-{desc_id}"
            }
        }

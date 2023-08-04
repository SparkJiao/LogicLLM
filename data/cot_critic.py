import json
import random

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.collators.wiki_seq2seq_collator import construct_seq2seq


def merit_pair_data_transform(raw_data, flat_mode: bool = False):
    """
    Inputs come from `data.wiki_entity_path_v9_1_2.convert_examples_into_features_raw` method.
    """
    examples, raw_texts = raw_data
    data = []
    for exp in tqdm(examples, total=len(examples), desc="Transforming merit examples..."):
        input_a, input_b = construct_seq2seq(exp, generative_mode=False)
        inputs = [a + " " + b for a, b in zip(input_a, input_b)]
        if not flat_mode:
            data.append({
                "pos_input": inputs[0],
                "neg_input": random.choice(inputs[1:]),
            })
        else:
            data.extend({
                            "pos_input": inputs[0],
                            "neg_input": neg,
                        } for neg in inputs[1:])

    return data


class CoTPairData(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, flat_mode: bool = True,
                 merit_data=None, merit_data_sample_num: int = 0):
        super().__init__()
        data = json.load(open(file_path))

        self.tokenizer = tokenizer
        self.flat_mode = flat_mode
        if self.flat_mode:
            self.data = []
            for item in data:
                self.data.extend([{
                    "pos_input": item["pos"],
                    "neg_input": neg,
                } for neg in item["neg"]])
        else:
            self.data = data

        if merit_data is not None:
            if isinstance(merit_data, tuple):
                merit_data = merit_data[0]
            self.merit_data = merit_data
            self.merit_data_sample_num = merit_data_sample_num
        else:
            self.merit_data = None
            self.merit_data_sample_num = 0.0

    def __len__(self):
        if self.merit_data is not None:
            return len(self.data) + self.merit_data_sample_num
        return len(self.data)

    def __getitem__(self, index):
        if index < len(self.data):
            return {
                "pos_input": self.data[index]["pos_input"],
                "neg_input": self.data[index]["neg_input"],
                "index": index,
                "label": -1,
            }
        else:
            item = random.choice(self.merit_data)
            if "pos_input" not in item:
                input_a, input_b = construct_seq2seq(item, generative_mode=False)
                inputs = [a + " " + b for a, b in zip(input_a, input_b)]
                item = {
                    "pos_input": inputs[0],
                    "neg_input": random.choice(inputs[1:]),
                }
            return {
                "pos_input": item["pos_input"],
                "neg_input": item["neg_input"],
                "index": index,
                "label": -1,
            }

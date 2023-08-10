import json
import random

from typing import Callable, List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
import re

# Add the directory of this file into pythonpath
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collators.wiki_seq2seq_collator import construct_seq2seq
from data.reclor_prompt import _format_option_list
from general_util.tokenization_utils import expand_special_tokenizer


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


def vanilla_seq2seq_convertor(inputs, outputs, tokenizer: PreTrainedTokenizer, max_seq_length, padding="longest",
                              decoder_only: bool = False):
    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding=padding,
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        new_input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        input_lens = input_lens - input_lens.eq(new_input_lens).to(input_lens.dtype) * (input_lens // 2)
        input_lens = input_lens.to(torch.long)
        if tokenizer.padding_side == "left":
            input_lens = model_inputs["input_ids"].eq(tokenizer.pad_token_id).to(torch.long).sum(dim=1) + input_lens
        model_inputs["input_lens"] = input_lens

    return model_inputs


class CoTNonPairCriticData(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, negative_data: str, cot_reader: Callable = None, ):
        super().__init__()

        if cot_reader is None:
            self.pos_data = json.load(open(file_path))
            self.neg_data = json.load(open(negative_data))
            self.original_format = True
        else:
            self.pos_data = cot_reader(file_path)
            self.neg_data = cot_reader(negative_data)
            self.original_format = False

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pos_data)

    def __getitem__(self, index):
        if self.original_format:
            neg_item = random.choice(self.neg_data)
            return {
                "pos_input": self.pos_data[index]["output"] + self.tokenizer.eos_token,
                "neg_input": neg_item["output"] + self.tokenizer.eos_token,
                "index": f"pos{self.pos_data[index]['index']}-neg{neg_item['index']}",
                "label": -1,
            }
        else:
            pass


def multiple_choice_ans_parser(ans: str):
    regrex = "A|B|C|D"
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    x = re.findall(regrex, ans)
    x = mapping[x[-1]]

    return x


def get_ans_parser(task_name: str):
    if task_name == "multiple_choice":
        return multiple_choice_ans_parser
    else:
        raise ValueError(f"task_name: {task_name} is not supported.")


class CoTActorRankingDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, original_data, read_func: Callable,
                 ans_parser: Callable = None, margin: float = 0.0,
                 pair_pattern: Tuple[str] = ("pn", "pp"),
                 prefix1: str = "", prefix2: str = "\n\n", prefix3: str = "\n\n", suffix: str = "\n\n"
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.margin = margin
        self.pair_pattern = pair_pattern

        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.prefix3 = prefix3
        self.suffix = suffix

        self.context, self.question, self.option_list, self.all_labels = read_func(original_data)

        # Read the critic model rewards predictions directly.
        data: List[Dict[str, Any]] = json.load(open(file_path))

        # annotating multiple choice predicted labels
        # if ans_parser is not None:
        ans_pos = []
        ans_neg = []

        for item in data:
            if item["cleaned_output"] == item["label"]:
                ans_pos.append(item)
            else:
                ans_neg.append(item)

        pairs = []
        assert len(pair_pattern)
        for pat in pair_pattern:
            if pat[0] == "p":
                left = ans_pos
            elif pat[0] == "n":
                left = ans_neg
            else:
                raise ValueError(f"Pattern {pat} is not supported.")

            if pat[1] == "p":
                right = ans_pos
            elif pat[1] == "n":
                right = ans_neg
            else:
                raise ValueError(f"Pattern {pat} is not supported.")

            for i, x in enumerate(left):
                for j, y in enumerate(right):
                    if i == j and pat[0] == pat[1]:
                        continue
                    if x["reward"] - y["reward"] > self.margin:
                        pairs.append((x, y))
        # else:
        #     raise NotImplementedError("Not implemented yet.")

        self.data = pairs

    def __len__(self):
        return len(self.data)

    def compose_example(self, item: Dict[str, Any]):
        orig_index = item["index"]
        inputs = self.prefix1 + self.context[orig_index] + self.prefix2 + self.question[orig_index] + self.prefix3 + \
                 _format_option_list(self.option_list[orig_index]) + self.suffix
        outputs = item["output"] + self.tokenizer.eos_token
        return inputs, outputs

    def __getitem__(self, index):
        pair = self.data[index]

        x_input, x_output = self.compose_example(pair[0])
        y_input, y_output = self.compose_example(pair[1])

        return {
            "pos_input": x_input,
            "pos_output": x_output,
            "neg_input": y_input,
            "neg_output": y_output,
            "index": f"pos{pair[0]['index']}-{pair[1]['index']}",
        }


class CoTActorRankingCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, padding: str = "longest", pp_inputs_processor: Callable = None, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)

        self.max_seq_length = max_seq_length
        self.padding = padding
        self.pp_inputs_processor = pp_inputs_processor

    def __call__(self, batch):
        inputs = [b["pos_input"] for b in batch] + [b["neg_input"] for b in batch]
        outputs = [b["pos_output"] for b in batch] + [b["neg_output"] for b in batch]
        index = [b["index"] for b in batch]

        model_inputs = vanilla_seq2seq_convertor(inputs, outputs, self.tokenizer, self.max_seq_length, self.padding, decoder_only=True)

        if self.pp_inputs_processor is not None:
            return self.pp_inputs_processor(model_inputs, self.tokenizer)

        model_inputs["meta_data"] = {
            "inputs": inputs,
            "outputs": outputs,
            "index": index,
        }
        return model_inputs


if __name__ == '__main__':

    from data.readers import LogiQAReaderV2
    from data.mp_inputs_process import LlamaDoubleHeadPpInputsProcess

    tokenizer = AutoTokenizer.from_pretrained("/export/home2/fangkai/pretrained-models/Llama-2-70b-hf")

    dataset = CoTActorRankingDataset("experiments/llama2.7b.rw.lqv2cot.w4.A100.v1.0/lqv2cot_dev_zs_cot_2k_llama2_chat_70b_rewards.v1.0/test-checkpoint-1000/cot_w_feedback/cot_feedback.json",
                                     tokenizer=tokenizer,
                                     original_data="logiqa-v2/dev.txt",
                                     read_func=LogiQAReaderV2(),
                                     margin=8.0)

    inputs_process = LlamaDoubleHeadPpInputsProcess()

    collator = CoTActorRankingCollator(tokenizer="/export/home2/fangkai/pretrained-models/Llama-2-70b-hf", max_seq_length=1024,
                                       padding="longest",
                                       pp_inputs_processor=inputs_process)

    batch = [dataset[0], dataset[1]]
    print(batch[0])
    print(batch[1])

    res = collator(batch)

    # print(res)
    print(res[0][0])
    print(res[0][1])
    print(res[0][2])
    print(res[0][3])
    print(res[1])

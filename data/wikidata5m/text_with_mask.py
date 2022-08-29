import json

import torch
from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy
from torch import Tensor
import random


# class LogicCircleDataset(Dataset):
#     def __init__(self, logical_circle: str, id2ent: str, id2rel: str, triplet2sent: str):
#         super(self).__init__()
#
#         self.logical_circle = json.load(open(logical_circle, 'r'))
#         self.id2ent = json.load(open(id2ent, 'r'))
#         self.id2rel = json.load(open(id2rel, 'r'))
#         self.triplet2sent = json.load(open(triplet2sent, 'r'))
#
#     def __iter__(self):
#         pass
#
#     def __len__(self):
#         pass


# class WholeWordMLMBatchCollatorMixin:
#     @staticmethod
def _align_sequence_with_special_token(input_ids: Union[Tensor, List[int]], sequence: List[int],
                                       tokenizer: PreTrainedTokenizer, padding: int = 0):
    assert len(input_ids.size()) == 1
    special_token_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    padded_sequence = []
    cnt = 0
    for mask in special_token_mask:
        if mask == 1:
            if cnt < len(sequence):
                padded_sequence.append(sequence[cnt])
                cnt += 1
            else:
                padded_sequence.append(padding)
        else:
            padded_sequence.append(padding)
    assert len(padded_sequence) == len(input_ids)
    return padded_sequence


# @staticmethod
def _pad_sequence_to_max_len(sequence: List[List[int]], padding: int = 0):
    max_len = max(map(len, sequence))
    padded_sequence = [seq + [padding] * (max_len - len(seq)) for seq in sequence]
    return padded_sequence


# @staticmethod
def mask_text(text: List[str], token2word_index: List[List[int]], indicate_mask: List[List[int]],
              entity_mask_ratio: float, entity_mask_num: int,
              non_entity_mask_ratio: float, non_entity_mask_num: int,
              tokenizer: PreTrainedTokenizer):
    model_inputs = tokenizer(text,
                             truncation=True,
                             padding=PaddingStrategy.LONGEST,
                             return_tensors="pt")

    aligned_token2word_index = []
    for input_ids, item_tk2wd_index in zip(model_inputs["input_ids"], token2word_index):
        item_padded_tk2wd_index = _align_sequence_with_special_token(input_ids, item_tk2wd_index, tokenizer)
        aligned_token2word_index.append(item_padded_tk2wd_index)
    token2word_index = torch.tensor(aligned_token2word_index, dtype=torch.long)
    assert token2word_index.size() == model_inputs["input_ids"].size()

    padded_indicate_mask = _pad_sequence_to_max_len(indicate_mask, padding=-1)
    indicate_mask = torch.tensor(padded_indicate_mask, dtype=torch.long)

    # Whole word level mask
    ww_mlm_mask = generate_mlm_mask_strategy1(indicate_mask,
                                              entity_mask_ratio,
                                              entity_mask_num,
                                              non_entity_mask_ratio,
                                              non_entity_mask_num)

    mlm_mask = torch.zeros(token2word_index.size(), dtype=torch.bool)
    mlm_mask = torch.gather(ww_mlm_mask, dim=1, index=token2word_index)

    labels = model_inputs["input_ids"].clone()
    labels[~mlm_mask] = -100

    model_inputs[mlm_mask] = tokenizer.mask_token_id

    return model_inputs, token2word_index, indicate_mask


# @staticmethod
def generate_mlm_mask_strategy1(indicate_mask: Tensor,
                                entity_mask_ratio: float = 0.4,
                                entity_mask_num: int = 2,
                                non_entity_mask_ratio: float = 0.4,
                                non_entity_mask_num: int = 2):
    """
    Args:
        indicate_mask (`Tensor`):
            [batch_size, max_word_num]
        entity_mask_ratio
        entity_mask_num
        non_entity_mask_ratio
        non_entity_mask_num
    """
    mask = torch.zeros(indicate_mask.size(), dtype=torch.bool)

    if entity_mask_num:
        entity_num = (indicate_mask == 1).sum(dim=-1)
        entity_masked_indices = torch.zeros(indicate_mask.size(), dtype=torch.bool)
        for item_id, item_value_num in enumerate(entity_num):
            item_mask = generate_mask_via_num(item_value_num, entity_mask_num)
            entity_masked_indices[indicate_mask[item_id] == 1] = item_mask
        mask = mask | entity_masked_indices
    elif entity_mask_ratio:
        entity_masked_indices = generate_mask_via_ratio(indicate_mask, entity_mask_ratio) & (indicate_mask == 1)
        mask = mask | entity_masked_indices

    if non_entity_mask_num:
        non_entity_num = (indicate_mask == 0).sum(dim=-1)
        non_entity_masked_indices = torch.zeros(indicate_mask.size(), dtype=torch.bool)
        for item_id, item_value_num in enumerate(non_entity_num):
            item_mask = generate_mask_via_num(item_value_num, non_entity_mask_num)
            non_entity_masked_indices[indicate_mask[item_id] == 0] = item_mask
        mask = mask | non_entity_masked_indices
    elif non_entity_mask_ratio:
        non_entity_masked_indices = generate_mask_via_ratio(indicate_mask, non_entity_mask_ratio) & (indicate_mask == 0)
        mask = mask | non_entity_masked_indices

    return mask


def generate_mask_via_num(total_num: int, mask_num: int):
    mask = [0] * (total_num - mask_num) + [1] * mask_num
    random.shuffle(mask)
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask


# @staticmethod
def generate_mask_via_ratio(tensor: Tensor, ratio: float):
    probability_matrix = torch.full(tensor.shape, ratio)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    return masked_indices


# @staticmethod
def generate_mlm_mask():
    pass


# @staticmethod
def word_mask2token_mask():
    pass

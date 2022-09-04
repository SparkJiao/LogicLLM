import json
import os.path

import torch
from torch.utils.data import Dataset
from typing import List, Union, Tuple, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy
from torch import Tensor
import random
from torch.utils.data.dataloader import default_collate
from data.data_utils import tokenizer_get_name
import glob
from general_util.logger import get_child_logger
from data.collators.dict2dict import DictTensorDataset

logger = get_child_logger(__name__)


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


def _pad_sequence_to_max_len(sequence: List[List[int]], padding: int = 0):
    max_len = max(map(len, sequence))
    padded_sequence = [seq + [padding] * (max_len - len(seq)) for seq in sequence]
    return padded_sequence


def _roberta_whole_word_mask(input_tokens: List[str], max_predictions: int = 512, mlm_probability: float = 0.15):
    cand_indices = []
    for i, token in enumerate(input_tokens):
        if token == "<s>" or token == "</s>":
            continue

        if len(cand_indices) >= 1 and token.startswith("Ġ"):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    random.shuffle(cand_indices)
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)

    if len(covered_indexes) != len(masked_lms):
        raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
    mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
    return mask_labels


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
        item_padded_tk2wd_index = _align_sequence_with_special_token(input_ids, item_tk2wd_index, tokenizer, padding=-1)
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

    padding_mask = token2word_index == -1
    token2word_index[padding_mask] = 0
    mlm_mask = torch.gather(ww_mlm_mask, dim=1, index=token2word_index)
    mlm_mask[padding_mask] = 0

    labels = model_inputs["input_ids"].clone()
    labels[~mlm_mask] = -100

    model_inputs[mlm_mask] = tokenizer.mask_token_id

    return model_inputs, token2word_index, indicate_mask


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


def generate_mask_via_ratio(tensor: Tensor, ratio: float):
    probability_matrix = torch.full(tensor.shape, ratio)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    return masked_indices


def seq2seq_batch_entity_indicate(text: List[str], token2word_index: List[List[int]], indicate_mask: List[List[int]],
                                  tokenizer: PreTrainedTokenizer):
    """
    All of this method to do is generate a mask indicate if each token (subword) belongs to an entity.
    """
    model_inputs = tokenizer(text,
                             truncation=True,
                             padding=PaddingStrategy.LONGEST,
                             return_tensors="pt")

    aligned_token2word_index = []
    for input_ids, item_tk2wd_index in zip(model_inputs["input_ids"], token2word_index):
        item_padded_tk2wd_index = _align_sequence_with_special_token(input_ids, item_tk2wd_index, tokenizer, padding=-1)
        aligned_token2word_index.append(item_padded_tk2wd_index)
    token2word_index = torch.tensor(aligned_token2word_index, dtype=torch.long)
    assert token2word_index.size() == model_inputs["input_ids"].size()

    padded_indicate_mask = _pad_sequence_to_max_len(indicate_mask, padding=-1)
    indicate_mask = torch.tensor(padded_indicate_mask, dtype=torch.long)

    token_padding_mask = token2word_index == -1
    token2word_index[token_padding_mask] = 0
    extended_indicate_mask = torch.gather(indicate_mask, dim=1, index=token2word_index)
    extended_indicate_mask[token_padding_mask] = -1

    return model_inputs["input_ids"], extended_indicate_mask


def load_seq2seq_data(file_path: str, tokenizer: PreTrainedTokenizer, max_input_length: int, max_output_length: int,
                      glob_mark: str = None, sample: bool = False):
    """
    Corresponding to the data generated from `preprocess.wikidata_5m.logical_circle_to_text`.
    """
    tokenizer_name = tokenizer_get_name(tokenizer)
    file_suffix = f"{tokenizer_name}_{max_input_length}_{max_output_length}_sss_v1_0"

    if os.path.exists(file_path):
        train_files = [file_path]
        cached_file_path = f"{file_path}_{file_suffix}"
    else:
        train_files = sorted(list(glob.glob(file_path)))
        cached_file_path = f"{train_files[-1]}_{glob_mark}_{file_suffix}"

    logger.info(train_files)

    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        data = torch.load(cached_file_path)
        if sample:
            data = {k: v[:10000] for k, v in data.items()}
        return DictTensorDataset(data)

    src_texts = []
    tgt_texts = []
    for file in train_files:
        src, tgt = list(zip(*json.load(open(file, 'r'))))
        src_texts.extend(src)
        tgt_texts.extend(tgt)

    model_inputs = tokenizer(src_texts,
                             padding=PaddingStrategy.LONGEST,
                             truncation=True,
                             max_length=max_input_length,
                             return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, padding=PaddingStrategy.LONGEST, truncation=True, max_length=max_output_length,
                           return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    logger.info(f"Input length: {model_inputs['input_ids'].size()}")
    logger.info(f"Output length: {model_inputs['labels'].size()}")
    logger.info(f"Saving to {cached_file_path}")
    torch.save(model_inputs, cached_file_path)

    return DictTensorDataset(model_inputs)


class Seq2SeqTextDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer):
        super().__init__()

        if os.path.exists(file_path):
            train_files = [file_path]
        else:
            train_files = sorted(list(glob.glob(file_path)))

        src_texts = []
        tgt_texts = []
        for file in train_files:
            src, tgt = list(zip(*json.load(open(file, 'r'))))
            src_texts.extend(src)
            tgt_texts.extend(tgt)

        self.src = src_texts
        self.tgt = tgt_texts

    def __getitem__(self, index) -> Dict[str, str]:
        return {
            "src": self.src[index],
            "tgt": self.tgt[index],
        }

    def __len__(self):
        return len(self.src)


class Seq2SeqTextCollator:
    def __init__(self, tokenizer, max_input_length: int, max_output_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __call__(self, batch):
        batch = default_collate(batch)
        src = batch.pop("src")
        tgt = batch.pop("tgt")

        model_inputs = self.tokenizer(src,
                                      padding=PaddingStrategy.LONGEST,
                                      truncation=True,
                                      max_length=self.max_input_length,
                                      return_tensors="pt")

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt,
                                    padding=PaddingStrategy.LONGEST,
                                    truncation=True,
                                    max_length=self.max_output_length,
                                    return_tensors="pt")

        model_inputs["labels"] = labels["input_ids"]

        meta_data = [
            {
                "src": _src,
                "tgt": _tgt,
            } for _src, _tgt in zip(src, tgt)
        ]
        model_inputs["meta_data"] = meta_data

        return model_inputs

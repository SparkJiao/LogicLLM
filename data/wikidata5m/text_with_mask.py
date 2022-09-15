import glob
import json
import os.path
import random
from typing import List, Union, Tuple, Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy

from data.collators.dict2dict import DictTensorDataset
from data.data_utils import tokenizer_get_name
from general_util.logger import get_child_logger
from modules.trie import MarisaTrie

logger = get_child_logger(__name__)


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Copied from transformers.data.data_collator import _torch_collate_batch."""
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result


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


def _roberta_whole_word_mask_v1(input_tokens: List[str], max_predictions: int = 512, mlm_probability: float = 0.15,
                                spans: List[Tuple[int, int]] = None, span_mlm_probability: float = -1, span_min_predictions: int = 2,
                                pad_token: str = "<pad>"):
    span_indicate = {span[0]: span[1] for span in spans} if spans is not None else {}

    cand_indices = []
    cand_span_mask = {}  # `cand_span_mask[i] == 1` indicates the index set starting with index-i is a span to be masked.
    true_token_num = 0
    for i, token in enumerate(input_tokens):
        if token == "<s>" or token == "</s>":
            continue

        if token == pad_token or token == "<pad>":
            break

        true_token_num += 1

        if i in span_indicate:
            cand_indices.append(list(range(i, span_indicate[i])))
            cand_span_mask[i] = 1
            continue

        if len(cand_indices) >= 1 and i <= cand_indices[-1][-1]:  # For entity span
            continue

        if len(cand_indices) == 0 or token.startswith("Ġ"):
            cand_indices.append([i])
            cand_span_mask[i] = 0
        else:
            cand_indices[-1].append(i)

    if spans is not None:
        span_indices = [i for i, _index_set in enumerate(cand_indices) if cand_span_mask[_index_set[0]]]
        span_num_to_predict = max(span_min_predictions, int(round(len(span_indices) * span_mlm_probability)))
        random.shuffle(span_indices)
        span_indices = set(span_indices[:span_num_to_predict])

        re_organized_cand_indices_prefix = []
        re_organized_cand_indices_suffix = []
        for i, _index_set in enumerate(cand_indices):
            if i in span_indices:
                re_organized_cand_indices_prefix.append(_index_set)
            else:
                re_organized_cand_indices_suffix.append(_index_set)
        random.shuffle(re_organized_cand_indices_suffix)
        cand_indices = re_organized_cand_indices_prefix + re_organized_cand_indices_suffix
    else:
        span_num_to_predict = -1
        random.shuffle(cand_indices)

    num_to_predict = min(max_predictions, max(1, int(round(true_token_num * mlm_probability))))
    masked_lms = []
    covered_indexes = set()
    for i, index_set in enumerate(cand_indices):
        # `span_num_to_predict` ensure that at least all the selected entities are to be masked.
        if i > span_num_to_predict and len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if i > span_num_to_predict and len(masked_lms) + len(index_set) > num_to_predict:
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


class MLMTextDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer):
        super().__init__()

        texts, spans = list(zip(*json.load(open(file_path, 'r'))))
        self.texts = texts
        self.spans = spans

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "span": self.spans[index],
        }

    def __len__(self):
        return len(self.texts)


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


class Seq2SeqEntityTextDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer):
        super().__init__()

        if os.path.exists(file_path):
            train_files = [file_path]
        else:
            train_files = sorted(list(glob.glob(file_path)))

        src_texts = []
        tgt_texts = []
        entities = []
        for file in train_files:
            src, tgt, entity = list(zip(*json.load(open(file, 'r'))))
            src_texts.extend(src)
            tgt_texts.extend(tgt)
            entities.extend(entity)

        self.src = src_texts
        self.tgt = tgt_texts
        self.entity = entities

    def __getitem__(self, index) -> Dict[str, str]:
        return {
            "src": self.src[index],
            "tgt": self.tgt[index],
            "entity": self.entity[index],
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


class Seq2SeqEntityTextCollator(Seq2SeqTextCollator):
    def __init__(self, tokenizer, max_input_length: int, max_output_length: int):
        super().__init__(tokenizer, max_input_length, max_output_length)

        self.tokenizer.add_tokens("<s>")

    def __call__(self, batch):
        entities = [item.pop("entity") for item in batch]

        batch = default_collate(batch)
        src = batch.pop("src")
        tgt = batch.pop("tgt")
        assert len(batch) == 0, list(batch.keys())

        concat_src = []
        for src_text, tgt_ent_ls in zip(src, entities):
            concat_src.append("<s>".join([src_text] + tgt_ent_ls))
        src = concat_src

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


class MLMTextCollator:
    def __init__(self, tokenizer,
                 max_seq_length: int,
                 mlm_probability: float = 0.15,
                 span_mlm_probability: float = 0.4,
                 span_min_predictions: int = 2):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.span_mlm_probability = span_mlm_probability
        self.span_min_predictions = span_min_predictions
        self.tokenizer_name = tokenizer_get_name(self.tokenizer)

        # If yes, check if we have a `pad_token`.
        if self.tokenizer.pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

    def __call__(self, batch):
        texts = [b.pop("text") for b in batch]
        spans = [b.pop("span") for b in batch]
        del batch

        raw_token_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)) for text in texts]

        model_inputs = self.tokenizer(texts,
                                      padding=PaddingStrategy.MAX_LENGTH,
                                      truncation=True,
                                      max_length=self.max_seq_length,
                                      return_tensors="pt")

        input_ids = model_inputs["input_ids"].tolist()

        mask_labels = []
        for raw_input_id, input_id, item_spans in zip(raw_token_ids, input_ids, spans):
            mask_labels.append(self._whole_word_mask(raw_input_id, input_id, item_spans))
        mask_labels = torch.tensor(mask_labels, dtype=torch.int)

        input_ids, labels = self.torch_mask_tokens(model_inputs["input_ids"], mask_labels)
        model_inputs["input_ids"] = input_ids
        model_inputs["labels"] = labels
        return model_inputs

    def _whole_word_mask(self, raw_input_ids: List[int], input_ids: List[int], spans: List[Tuple[int, int]]):
        special_token_mask = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

        # Calculate the mapping from the original position to the new position after adding special tokens.
        orig2now = []
        for i, (token_mask, token_id) in enumerate(zip(special_token_mask, input_ids)):
            if token_mask == 1:
                continue
            orig2now.append(i)
            assert raw_input_ids[len(orig2now) - 1] == token_id

        # Map the original spans to new positions.
        mapped_spans = []
        for span in spans:
            if span[0] < len(orig2now) and span[1] < len(orig2now):
                mapped_spans.append((orig2now[span[0]], orig2now[span[1]]))

        mask_labels = _roberta_whole_word_mask_v1(self.tokenizer.convert_ids_to_tokens(input_ids),
                                                  max_predictions=256,
                                                  mlm_probability=self.mlm_probability,
                                                  spans=mapped_spans,
                                                  span_mlm_probability=self.span_mlm_probability,
                                                  span_min_predictions=self.span_min_predictions,
                                                  pad_token=self.tokenizer.pad_token)

        assert len(mask_labels) == len(input_ids)

        return mask_labels

    def torch_mask_tokens(self, input_ids: Tensor, mask_labels: Tensor):
        """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
            'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels


def seq2seq_text_trie(file_path, tokenizer: str, max_seq_length: int = 512):
    data = json.load(open(file_path))
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # trie = Trie()
    # trie = MarisaTrie()

    src, tgt = list(zip(*data))
    all_seq_ls = []
    for tgt_text in tgt:
        tgt_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tgt_text))[:max_seq_length - 2]
        tgt_tokens.append(tokenizer.eos_token_id)
        tgt_tokens = [tokenizer.pad_token_id] + tgt_tokens
        # trie.add(tgt_tokens)
        all_seq_ls.append(tgt_tokens)

    trie = MarisaTrie(all_seq_ls)

    return trie

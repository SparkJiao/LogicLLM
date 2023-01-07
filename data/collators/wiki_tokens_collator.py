import random
from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.tokens.collator")


def mask_tokens(
        tokenizer: PreTrainedTokenizer, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,
        mlm_probability: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # Remove padding.
        special_tokens_mask = special_tokens_mask | (labels == tokenizer.pad_token_id)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class WikiPathTokensDatasetCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15):
        self.max_seq_length = max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        examples = [b["example"] for b in batch]
        texts = [b["text"] for b in batch]

        option_num = len(examples[0]["tokens"])
        max_seq_length = 0
        for exp in examples:
            for op_id, op in enumerate(exp["tokens"]):
                if isinstance(op, str):
                    assert op_id > 0
                    exp["tokens"][op_id] = self.tokenizer.tokenize(op)
                max_seq_length = max(max_seq_length, len(exp["tokens"][op_id]))

        # max_seq_length = max(map(lambda x: max(map(len, x["tokens"])), examples))
        assert max_seq_length <= self.max_seq_length, max_seq_length
        input_ids = torch.zeros(len(examples), option_num, max_seq_length, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(len(examples), option_num, max_seq_length, dtype=torch.long)
        for exp_id, exp in enumerate(examples):
            for op_id, op_tokens in enumerate(exp["tokens"]):
                input_ids[exp_id, op_id, :len(op_tokens)] = torch.tensor(self.tokenizer.convert_tokens_to_ids(op_tokens), dtype=torch.long)
                attention_mask[exp_id, op_id, :len(op_tokens)] = 1

        max_sent_num = 0
        max_sent_len = 0
        for exp in examples:
            max_sent_num = max(max_sent_num, len(exp["h_spans"][0]))  # [batch_size, option_num, sent_num, span_num]
            max_sent_len = max(max_sent_len, max(map(lambda x: x[1] - x[0], exp["sentence_spans"][0])))

        sent_token_index = torch.zeros(len(examples), max_sent_num, max_sent_len, dtype=torch.long)
        sent_token_mask = torch.zeros(len(examples), max_sent_num, max_sent_len)
        tagging_labels = torch.zeros(len(examples), max_seq_length, dtype=torch.long)
        h_span_marks = torch.zeros(len(examples), max_sent_num, max_seq_length)
        t_span_marks = torch.zeros(len(examples), max_sent_num, max_seq_length)
        entity_pair_mask = torch.zeros(len(examples), max_sent_num)
        for exp_id, exp in enumerate(examples):
            for sent_id, sent_h_spans in enumerate(exp["h_spans"][0]):
                token_num = 0

                for span in sent_h_spans:
                    tagging_labels[exp_id, span[0]: span[1]] = 1
                    h_span_marks[exp_id, sent_id, span[0]: span[1]] = 1
                    token_num += span[1] - span[0]

                if token_num:
                    h_span_marks[exp_id, sent_id] = h_span_marks[exp_id, sent_id] * 1.0 / token_num
                else:
                    entity_pair_mask[exp_id, sent_id] = 1

            assert len(exp["h_spans"][0]) == len(batch[exp_id]["rel_labels"]) - 1 or batch[exp_id]["rel_labels"] == [-1], (
                exp["h_spans"][0], batch[exp_id]["rel_labels"])

            for sent_id, sent_t_spans in enumerate(exp["t_spans"][0]):
                token_num = 0
                # assert len(sent_t_spans) >= 1, (sent_id, exp["t_spans"][0])

                for span in sent_t_spans:
                    tagging_labels[exp_id, span[0]: span[1]] = 1
                    t_span_marks[exp_id, sent_id, span[0]: span[1]] = 1
                    token_num += span[1] - span[0]

                if token_num:
                    t_span_marks[exp_id, sent_id] = t_span_marks[exp_id, sent_id] * 1.0 / token_num
                else:
                    entity_pair_mask[exp_id, sent_id] = 1

            for sent_id, sent_span in enumerate(exp["sentence_spans"][0]):
                _sent_len = sent_span[1] - sent_span[0]
                sent_token_index[exp_id, sent_id, :_sent_len] = torch.arange(sent_span[0], sent_span[1], dtype=torch.long)
                sent_token_mask[exp_id, sent_id, :_sent_len] = 1

            assert len(exp["t_spans"][0]) == len(exp["h_spans"][0]) == len(exp["sentence_spans"][0])

        tagging_labels[input_ids[:, 0] == self.tokenizer.pad_token_id] = -1

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = mask_tokens(self.tokenizer, mlm_input_ids, mlm_probability=self.mlm_probability)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.zeros(len(examples), dtype=torch.long),
            "tagging_labels": tagging_labels,
            "h_span_marks": h_span_marks,
            "t_span_marks": t_span_marks,
            "sent_token_index": sent_token_index,
            "sent_token_mask": sent_token_mask,
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "entity_pair_mask": entity_pair_mask.bool(),
        }


def transform_rel_label_to_tensor(rel_labels):
    if isinstance(rel_labels[0][0], list):
        new_rel_labels = []
        for item_rel_labels in rel_labels:
            new_rel_labels.extend(item_rel_labels)
        rel_labels = new_rel_labels

    assert isinstance(rel_labels[0][0], int)

    max_input_len = max(map(len, rel_labels))
    invalid = 0
    decoder_input_ids = torch.zeros(len(rel_labels), max_input_len, dtype=torch.long).fill_(-1)
    for b, b_decoder_inputs in enumerate(rel_labels):
        decoder_input_ids[b, :len(b_decoder_inputs)] = torch.tensor(b_decoder_inputs, dtype=torch.long)
        if b_decoder_inputs[0] == -1:
            invalid += 1

    return decoder_input_ids, invalid


def get_sentence_level_label_v1(example, rel_ids):
    path_s_ids_order = example["path_s_ids_order"]
    path_s_ids = example["path_s_ids"]

    assert len(path_s_ids_order) == len(path_s_ids) == len(rel_ids["input_a"])
    s_id2rel_id = {s_id: rel_id for s_id, rel_id in zip(path_s_ids_order, rel_ids["input_a"])}
    rel_ids_input_order = [s_id2rel_id[s_id] for s_id in path_s_ids] + [rel_ids["input_b"]]
    return rel_ids_input_order


def get_sentence_level_label_v2(example, rel_ids):
    path_s_ids_order = example["path_s_ids_order"]
    path_s_ids = example["path_s_ids"]

    if rel_ids == [-1]:
        return rel_ids * (len(path_s_ids) + 2)

    assert len(path_s_ids_order) == len(path_s_ids) == (len(rel_ids) - 2), (path_s_ids_order, path_s_ids, rel_ids,
                                                                            len(example["h_spans"][0]))
    s_id2rel_id = {s_id: rel_id for s_id, rel_id in zip(path_s_ids_order, rel_ids[:-2])}
    rel_ids_input_order = [s_id2rel_id[s_id] for s_id in path_s_ids] + rel_ids[-2:]
    assert len(rel_ids_input_order) == len(rel_ids)
    return rel_ids_input_order


class WikiPathDatasetCollatorRelSeqGenV1(WikiPathTokensDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15,
                 gen_only: bool = False, option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability)
        self.gen_only = gen_only
        self.option_dropout = option_dropout

    def __call__(self, batch):
        rel_decode = []
        for b in batch:
            rel_decode.append(get_sentence_level_label_v2(b["example"], b["rel_labels"]))

        decoder_input_ids, invalid = transform_rel_label_to_tensor(rel_decode)

        res = super().__call__(batch)
        res["rel_labels"] = decoder_input_ids
        res["invalid_path"] = invalid

        if self.gen_only:
            res["input_ids"] = res["input_ids"][:, :1]  # To avoid further modification on model level.
            res["attention_mask"] = res["attention_mask"][:, :1]

        return res

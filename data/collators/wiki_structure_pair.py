import copy
import random

import torch
from torch.utils.data.dataset import T_co
from typing import List
from transformers import PreTrainedTokenizer

from data.collators.wiki import WikiPathDatasetV5
from data.collators.wiki_tokens_collator import WikiPathTokensDatasetCollator
from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.structure.pair.collator")


class WikiPathDatasetV6wPair(WikiPathDatasetV5):
    def __init__(self, examples, group_pos_pairs, raw_texts, add_hard_negative: bool = True, max_neg_num: int = 2,
                 add_negative: bool = True, hard_negative_num: int = 1):
        super().__init__(examples, raw_texts)
        self.add_hard_negative = add_hard_negative
        self.hard_negative_num = hard_negative_num
        self.max_neg_num = max_neg_num
        self.add_negative = add_negative

        logger.info(f"{self.add_hard_negative}\t{self.add_negative}\t{self.max_neg_num}\t{self.hard_negative_num}")

        self.group_pos_pairs = {}
        cnt = 0
        for k, v in group_pos_pairs.items():
            if len(v):
                self.group_pos_pairs[k] = v
                cnt += 1
        logger.info(f"{cnt}")

    def __getitem__(self, index) -> T_co:
        item = super().__getitem__(index)

        if index < len(self.group_pos_pairs):
            _idx = list(self.group_pos_pairs.keys())[index]
        else:
            _idx = random.choice(list(self.group_pos_pairs.keys()))

        pair = random.choice(self.group_pos_pairs[_idx])
        pair = [pair[0], pair[1]]
        random.shuffle(pair)
        pair_example_q = self.examples[pair[0]]
        pair_example_k = self.examples[pair[1]]

        # assert len(pair_example_q["hard_neg_ids"]) + len(pair_example_q["neg_ids"])
        neg_examples = []
        neg_num = min(len(pair_example_q["hard_neg_ids"]), self.hard_negative_num)
        if len(pair_example_q["hard_neg_ids"]) and self.add_hard_negative:
            neg_examples.extend(self.examples[idx] for idx in random.sample(pair_example_q["hard_neg_ids"], neg_num))

        neg_num = min(len(pair_example_q["neg_ids"]), self.max_neg_num - len(neg_examples))
        if len(pair_example_q["neg_ids"]) and self.add_negative:
            neg_examples.extend([self.examples[idx] for idx in random.sample(pair_example_q["neg_ids"], neg_num)])

        item["pair_q"] = pair_example_q
        item["pair_k"] = pair_example_k
        item["pair_neg"] = neg_examples

        return item


def prepare_single_example_positive(example, dropout_option: bool = False):
    if dropout_option:
        option_span = example["sentence_spans"][0][-1]
        tokens = example["tokens"][0]
        new_tokens = tokens[:option_span[0]] + tokens[option_span[1]:]
        return new_tokens
    return example["tokens"][0]


class WikiPathDatasetPairCollator(WikiPathTokensDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15,
                 option_dropout: float = 0.0, max_neg_num: int = 2, ):
        super().__init__(max_seq_length, tokenizer, mlm_probability)
        self.option_dropout = option_dropout
        self.max_neg_num = max_neg_num

    def __call__(self, batch):
        batch_size = len(batch)
        batch_input_tokens = []
        max_seq_length = 0
        batch_pair_mask = torch.zeros(batch_size, self.max_neg_num + 1)
        dropped_op_cnt = 0
        for b_id, b in enumerate(batch):
            item_input_tokens = []
            r = random.random()
            for exp_id, exp in enumerate([b["pair_q"], b["pair_k"]] + b["pair_neg"]):
                if r < self.option_dropout and exp_id == 0:
                    dropped_op_cnt += 1
                    exp_tokens = prepare_single_example_positive(exp, dropout_option=True)
                else:
                    exp_tokens = prepare_single_example_positive(exp, dropout_option=False)

                max_seq_length = max(max_seq_length, len(exp_tokens))
                item_input_tokens.append(exp_tokens)
            assert len(b["pair_neg"]) <= self.max_neg_num, len(b["pair_neg"])

            batch_pair_mask[b_id, :(len(b["pair_neg"]) + 1)] = 1

            batch_input_tokens.append(item_input_tokens)

        input_ids = torch.zeros(batch_size, self.max_neg_num + 2, max_seq_length, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(batch_size, self.max_neg_num + 2, max_seq_length, dtype=torch.long)
        for b, item_tokens in enumerate(batch_input_tokens):
            for i, exp_tokens in enumerate(item_tokens):
                input_ids[b, i, :len(exp_tokens)] = torch.tensor(self.tokenizer.convert_tokens_to_ids(exp_tokens), dtype=torch.long)
                attention_mask[b, i, :len(exp_tokens)] = 1

        res = super().__call__(batch)

        res["pair_input_ids"] = input_ids
        res["pair_attention_mask"] = attention_mask
        res["pair_mask"] = batch_pair_mask
        res["pair_labels"] = torch.zeros(batch_size, dtype=torch.long)

        no_neg_mask = batch_pair_mask.sum(dim=1) == 1
        res["pair_labels"].masked_fill_(no_neg_mask, -1)

        res["dropped_op_cnt"] = dropped_op_cnt

        return res


def re_organize_tokens(tokens, sentence_spans, sentence_ids):
    new_tokens = []
    sentence_ids = set(sentence_ids)
    s = 0
    for sent_id, span in enumerate(sentence_spans):
        if sent_id in sentence_ids:
            new_tokens.extend(tokens[s:span[1]])
            s = span[1]
        else:
            new_tokens.extend(tokens[s:span[0]])
            s = span[1]
    new_tokens.extend(tokens[s:])
    return new_tokens


def prepare_partial_single_example_positive(example, sentence_ids: List[int] = None):
    if sentence_ids is None:
        sentence_num = len(example["sentence_spans"][0])
        kept_sentence_ids = random.sample(list(range(len(example["sentence_spans"][0]))), sentence_num // 2)
        sentence_ids = [i for i in range(len(example["sentence_spans"][0])) if i not in kept_sentence_ids]
        return re_organize_tokens(example["tokens"][0], example["sentence_spans"][0], kept_sentence_ids), sentence_ids
    else:
        return re_organize_tokens(example["tokens"][0], example["sentence_spans"][0], sentence_ids)


def get_sent_rank2id(example):
    s_ids = example["path_s_ids"]
    rank2id = list(copy.deepcopy(s_ids))
    rank2id.append("pos")  # For positive padding.
    assert len(rank2id) == len(example["sentence_spans"][0])
    return rank2id


def get_sent_id2rank(example):
    s_ids = example["path_s_ids"]
    id2rank = {}
    for s_rank, s_id in enumerate(s_ids):
        id2rank[s_id] = s_rank
    id2rank["pos"] = len(example["sentence_spans"][0]) - 1
    return id2rank


def prepare_partial_single_example_positive_fixed(example, sentence_ids: List[str] = None):
    if sentence_ids is None:
        sentence_num = len(example["sentence_spans"][0])
        kept_sentence_ranks = random.sample(list(range(len(example["sentence_spans"][0]))), sentence_num // 2)
        sentence_ids = []
        sent_rank2id = get_sent_rank2id(example)
        for i in range(sentence_num):
            if i not in kept_sentence_ranks:
                sentence_ids.append(sent_rank2id[i])
        return re_organize_tokens(example["tokens"][0], example["sentence_spans"][0], kept_sentence_ranks), sentence_ids
    else:
        sent_id2rank = get_sent_id2rank(example)
        try:
            sentence_ranks = [sent_id2rank[s_id] for s_id in sentence_ids]
        except:
            logger.info(f"{sentence_ids}\t{sent_id2rank}")
        return re_organize_tokens(example["tokens"][0], example["sentence_spans"][0], sentence_ranks)


def combine_two_token_sequences(tokens1, tokens2):
    return tokens1 + tokens2[1:]


def truncate_tokens(tokens, max_seq_length, tokenizer: PreTrainedTokenizer):
    if len(tokens) > max_seq_length:
        tokens = tokens[:(max_seq_length - 1)] + [tokenizer.sep_token]
        assert len(tokens) == max_seq_length
    return tokens


class WikiPathDatasetPartialPair(WikiPathTokensDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_neg_num: int = 2):
        # super().__init__(max_seq_length, tokenizer, mlm_probability)
        super().__init__(max_seq_length, tokenizer, decoder_tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.max_neg_num = max_neg_num

    def __call__(self, batch):
        batch_size = len(batch)
        batch_input_tokens = []
        max_seq_length = 0
        batch_pair_mask = torch.zeros(batch_size, self.max_neg_num + 1)
        for b_id, b in enumerate(batch):
            item_input_tokens = []

            # exp_tokens, left_sentence_ids = prepare_partial_single_example_positive(b["pair_q"], sentence_ids=None)
            exp_tokens, left_sentence_ids = prepare_partial_single_example_positive_fixed(b["pair_q"], sentence_ids=None)
            item_input_tokens.append(exp_tokens)
            max_seq_length = max(max_seq_length, len(exp_tokens))

            exp_tokens = prepare_partial_single_example_positive_fixed(b["pair_k"], sentence_ids=left_sentence_ids)
            item_input_tokens.append(exp_tokens)
            max_seq_length = max(max_seq_length, len(exp_tokens))

            for exp in b["pair_neg"]:
                exp_tokens, _ = prepare_partial_single_example_positive_fixed(exp, sentence_ids=None)
                item_input_tokens.append(exp_tokens)
                max_seq_length = max(max_seq_length, len(exp_tokens))

            assert len(b["pair_neg"]) <= self.max_neg_num, len(b["pair_neg"])

            batch_pair_mask[b_id, :(len(b["pair_neg"]) + 1)] = 1

            batch_input_tokens.append(item_input_tokens)

        input_ids = torch.zeros(batch_size, self.max_neg_num + 2, max_seq_length, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(batch_size, self.max_neg_num + 2, max_seq_length, dtype=torch.long)
        for b, item_tokens in enumerate(batch_input_tokens):
            for i, exp_tokens in enumerate(item_tokens):
                input_ids[b, i, :len(exp_tokens)] = torch.tensor(self.tokenizer.convert_tokens_to_ids(exp_tokens), dtype=torch.long)
                attention_mask[b, i, :len(exp_tokens)] = 1

        res = super().__call__(batch)

        res["pair_input_ids"] = input_ids
        res["pair_attention_mask"] = attention_mask
        res["pair_mask"] = batch_pair_mask
        res["pair_labels"] = torch.zeros(batch_size, dtype=torch.long)

        no_neg_mask = batch_pair_mask.sum(dim=1) == 1
        res["pair_labels"].masked_fill_(no_neg_mask, -1)

        return res


class WikiPathCollatorPartialPairOneTower(WikiPathTokensDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15,
                 max_neg_num: int = 2, ):
        super().__init__(max_seq_length, tokenizer, mlm_probability)
        self.max_neg_num = max_neg_num

    def __call__(self, batch):
        batch_size = len(batch)
        batch_input_tokens = []
        max_seq_length = 0
        batch_pair_mask = torch.zeros(batch_size, self.max_neg_num + 1)
        for b_id, b in enumerate(batch):
            item_input_tokens = []

            anchor_tokens, left_sentence_ids = prepare_partial_single_example_positive_fixed(b["pair_q"], sentence_ids=None)

            exp_tokens = prepare_partial_single_example_positive_fixed(b["pair_k"], sentence_ids=left_sentence_ids)
            exp_tokens = combine_two_token_sequences(anchor_tokens, exp_tokens)
            exp_tokens = truncate_tokens(exp_tokens, self.max_seq_length, self.tokenizer)
            item_input_tokens.append(exp_tokens)
            max_seq_length = max(max_seq_length, len(exp_tokens))

            for exp in b["pair_neg"]:
                exp_tokens, _ = prepare_partial_single_example_positive_fixed(exp, sentence_ids=None)
                exp_tokens = combine_two_token_sequences(anchor_tokens, exp_tokens)
                exp_tokens = truncate_tokens(exp_tokens, self.max_seq_length, self.tokenizer)
                item_input_tokens.append(exp_tokens)
                max_seq_length = max(max_seq_length, len(exp_tokens))

            assert len(b["pair_neg"]) <= self.max_neg_num, len(b["pair_neg"])

            batch_pair_mask[b_id, :(len(b["pair_neg"]) + 1)] = 1

            batch_input_tokens.append(item_input_tokens)

        input_ids = torch.zeros(batch_size, self.max_neg_num + 1, max_seq_length, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(batch_size, self.max_neg_num + 1, max_seq_length, dtype=torch.long)
        for b, item_tokens in enumerate(batch_input_tokens):
            for i, exp_tokens in enumerate(item_tokens):
                input_ids[b, i, :len(exp_tokens)] = torch.tensor(self.tokenizer.convert_tokens_to_ids(exp_tokens), dtype=torch.long)
                attention_mask[b, i, :len(exp_tokens)] = 1

        res = super().__call__(batch)

        res["pair_input_ids"] = input_ids
        res["pair_attention_mask"] = attention_mask
        res["pair_mask"] = batch_pair_mask
        res["pair_labels"] = torch.zeros(batch_size, dtype=torch.long)

        no_neg_mask = batch_pair_mask.sum(dim=1) == 1
        res["pair_labels"].masked_fill_(no_neg_mask, -1)

        return res


import collections
import pickle
import random
from collections import Counter
from typing import Tuple, Optional, Dict, Any

import torch
import transformers
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, PreTrainedTokenizer, T5Tokenizer, BartTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path")


class WikiPathDatasetV5(Dataset):
    def __init__(self, examples, raw_texts):
        self.examples = examples

        _aligned_texts = []
        while len(_aligned_texts) < len(examples):
            diff = len(examples) - len(_aligned_texts)
            if diff < len(raw_texts):
                _aligned_texts.extend(random.sample(raw_texts, diff))
            else:
                _aligned_texts.extend(raw_texts[:])
        assert len(_aligned_texts) == len(self.examples)

        cnt = Counter(list(map(lambda x: len(x["negative"]) if "negative" in x else len(x["negative_context"]), examples)))
        assert len(cnt) == 1, cnt

        self.raw_texts = _aligned_texts

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        text = self.raw_texts[index]
        return {
            "example": example,
            "text": text,
            "index": index,
        }


class WikiPathDatasetV6wPatternPair(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, pattern_pair_file: str, add_cf_pair_data: bool = True):
        super().__init__(examples, raw_texts)

        self.id2exp = collections.defaultdict(list)
        for exp_id, exp in enumerate(self.examples):
            self.id2exp[exp["orig_id"]].append(exp_id)

        self.pattern_pairs = pickle.load(open(pattern_pair_file, "rb"))

        self.add_cf_pair_data = add_cf_pair_data

    def __getitem__(self, index) -> T_co:
        item = super().__getitem__(index)

        paired_example_ids = []
        if item["example"]["orig_id"] in self.pattern_pairs and len(self.pattern_pairs[item["example"]["orig_id"]]):
            paired_example_orig_ids = self.pattern_pairs[item["example"]["orig_id"]]
            for paired_exp_id in paired_example_orig_ids:
                paired_example_ids.extend(self.id2exp[paired_exp_id])
        else:
            paired_example_orig_ids = []
            if self.add_cf_pair_data:
                # If there is no paired examples, we add corresponding original examples of itself as the paired examples.
                # The only thing to check is to avoid the specific example itself.
                paired_example_ids.extend(self.id2exp[item["example"]["orig_id"]])

        paired_example_ids = list(set(paired_example_ids) - {index})

        if len(paired_example_ids) > 0:
            paired_example = self.examples[random.choice(paired_example_ids)]
        else:
            paired_example = None

        item["paired_example_orig_ids"] = paired_example_orig_ids
        item["paired_example"] = paired_example

        return item


class WikiPathDatasetV6wPatternPairFull(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, pattern_pair_file: str):
        super().__init__(examples, raw_texts)

        logger.info(f"Concatenating example sentences...")
        for exp in self.examples:
            exp["context_sentences"] = exp["context"]
            exp["context"] = " ".join(exp["context"])
            if "negative_context" in exp:
                exp["negative_context"] = [" ".join(neg_ctx) for neg_ctx in exp["negative_context"]]

        self.id2exp = collections.defaultdict(list)
        for exp_id, exp in enumerate(self.examples):
            self.id2exp[exp["orig_id"]].append(exp_id)

        pattern_pairs = pickle.load(open(pattern_pair_file, "rb"))
        pattern_pair_cnt = 0
        self.pattern_pairs = {}
        for orig_id in pattern_pairs:
            if len(pattern_pairs[orig_id]):
                self.pattern_pairs[orig_id] = pattern_pairs[orig_id]
                pattern_pair_cnt += len(pattern_pairs[orig_id])
        self.pattern_pair_keys = list(self.pattern_pairs.keys())
        logger.info(f"Pattern pair keys: {len(self.pattern_pair_keys)}")
        logger.info(f"Total pattern pairs: {pattern_pair_cnt}")

    def __getitem__(self, index) -> T_co:
        item = super().__getitem__(index)

        # paired_example_ids = []
        # if item["example"]["orig_id"] in self.pattern_pairs and len(self.pattern_pairs[item["example"]["orig_id"]]):
        #     paired_example_orig_ids = self.pattern_pairs[item["example"]["orig_id"]]
        #     for paired_exp_id in paired_example_orig_ids:
        #         paired_example_ids.extend(self.id2exp[paired_exp_id])
        # else:
        #     paired_example_orig_ids = []
        #     if self.add_cf_pair_data:
        #         # If there is no paired examples, we add corresponding original examples of itself as the paired examples.
        #         # The only thing to check is to avoid the specific example itself.
        #         paired_example_ids.extend(self.id2exp[item["example"]["orig_id"]])
        #
        # paired_example_ids = list(set(paired_example_ids) - {index})
        #
        # if len(paired_example_ids) > 0:
        #     paired_example = self.examples[random.choice(paired_example_ids)]
        # else:
        #     paired_example = None
        if index < len(self.pattern_pairs):
            _idx = index
        else:
            _idx = random.choice(list(range(len(self.pattern_pair_keys))))
        pair_example = self.examples[random.choice(self.id2exp[self.pattern_pair_keys[_idx]])]
        assert pair_example["orig_id"] == self.pattern_pair_keys[_idx]

        paired_example_orig_ids = self.pattern_pairs[pair_example["orig_id"]]
        paired_example_ids = []
        for paired_exp_orig_id in paired_example_orig_ids:
            paired_example_ids.extend(self.id2exp[paired_exp_orig_id])
        assert len(paired_example_ids)

        paired_example = self.examples[random.choice(paired_example_ids)]
        assert paired_example["orig_id"] in paired_example_orig_ids

        item["pair_q"] = pair_example
        item["pair_k_orig_ids"] = paired_example_orig_ids
        item["pair_k"] = paired_example

        return item


class WikiPathDatasetRelGenerateV1(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, id2rel_path_decode_id_file: str, rel_vocab: str):
        super().__init__(examples, raw_texts)

        self.id2rel_path_decode_ids = pickle.load(open(id2rel_path_decode_id_file, "rb"))
        self.rel_vocab = pickle.load(open(rel_vocab, "rb"))
        self.eos_token_id = len(self.rel_vocab)
        self.pad_token_id = len(self.rel_vocab) + 1

    def __getitem__(self, index):
        item = super().__getitem__(index)

        example_id = item["example"]["orig_id"]

        if example_id in self.id2rel_path_decode_ids:
            path_decode_input_a = self.id2rel_path_decode_ids[example_id]["input_a"]
            path_decode_input_b = self.id2rel_path_decode_ids[example_id]["input_b"]

            if path_decode_input_b == -1:
                item["rel_labels"] = path_decode_input_a + [len(self.rel_vocab)]  # as </s> token
            else:
                item["rel_labels"] = path_decode_input_a + [path_decode_input_b, len(self.rel_vocab)]
        else:
            item["rel_labels"] = [-1]

        return item


class WikiPathDatasetGenerate(Dataset):
    """
    It seems that the dataset class is not relevant to ``generation``.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        return {
            "example": example,
            "index": index,
        }


class WikiPathDatasetCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4):
        self.max_seq_length = max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.mlm_probability = mlm_probability
        self.max_option_num = max_option_num

        transformers.logging.set_verbosity_error()  # Avoid truncation warning.

    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        examples, texts = [], []
        for b in batch:
            # assert list(b.keys()) == ["example", "text"], b.keys()
            examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch

        sentences = []
        options = []
        for e in examples:
            op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            options.extend(op)
            sentences.extend([e["context"]] * len(op))
        batch_size = len(examples)
        # option_num = len(examples[0]["negative"]) + 1
        option_num = min(len(examples[0]["negative"]) + 1, self.max_option_num)

        tokenizer_outputs = self.tokenizer(sentences, options, padding=PaddingStrategy.MAX_LENGTH,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, self.max_seq_length),
            "attention_mask": attention_mask.reshape(batch_size, option_num, self.max_seq_length),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            # Remove padding.
            special_tokens_mask = special_tokens_mask | (labels == self.tokenizer.pad_token_id)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class WikiPathDatasetCollatorOnlyMLM(WikiPathDatasetCollator):
    def __call__(self, batch):
        texts = []
        for b in batch:
            texts.append(b.pop("text"))
        del batch

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        return {
            "input_ids": mlm_input_ids,
            "attention_mask": mlm_attention_mask,
            "labels": mlm_labels
        }


class WikiPathDatasetCollatorWithContext(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap

    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        op_examples, ctx_examples, texts = [], [], []
        for b in batch:
            example = b.pop("example")
            if "negative_context" in example:
                ctx_examples.append(example)
            else:
                op_examples.append(example)
            # examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch
        batch_size = len(op_examples) + len(ctx_examples)
        assert batch_size == len(texts)

        # TODO: Check if other possible input formats are ok, e.g., <rest context> <sep> <pseudo/ground truth edge> <sep> <sep> <option>

        input_a = []
        input_b = []
        option_num = -1

        for e in op_examples:
            op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([e["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([e["context"]] * len(op))
            if option_num == -1:
                option_num = len(op)
            else:
                assert option_num == len(op)

        for e in ctx_examples:
            positive_context = e.pop("context")
            negative_context = e.pop("negative_context")
            op = e.pop("condition")
            input_a.extend([positive_context] + negative_context)
            input_b.extend([op] * (len(negative_context) + 1))
            if option_num == -1:
                option_num = len(negative_context) + 1
            else:
                assert option_num == len(negative_context) + 1, (option_num, len(negative_context))

        option_num = min(option_num, self.max_option_num)

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.MAX_LENGTH,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, self.max_seq_length),
            "attention_mask": attention_mask.reshape(batch_size, option_num, self.max_seq_length),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


class WikiPathDatasetCollatorWithContextAndPair(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b)

        paired_input_a = self.tokenizer.pad_token
        paired_input_b = self.tokenizer.pad_token
        paired_example = item["paired_example"]
        paired_example_id = -1
        if paired_example is not None:
            paired_example_id = paired_example["orig_id"]
            if "negative_context" in paired_example:
                paired_input_a = paired_example["context"]
                paired_input_b = paired_example["condition"]
            else:
                if self.swap:
                    paired_input_a = paired_example["context"]
                    paired_input_b = paired_example["positive"]
                else:
                    paired_input_a = paired_example["positive"]
                    paired_input_b = paired_example["context"]

        return input_a, input_b, paired_input_a, paired_input_b, item["example"]["orig_id"], paired_example_id, \
               item["paired_example_orig_ids"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        paired_input_a, paired_input_b, orig_ids, paired_orig_ids, paired_orig_id_list = [], [], [], [], []
        for b in batch:
            b_input_a, b_input_b, b_paired_input_a, b_paired_input_b, b_orig_id, b_paired_orig_id, b_pair_id_list = \
                self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)
            paired_input_a.append(b_paired_input_a)
            paired_input_b.append(b_paired_input_b)
            orig_ids.append(b_orig_id)
            paired_orig_ids.append(b_paired_orig_id)
            paired_orig_id_list.append(b_pair_id_list)
            texts.append(b["text"])

        del batch

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.LONGEST,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_tokenizer_outputs = self.tokenizer(paired_input_a, paired_input_b, padding=PaddingStrategy.LONGEST,
                                                truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                                return_tensors="pt")
        pair_input_ids = pair_tokenizer_outputs["input_ids"]
        pair_attention_mask = pair_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        assert len(paired_orig_id_list) == len(orig_ids) == len(paired_orig_ids)
        for a_id, (a_orig_id, a_paired_orig_id_list) in enumerate(zip(orig_ids, paired_orig_id_list)):
            # pair dot product
            pair_mask = []
            for b_id, b_paired_orig_id in enumerate(paired_orig_ids):
                if a_id == b_id:
                    if b_paired_orig_id != -1:
                        pair_align_labels.append(b_id)
                        assert b_paired_orig_id in a_paired_orig_id_list or b_paired_orig_id == a_orig_id
                    else:
                        pair_align_labels.append(-1)
                    pair_mask.append(0)
                else:
                    if b_paired_orig_id in a_paired_orig_id_list or b_paired_orig_id == -1 or b_paired_orig_id == a_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)

            # self dot product
            for c_id, c_orig_id in enumerate(orig_ids):
                if c_id == a_id:
                    pair_mask.append(1)
                else:
                    if c_orig_id in a_paired_orig_id_list or c_orig_id == a_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)

            # assert sum(pair_mask) + 2 <= len(pair_mask), pair_mask
            assert len(pair_mask) == len(orig_ids) * 2
            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_input_ids": pair_input_ids,
            "pair_attention_mask": pair_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": (pair_align_labels > -1).sum(),
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / batch_size,
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


class WikiPathDatasetCollatorWithContextAndPairComplete(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 option_dropout: float = 0.0, k_option_dropout: float = 0.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap
        self.option_dropout = option_dropout
        self.k_option_dropout = k_option_dropout

    @staticmethod
    def prepare_single_example_positive(example):
        if "negative_context" in example:
            input_a = example["context"]
            input_b = example["condition"]
        else:
            # Remove `swap` for easy implementation of `option` dropout.
            # if self.swap:
            #     input_a = example["context"]
            #     input_b = example["positive"]
            # else:
            #     input_a = example["positive"]
            #     input_b = example["context"]
            input_a = example["context"]
            input_b = example["positive"]
        return input_a, input_b

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b) == self.max_option_num

        pair_q_input_a, pair_q_input_b = self.prepare_single_example_positive(item["pair_q"])
        pair_k_input_a, pair_k_input_b = self.prepare_single_example_positive(item["pair_k"])

        return input_a, input_b, item["pair_q"]["orig_id"], pair_q_input_a, pair_q_input_b, \
               item["pair_k"]["orig_id"], pair_k_input_a, pair_k_input_b, \
               item["pair_k_orig_ids"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        pair_q_orig_ids, pair_q_a, pair_q_b, pair_k_orig_ids, pair_k_a, pair_k_b, all_paired_orig_ids = [], [], [], [], [], [], []
        dropped_op_cnt = 0
        for b in batch:
            b_input_a, b_input_b, b_pair_q_orig_id, b_pair_q_a, b_pair_q_b, \
            b_pair_k_orig_id, b_pair_k_a, b_pair_k_b, \
            b_pair_k_orig_ids = self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)
            pair_q_orig_ids.append(b_pair_q_orig_id)
            pair_q_a.append(b_pair_q_a)

            _r = random.random()
            if _r < self.option_dropout:
                pair_q_b.append("")
                dropped_op_cnt += 1
            else:
                pair_q_b.append(b_pair_q_b)

            pair_k_orig_ids.append(b_pair_k_orig_id)
            pair_k_a.append(b_pair_k_a)

            _r = random.random()
            if _r < self.k_option_dropout:
                pair_k_b.append("")
            else:
                pair_k_b.append(b_pair_k_b)

            all_paired_orig_ids.append(b_pair_k_orig_ids)
            texts.append(b["text"])

        del batch
        _tokenize_kwargs = {
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "max_length": self.max_seq_length,
            "return_tensors": "pt"
        }

        tokenizer_outputs = self.tokenizer(input_a, input_b, **_tokenize_kwargs)
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_q_tokenizer_outputs = self.tokenizer(pair_q_a, pair_q_b, **_tokenize_kwargs)
        pair_q_input_ids = pair_q_tokenizer_outputs["input_ids"]
        pair_q_attention_mask = pair_q_tokenizer_outputs["attention_mask"]

        pair_k_tokenizer_outputs = self.tokenizer(pair_k_a, pair_k_b, **_tokenize_kwargs)
        pair_k_input_ids = pair_k_tokenizer_outputs["input_ids"]
        pair_k_attention_mask = pair_k_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        assert len(pair_q_orig_ids) == len(pair_k_orig_ids) == len(all_paired_orig_ids)
        for q_id, (q_orig_id, q_paired_orig_id_list) in enumerate(zip(pair_q_orig_ids, all_paired_orig_ids)):
            # pair dot product
            pair_mask = []
            for k_id, k_orig_id in enumerate(pair_k_orig_ids):
                if q_id == k_id:
                    assert k_orig_id in q_paired_orig_id_list
                    pair_mask.append(0)
                else:
                    if k_orig_id in q_paired_orig_id_list or k_orig_id == q_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)
            if sum(pair_mask) + 1 == len(pair_k_orig_ids):
                pair_align_labels.append(-1)
            else:
                pair_align_labels.append(q_id)

            # TODO: Add self-product for more negative samples? Then the option dropout should be modified.

            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)
        num_labels = (pair_align_labels > -1).sum()

        mlm_tokenize_outputs = self.tokenizer(texts, **_tokenize_kwargs)
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_q_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_q_input_ids": pair_q_input_ids,
            "pair_q_attention_mask": pair_q_attention_mask,
            "pair_k_input_ids": pair_k_input_ids,
            "pair_k_attention_mask": pair_k_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": num_labels,
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / num_labels,
            "dropped_op_cnt": torch.tensor([dropped_op_cnt])
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        # print("pair_value_num", res["pair_value_num"])
        return res


class WikiPathDatasetCollatorWithContextAndPairCompleteDropout(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 q_option_dropout: float = 0.0, k_option_dropout: float = 0.0, k_context_dropout: float = 0.0, add_tagging: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap
        self.q_option_dropout = q_option_dropout
        self.k_option_dropout = k_option_dropout
        self.k_context_dropout = k_context_dropout
        self.add_tagging = add_tagging

    def concat_sentence_and_dropout(self, item):
        # item["example"]["context"] = " ".join(item["example"]["context"])
        # if "negative_context" in item["example"]:
        #     item["example"]["negative_context"] = [" ".join(ctx) for ctx in item["example"]["negative_context"]]
        #
        # item["pair_q"]["context"] = " ".join(item["pair_q"]["context"])

        k_sent_drop_cnt = 0
        if self.k_context_dropout > 0:
            dropped_context = []
            for _sent in item["pair_k"]["context_sentences"]:
                _r = random.random()
                if _r < self.k_context_dropout:
                    k_sent_drop_cnt += 1
                else:
                    dropped_context.append(_sent)
            item["pair_k"]["context"] = " ".join(dropped_context)

        # item["pair_k"]["context"] = " ".join(item["pair_k"]["context"])
        # if "negative_context" in item["pair_k"]:
        #     item["pair_k"]["negative_context"] = [" ".join(ctx) for ctx in item["pair_k"]["negative_context"]]

        return item, k_sent_drop_cnt

    @staticmethod
    def prepare_single_example_positive(example):
        if "negative_context" in example:
            input_a = example["context"]
            input_b = example["condition"]
        else:
            input_a = example["context"]
            input_b = example["positive"]
        return input_a, input_b

    def prepare_single_example(self, item):
        input_a = []
        input_b = []
        if "negative_context" in item["example"]:
            input_a.extend(([item["example"]["context"]] + item["example"]["negative_context"])[:self.max_option_num])
            input_b.extend([item["example"]["condition"]] * self.max_option_num)
        else:
            op = ([item["example"]["positive"]] + item["example"]["negative"])[:self.max_option_num]
            if self.swap:
                input_a.extend([item["example"]["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([item["example"]["context"]] * len(op))

        assert len(input_a) == len(input_b) == self.max_option_num

        pair_q_input_a, pair_q_input_b = self.prepare_single_example_positive(item["pair_q"])
        pair_k_input_a, pair_k_input_b = self.prepare_single_example_positive(item["pair_k"])

        return input_a, input_b, item["pair_q"]["orig_id"], pair_q_input_a, pair_q_input_b, \
               item["pair_k"]["orig_id"], pair_k_input_a, pair_k_input_b, \
               item["pair_k_orig_ids"]

    def __call__(self, batch):
        input_a, input_b, texts = [], [], []
        pair_q_orig_ids, pair_q_a, pair_q_b, pair_k_orig_ids, pair_k_a, pair_k_b, all_paired_orig_ids = [], [], [], [], [], [], []
        dropped_op_cnt = 0
        k_sent_drop_cnt = 0
        for b in batch:
            b, b_k_sent_drop_cnt = self.concat_sentence_and_dropout(b)
            k_sent_drop_cnt += b_k_sent_drop_cnt

            b_input_a, b_input_b, b_pair_q_orig_id, b_pair_q_a, b_pair_q_b, \
            b_pair_k_orig_id, b_pair_k_a, b_pair_k_b, \
            b_pair_k_orig_ids = self.prepare_single_example(b)

            input_a.extend(b_input_a)
            input_b.extend(b_input_b)
            pair_q_orig_ids.append(b_pair_q_orig_id)
            pair_q_a.append(b_pair_q_a)

            _r = random.random()
            if _r < self.q_option_dropout:
                pair_q_b.append("")
                dropped_op_cnt += 1
            else:
                pair_q_b.append(b_pair_q_b)

            pair_k_orig_ids.append(b_pair_k_orig_id)
            pair_k_a.append(b_pair_k_a)

            _r = random.random()
            if _r < self.k_option_dropout:
                pair_k_b.append("")
            else:
                pair_k_b.append(b_pair_k_b)

            all_paired_orig_ids.append(b_pair_k_orig_ids)
            texts.append(b["text"])

        if self.add_tagging:
            tagging_labels = [b["example"].pop("tagging_label") for b in batch]
        else:
            tagging_labels = None

        del batch
        _tokenize_kwargs = {
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "max_length": self.max_seq_length,
            "return_tensors": "pt"
        }

        tokenizer_outputs = self.tokenizer(input_a, input_b, **_tokenize_kwargs)
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        pair_q_tokenizer_outputs = self.tokenizer(pair_q_a, pair_q_b, **_tokenize_kwargs)
        pair_q_input_ids = pair_q_tokenizer_outputs["input_ids"]
        pair_q_attention_mask = pair_q_tokenizer_outputs["attention_mask"]

        pair_k_tokenizer_outputs = self.tokenizer(pair_k_a, pair_k_b, **_tokenize_kwargs)
        pair_k_input_ids = pair_k_tokenizer_outputs["input_ids"]
        pair_k_attention_mask = pair_k_tokenizer_outputs["attention_mask"]

        # prepare pair labels and mask
        pair_align_mask = []  # `1` for mask and `0` for true value.
        pair_align_labels = []  # `-1` for non-pair examples.
        assert len(pair_q_orig_ids) == len(pair_k_orig_ids) == len(all_paired_orig_ids)
        for q_id, (q_orig_id, q_paired_orig_id_list) in enumerate(zip(pair_q_orig_ids, all_paired_orig_ids)):
            # pair dot product
            pair_mask = []
            for k_id, k_orig_id in enumerate(pair_k_orig_ids):
                if q_id == k_id:
                    assert k_orig_id in q_paired_orig_id_list
                    pair_mask.append(0)
                else:
                    if k_orig_id in q_paired_orig_id_list or k_orig_id == q_orig_id:
                        pair_mask.append(1)
                    else:
                        pair_mask.append(0)
            if sum(pair_mask) + 1 == len(pair_k_orig_ids):
                pair_align_labels.append(-1)
            else:
                pair_align_labels.append(q_id)

            # TODO: Add self-product for more negative samples? Then the option dropout should be modified.

            pair_align_mask.append(pair_mask)

        pair_align_mask = torch.tensor(pair_align_mask, dtype=torch.int)
        pair_align_labels = torch.tensor(pair_align_labels, dtype=torch.long)
        num_labels = (pair_align_labels > -1).sum()

        mlm_tokenize_outputs = self.tokenizer(texts, **_tokenize_kwargs)
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        batch_size = len(pair_q_input_ids)
        option_num = self.max_option_num

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, -1),
            "attention_mask": attention_mask.reshape(batch_size, option_num, -1),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
            "pair_q_input_ids": pair_q_input_ids,
            "pair_q_attention_mask": pair_q_attention_mask,
            "pair_k_input_ids": pair_k_input_ids,
            "pair_k_attention_mask": pair_k_attention_mask,
            "pair_mask": pair_align_mask,
            "pair_labels": pair_align_labels,
            "pair_label_num": num_labels,
            "pair_value_num": (1 - pair_align_mask).sum(dim=-1)[pair_align_labels > -1].sum() / num_labels,
            "dropped_op_cnt": dropped_op_cnt,
            "k_sent_drop_cnt": k_sent_drop_cnt / batch_size
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]

        if self.add_tagging:
            tagging_labels = torch.tensor(tagging_labels, dtype=torch.long)
            tagging_labels = tagging_labels[:, :input_ids.size(1)]
            tagging_labels[res["input_ids"][:, 0] == self.tokenizer.pad_token_id] = -1
            res["tagging_labels"] = tagging_labels

        # print("pair_value_num", res["pair_value_num"])
        return res


class WikiPathDatasetCollatorWithContextInMLMPredict(WikiPathDatasetCollatorWithContext):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False,
                 use_mask: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap
        self.use_mask = use_mask

    def __call__(self, batch):
        index = torch.tensor([b["index"] for b in batch], dtype=torch.long)
        inputs = super().__call__(batch)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        assert input_ids.size(1) == self.max_option_num
        input_ids = input_ids.reshape(-1, self.max_seq_length)
        attention_mask = attention_mask.reshape(-1, self.max_seq_length)

        sep_mask = input_ids == self.tokenizer.sep_token_id
        acc_sep_mask = sep_mask.cumsum(dim=1)
        sep_mask_num = acc_sep_mask.max(dim=1, keepdim=True)[0]
        mlm_mask = (acc_sep_mask.eq(sep_mask_num) * attention_mask).bool()

        if self.use_mask:
            input_ids[mlm_mask] = self.tokenizer.mask_token_id

        labels = input_ids.clone()
        labels[~mlm_mask] = -1

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": labels,
            "labels": inputs["labels"],
            "index": index
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids.reshape(-1, self.max_seq_length)

        return inputs


class WikiPathDatasetCollatorWithContextBinary(WikiPathDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num)
        self.swap = swap

    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        op_examples, ctx_examples, texts = [], [], []
        for b in batch:
            example = b.pop("example")
            if "negative_context" in example:
                ctx_examples.append(example)
            else:
                op_examples.append(example)
            # examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch
        batch_size = len(op_examples) + len(ctx_examples)
        assert batch_size == len(texts)

        input_a = []
        input_b = []
        option_num = -1

        for e in op_examples:
            # op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            op = [e["positive"]] + [random.choice(e["negative"])]
            if self.swap:
                input_a.extend([e["context"]] * len(op))
                input_b.extend(op)
            else:
                input_a.extend(op)
                input_b.extend([e["context"]] * len(op))
            if option_num == -1:
                option_num = len(op)
            else:
                assert option_num == len(op)

        for e in ctx_examples:
            positive_context = e.pop("context")
            negative_context = e.pop("negative_context")
            op = e.pop("condition")
            # input_a.extend([positive_context] + negative_context)
            # input_b.extend([op] * (len(negative_context) + 1))
            input_a.extend([positive_context] + [random.choice(negative_context)])
            input_b.extend([op] * 2)
            if option_num == -1:
                option_num = 2
            else:
                assert option_num == 1 + 1, option_num

        option_num = min(option_num, self.max_option_num)

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.MAX_LENGTH,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        labels_true = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        labels_false = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        labels = torch.cat([labels_true, labels_false], dim=1).reshape(-1)
        assert labels.size(0) == input_ids.size(0)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res


class WikiPathDatasetCollatorSeq2Seq:
    def __init__(self, max_input_length: int, max_output_length: int, tokenizer: str, sent_sample_ratio: float = 0.4,
                 use_template: bool = True):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        if isinstance(self.tokenizer, T5Tokenizer):
            self.sep_token = "<extra_id_0>"
        elif isinstance(self.tokenizer, BartTokenizer):
            self.sep_token = "<s>"
        else:
            raise RuntimeError("Unsupported tokenizer {}".format(tokenizer.__class__.__name__))
        self.sent_sample_ratio = sent_sample_ratio
        self.prefix = "generate logically consistent deductions: "
        self.templates = [
            "What would happen if {}?",
            "If {} then what would happen?",
        ]
        self.use_template = use_template

        transformers.logging.set_verbosity_error()

    def __call__(self, batch):

        inputs = []
        outputs = []
        for b in batch:
            example = b["example"]
            ctx = example["context"]
            condition = example["condition"] if "condition" in example else example["positive"]
            if self.use_template:
                condition = random.choice(self.templates).format(condition)

            output_sent_num = int(len(ctx) * self.sent_sample_ratio)
            output_sent_num = max(output_sent_num, 1)

            sent_ids = list(range(len(ctx)))
            output_sent_ids = set(random.sample(sent_ids, output_sent_num))
            output_sents = [ctx[i] for i in output_sent_ids]
            input_sents = [ctx[i] for i in sent_ids if i not in output_sent_ids]

            inputs.append(' '.join([self.prefix] + input_sents + [condition]))
            outputs.append(self.sep_token.join(output_sents))

        model_inputs = self.tokenizer(inputs, return_tensors="pt",
                                      padding=PaddingStrategy.LONGEST,
                                      truncation=True, max_length=self.max_input_length)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(outputs, return_tensors="pt",
                                    padding=PaddingStrategy.LONGEST,
                                    truncation=True, max_length=self.max_output_length)["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs


class WikiPathDatasetCollatorSeq2SeqV2:
    def __init__(self, max_input_length: int, max_output_length: int, tokenizer: str, use_template: bool = True):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        if isinstance(self.tokenizer, T5Tokenizer):
            self.sep_token = "<extra_id_0>"
        elif isinstance(self.tokenizer, BartTokenizer):
            self.sep_token = "<s>"
        else:
            raise RuntimeError("Unsupported tokenizer {}".format(tokenizer.__class__.__name__))
        self.prefix = "generate logically consistent deductions: "
        self.templates = [
            "What would happen if {}?",
            "If {} then what would happen?",
        ]
        self.use_template = use_template

        transformers.logging.set_verbosity_error()

    def __call__(self, batch):

        inputs = []
        outputs = []
        for b in batch:
            example = b["example"]
            ctx = example["context"]
            output = example["condition"] if "condition" in example else example["positive"]

            if self.use_template:
                question = ctx[-1]
                question = random.choice(self.templates).format(question)
                context = ctx[:-1]
                assert len(context) > 0

                inputs.append(' '.join([self.prefix] + context + [question]))
            else:
                inputs.append(' '.join([self.prefix] + ctx))
            outputs.append(output)

        model_inputs = self.tokenizer(inputs, return_tensors="pt",
                                      padding=PaddingStrategy.LONGEST,
                                      truncation=True, max_length=self.max_input_length)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(outputs, return_tensors="pt",
                                    padding=PaddingStrategy.LONGEST,
                                    truncation=True, max_length=self.max_output_length)["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs


class WikiPathDatasetCollatorRelSeqGenV1(WikiPathDatasetCollatorWithContext):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4, swap: bool = False):
        super().__init__(max_seq_length, tokenizer, mlm_probability, max_option_num, swap)

    def __call__(self, batch):
        rel_decode = []
        for b in batch:
            rel_decode.append(b["rel_labels"])

        max_input_len = max(map(len, rel_decode))
        invalid = 0
        decoder_input_ids = torch.zeros(len(batch), max_input_len, dtype=torch.long).fill_(-1)
        for b, b_decoder_inputs in enumerate(rel_decode):
            decoder_input_ids[b, :len(b_decoder_inputs)] = torch.tensor(b_decoder_inputs, dtype=torch.long)
            if b_decoder_inputs[0] == -1:
                invalid += 1

        res = super().__call__(batch)
        res["rel_labels"] = decoder_input_ids
        res["invalid_path"] = invalid

        return res

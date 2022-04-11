import random
from collections import Counter
from typing import Tuple, Optional

import torch
import transformers
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.V5")


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

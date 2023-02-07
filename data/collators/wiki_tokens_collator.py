import collections
import os.path
import random
from typing import Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger
from data.collators.wiki import WikiPathDatasetV5

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

            if "rel_labels" in batch[exp_id]:
                wo_noise_num = len([span for span in exp["h_spans"][0] if len(span) > 0])
                assert wo_noise_num == len(batch[exp_id]["rel_labels"]) - 1 or batch[exp_id]["rel_labels"] == [-1], (
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
        # if b_decoder_inputs[0] == -1:
        #     invalid += 1
        if all(x == -1 for x in b_decoder_inputs):
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

    if "-1" in path_s_ids:
        wo_noise_s_num = [tmp for tmp in path_s_ids if tmp != "-1"]
        assert len(path_s_ids_order) == len(wo_noise_s_num) == (len(rel_ids) - 2), (path_s_ids_order, path_s_ids, rel_ids,
                                                                                    len(example["h_spans"][0]))
    else:
        assert len(path_s_ids_order) == len(path_s_ids) == (len(rel_ids) - 2), (path_s_ids_order, path_s_ids, rel_ids,
                                                                                len(example["h_spans"][0]))

    s_id2rel_id = {s_id: rel_id for s_id, rel_id in zip(path_s_ids_order, rel_ids[:-2])}
    rel_ids_input_order = [s_id2rel_id[s_id] if s_id != "-1" else -1 for s_id in path_s_ids] + rel_ids[-2:]
    # assert len(rel_ids_input_order) == len(rel_ids)
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


class WikiPathDatasetCollatorRelGenV1(WikiPathTokensDatasetCollator):
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, rmlm_probability: float = 1.0):
        super().__init__(max_seq_length, tokenizer, mlm_probability)
        # self.tokenizer.add_tokens([f"<entity_{i}>" for i in range(10)])
        self.rmlm_probability = rmlm_probability

    def __call__(self, batch):
        # Generating the rationale part of some sentence given the other parts of the logical circle.
        examples = [b["example"] for b in batch]
        sentence_spans = [exp["sentence_spans"][0] for exp in examples]
        h_spans = [exp["h_spans"][0] for exp in examples]
        t_spans = [exp["t_spans"][0] for exp in examples]

        inputs = super().__call__(batch)

        input_ids = inputs["input_ids"][:, 0].clone()
        attention_mask = inputs["attention_mask"][:, 0].clone()

        # labels = input_ids.clone()

        non_mask_field = torch.ones(input_ids.size(), dtype=torch.int)

        for b in range(len(examples)):
            target_sentence_id = random.choice(list(range(len(sentence_spans[b]))))
            assert len(sentence_spans[b]) == len(h_spans[b]) == len(t_spans[b]), (len(sentence_spans), len(h_spans[b]), len(t_spans[b]))

            sent_span = sentence_spans[b][target_sentence_id]
            # h_span = h_spans[b][target_sentence_id]
            # t_span = t_spans[b][target_sentence_id]
            # logger.info(input_ids.size(), sent_span, h_span, t_span)
            # input_ids[b, sent_span[0]: sent_span[1]] = self.tokenizer.mask_token_id
            non_mask_field[b, sent_span[0]: sent_span[1]] = 0
            for h_span in h_spans[b][target_sentence_id]:
                # input_ids[b, h_span[0]: h_span[1]] = labels[b, h_span[0]: h_span[1]].clone()
                non_mask_field[b, h_span[0]: h_span[1]] = 1
            for t_span in t_spans[b][target_sentence_id]:
                # input_ids[b, t_span[0]: t_span[1]] = labels[b, t_span[0]: t_span[1]].clone()
                non_mask_field[b, t_span[0]: t_span[1]] = 1

            # print(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[b])))

        input_ids, labels = mask_tokens(self.tokenizer, input_ids, non_mask_field, self.rmlm_probability)
        # print(f"++++++++++++++++++++++++++++++++")
        # print(self.tokenizer.convert_tokens_to_ids(self.tokenizer.convert_ids_to_tokens(input_ids[0])))
        # print(f"=================================")

        # mlm_mask = input_ids == self.tokenizer.mask_token_id
        # labels[~mlm_mask] = -1

        inputs.update({
            "rmlm_input_ids": input_ids,
            "rmlm_attention_mask": attention_mask,
            "rmlm_labels": labels,
        })
        return inputs


class WikiPathDatasetCollatorRelSeqGenMEV1(WikiPathDatasetCollatorRelSeqGenV1):
    def __call__(self, batch):
        me_input = [b["example"]["me_input"] for b in batch]
        me_candidate_ent = [b["example"]["me_candidate_ent"] for b in batch]
        me_label = [b["example"]["me_target_id"] for b in batch]

        me_tokenizer_outputs = self.tokenizer(me_input, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")

        max_candidate_num = max(map(len, me_candidate_ent))
        max_ent_len = 0
        me_label_ids = []
        me_candidate_ent_input_ids = []
        for b in range(len(batch)):
            b_ent_input_ids = []
            for i, (ent_id, ent_str) in enumerate(me_candidate_ent[b].items()):
                ent_tokens = self.tokenizer.tokenize(ent_str)
                ent_input_ids = self.tokenizer.convert_tokens_to_ids(ent_tokens)
                b_ent_input_ids.append(ent_input_ids)
                max_ent_len = max(max_ent_len, len(ent_input_ids))

                if ent_id == me_label[b]:
                    me_label_ids.append(i)
            me_candidate_ent_input_ids.append(b_ent_input_ids)

        assert len(me_label_ids) == len(batch)

        me_ent_input_ids_tensor = torch.zeros(len(batch), max_candidate_num, max_ent_len, dtype=torch.long).fill_(
            self.tokenizer.pad_token_id)
        me_ent_mask = torch.zeros(len(batch), max_candidate_num)
        for b in range(len(batch)):
            for ent_id, ent_input_ids in enumerate(me_candidate_ent_input_ids[b]):
                me_ent_input_ids_tensor[b, ent_id, :len(ent_input_ids)] = torch.tensor(ent_input_ids, dtype=torch.long)
                me_ent_mask[b, ent_id] = 1

        results = super().__call__(batch)
        results.update({
            "me_input_ids": me_tokenizer_outputs["input_ids"],
            "me_attention_mask": me_tokenizer_outputs["attention_mask"],
            "me_ent_input_ids": me_ent_input_ids_tensor,
            "me_ent_mask": me_ent_mask,
            "me_labels": torch.tensor(me_label_ids, dtype=torch.long),
        })
        return results


def _init_():
    global roberta_tokenizer
    roberta_tokenizer = AutoTokenizer.from_pretrained('pretrained-models/roberta-large')


def _extract_text_single_example(example):
    example_id, example = example
    examples = []
    indices = []
    for op_id, op in enumerate(example["tokens"]):
        if isinstance(op, list):
            token_ids = roberta_tokenizer.convert_tokens_to_ids(op)
        else:
            tokens = roberta_tokenizer.tokenize(op)
            token_ids = roberta_tokenizer.convert_tokens_to_ids(tokens)
        text = roberta_tokenizer.decode(token_ids, skip_special_tokens=True)

        if op_id == 0:
            index = f"{example_id}-{example['orig_id']}-pos-{op_id}"
            if "h" in example:
                index = index + "-aug"
            else:
                index = index + "-nom"
        else:
            index = f"{example_id}-{example['orig_id']}-neg-{op_id}"
            if "h" in example:
                index = index + "-aug"
            else:
                index = index + "-nom"
        examples.append(text)
        indices.append(index)
    return examples, indices


def extracting_text_entrance(file_path, num_workers: int = 48):
    all_examples, raw_texts = torch.load(file_path)

    examples = []
    indices = []
    with Pool(num_workers, initializer=_init_) as p:
        _results = list(tqdm(
            p.imap(_extract_text_single_example, list(enumerate(all_examples)), chunksize=32),
            total=len(all_examples),
            desc="Reading examples"
        ))
    for res in _results:
        examples.extend(res[0])
        indices.extend(res[1])
    return examples, indices


class GPTInference(Dataset):
    def __init__(self, file_path, tokenizer, num_workers: int = 48):
        super().__init__()

        cache_path = f"{file_path}_kkkkktmp"
        if os.path.exists(cache_path):
            self.examples, self.indices = torch.load(cache_path)
        else:
            examples, indices = extracting_text_entrance(file_path, num_workers=num_workers)
            torch.save((examples, indices), cache_path)
            self.examples = examples
            self.indices = indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {
            "text": self.examples[index],
            "index": self.indices[index],
        }


class GPTInferenceCollator:
    def __init__(self, tokenizer_path, max_seq_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.max_seq_length = max_seq_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        indices = [b["index"] for b in batch]

        inputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                truncation=TruncationStrategy.LONGEST_FIRST,
                                max_length=self.max_seq_length,
                                return_tensors="pt")
        labels = inputs["input_ids"].clone()

        inputs["labels"] = labels
        inputs["meta_data"] = {
            "index": indices
        }
        return inputs


# ===================== Relation-based graph-to-text generation

class WikiReconstructDataset(WikiPathDatasetV5):
    def __init__(self, examples, raw_texts, remove_cf_data_decoding: bool = False):
        super().__init__(examples, raw_texts)

        self.orig_id2exp_id = collections.defaultdict(list)
        for exp_id, exp in enumerate(examples):
            self.orig_id2exp_id[exp["orig_id"]].append(exp_id)

    def __getitem__(self, index):
        item = super().__getitem__(index)

        example = item["example"]
        exp_id = item["index"]
        pair_exp_ids = list(set(self.orig_id2exp_id[example["orig_id"]]) - {exp_id})
        if len(pair_exp_ids):
            pair_example = self.examples[random.choice(pair_exp_ids)]
        else:
            pair_example = example
        item["paired_example"] = pair_example

        return item


class WikiPathTokensDatasetCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, decoder_tokenizer: str,
                 decoder_max_seq_length: int = 1024, mlm_probability: float = 0.15):
        self.max_seq_length = max_seq_length
        self.decoder_max_seq_length = decoder_max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.decoder_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer, use_fast=False)
        self.mlm_probability = mlm_probability

        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

    def __call__(self, batch):
        examples = [b["example"] for b in batch]
        paired_examples = [b["paired_example"] for b in batch]
        texts = [b["text"] for b in batch]

        max_seq_length = 0
        for exp in examples:
            max_seq_length = max(max_seq_length, len(exp["tokens"][0]))

        input_ids = torch.zeros(len(examples), max_seq_length, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(len(examples), max_seq_length, dtype=torch.long)
        for exp_id, exp in enumerate(examples):
            input_ids[exp_id, :len(exp["tokens"][0])] = torch.tensor(self.tokenizer.convert_tokens_to_ids(exp["tokens"][0]),
                                                                     dtype=torch.long)
            attention_mask[exp_id, :len(exp["tokens"][0])] = 1

        output_texts = []
        entity_mentions = []
        for exp_id, exp in enumerate(paired_examples):
            text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(exp["tokens"][0]), skip_special_tokens=True)
            output_texts.append(text)

            exp_ent_mentions = []
            for ent_pair in exp["ent_mentions"]:
                exp_ent_mentions.append(" ".join(ent_pair))
            entity_mentions.append(" ".join(exp_ent_mentions))

        decoding_inputs = self.decoder_tokenizer(entity_mentions, text_target=output_texts,
                                                 padding=PaddingStrategy.LONGEST,
                                                 truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.decoder_max_seq_length,
                                                 return_tensors="pt")

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = mask_tokens(self.tokenizer, mlm_input_ids, mlm_probability=self.mlm_probability)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
        }
        for k, v in decoding_inputs.items():
            inputs[f"decoder_{k}"] = v
        return inputs


class WikiPathTokensDatasetCollatorMC(WikiPathTokensDatasetCollator):
    def __call__(self, batch):
        inputs = super().__call__(batch)

        examples = [b["example"] for b in batch]

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

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        inputs["labels"] = torch.zeros(len(examples), dtype=torch.long)
        return inputs


def processing_examples_w_spans(examples, tokenizer: PreTrainedTokenizer):
    option_num = len(examples[0]["tokens"])
    max_seq_length = 0
    for exp in examples:
        for op_id, op in enumerate(exp["tokens"]):
            if isinstance(op, str):
                assert op_id > 0
                exp["tokens"][op_id] = tokenizer.tokenize(op)
            max_seq_length = max(max_seq_length, len(exp["tokens"][op_id]))

    output_texts = []
    entity_mentions = []
    input_ids = torch.zeros(len(examples), option_num, max_seq_length, dtype=torch.long).fill_(tokenizer.pad_token_id)
    attention_mask = torch.zeros(len(examples), option_num, max_seq_length, dtype=torch.long)
    for exp_id, exp in enumerate(examples):
        b_texts = []
        for op_id, op_tokens in enumerate(exp["tokens"]):
            op_token_ids = tokenizer.convert_tokens_to_ids(op_tokens)

            input_ids[exp_id, op_id, :len(op_tokens)] = torch.tensor(op_token_ids, dtype=torch.long)
            attention_mask[exp_id, op_id, :len(op_tokens)] = 1

            text = tokenizer.decode(op_token_ids, skip_special_tokens=True)
            b_texts.append(text)

        output_texts.append(b_texts)

        exp_ent_mentions = []
        for ent_pair in exp["ent_mentions"]:
            exp_ent_mentions.append(" ".join(ent_pair))
        entity_mentions.append(" ".join(exp_ent_mentions))

    max_sent_num = 0
    max_sent_len = 0
    for exp in examples:
        max_sent_num = max(max_sent_num, len(exp["h_spans"][0]))  # [batch_size, option_num, sent_num, span_num]
        max_sent_len = max(max_sent_len, max(map(lambda x: x[1] - x[0], exp["sentence_spans"][0])))

    # sent_token_index = torch.zeros(len(examples), max_sent_num, max_sent_len, dtype=torch.long)
    # sent_token_mask = torch.zeros(len(examples), max_sent_num, max_sent_len)
    # tagging_labels = torch.zeros(len(examples), max_seq_length, dtype=torch.long)
    h_span_marks = torch.zeros(len(examples), max_sent_num, max_seq_length)
    t_span_marks = torch.zeros(len(examples), max_sent_num, max_seq_length)
    entity_pair_mask = torch.zeros(len(examples), max_sent_num)
    for exp_id, exp in enumerate(examples):
        for sent_id, sent_h_spans in enumerate(exp["h_spans"][0]):
            token_num = 0

            for span in sent_h_spans:
                # tagging_labels[exp_id, span[0]: span[1]] = 1
                h_span_marks[exp_id, sent_id, span[0]: span[1]] = 1
                token_num += span[1] - span[0]

            if token_num:
                h_span_marks[exp_id, sent_id] = h_span_marks[exp_id, sent_id] * 1.0 / token_num
            else:
                entity_pair_mask[exp_id, sent_id] = 1

        # if "rel_labels" in batch[exp_id]:
        #     wo_noise_num = len([span for span in exp["h_spans"][0] if len(span) > 0])
        #     assert wo_noise_num == len(batch[exp_id]["rel_labels"]) - 1 or batch[exp_id]["rel_labels"] == [-1], (
        #         exp["h_spans"][0], batch[exp_id]["rel_labels"])

        for sent_id, sent_t_spans in enumerate(exp["t_spans"][0]):
            token_num = 0
            # assert len(sent_t_spans) >= 1, (sent_id, exp["t_spans"][0])

            for span in sent_t_spans:
                # tagging_labels[exp_id, span[0]: span[1]] = 1
                t_span_marks[exp_id, sent_id, span[0]: span[1]] = 1
                token_num += span[1] - span[0]

            if token_num:
                t_span_marks[exp_id, sent_id] = t_span_marks[exp_id, sent_id] * 1.0 / token_num
            else:
                entity_pair_mask[exp_id, sent_id] = 1

        # for sent_id, sent_span in enumerate(exp["sentence_spans"][0]):
        #     _sent_len = sent_span[1] - sent_span[0]
        #     sent_token_index[exp_id, sent_id, :_sent_len] = torch.arange(sent_span[0], sent_span[1], dtype=torch.long)
        #     sent_token_mask[exp_id, sent_id, :_sent_len] = 1

        # assert len(exp["t_spans"][0]) == len(exp["h_spans"][0]) == len(exp["sentence_spans"][0])
        return input_ids, attention_mask, output_texts, entity_mentions, h_span_marks, t_span_marks, entity_pair_mask


class WikiPathTokensDatasetCollatorContiguous:
    def __init__(self, max_seq_length: int, tokenizer: str, decoder_tokenizer: str,
                 decoder_max_seq_length: int = 1024, mlm_probability: float = 0.15, add_mc: bool = False):
        self.max_seq_length = max_seq_length
        self.decoder_max_seq_length = decoder_max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.decoder_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer, use_fast=False)
        self.mlm_probability = mlm_probability
        self.add_mc = add_mc

        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

    def __call__(self, batch):
        examples = [b["example"] for b in batch]
        paired_examples = [b["paired_example"] for b in batch]
        texts = [b["text"] for b in batch]

        input_ids, attention_mask, output_texts, _, h_span_marks, t_span_marks, \
            entity_pair_mask = processing_examples_w_spans(examples, self.tokenizer)

        p_input_ids, p_attention_mask, p_output_texts, _, p_h_span_marks, p_t_span_marks, \
            p_entity_pair_mask = processing_examples_w_spans(paired_examples, self.tokenizer)

        p_input_ids = p_input_ids[:, 0]
        p_attention_mask = p_attention_mask[:, 0]
        p_output_texts = [x[0] for x in p_output_texts]
        if not self.add_mc:
            input_ids = input_ids[:, 0]
            attention_mask = attention_mask[:, 0]
            output_texts = [x[0] for x in output_texts]

        decoding_outputs = self.decoder_tokenizer(output_texts,
                                                  padding=PaddingStrategy.LONGEST,
                                                  truncation=TruncationStrategy.LONGEST_FIRST,
                                                  max_length=self.decoder_max_seq_length,
                                                  return_tensors="pt")
        p_decoding_outputs = self.decoder_tokenizer(p_output_texts,
                                                    padding=PaddingStrategy.LONGEST,
                                                    truncation=TruncationStrategy.LONGEST_FIRST,
                                                    max_length=self.decoder_max_seq_length,
                                                    return_tensors="pt")

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.LONGEST,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = mask_tokens(self.tokenizer, mlm_input_ids, mlm_probability=self.mlm_probability)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "p_input_ids": p_input_ids,
            "p_attention_mask": p_attention_mask,
            "decoding_input_ids": decoding_outputs["input_ids"],
            "decoding_attention_mask": decoding_outputs["attention_mask"],
            "p_decoding_input_ids": p_decoding_outputs["input_ids"],
            "p_decoding_attention_mask": p_decoding_outputs["attention_mask"],
            "h_span_marks": h_span_marks,
            "t_span_marks": t_span_marks,
            "entity_pair_mask": entity_pair_mask,
            "p_h_span_marks": p_h_span_marks,
            "p_t_span_marks": p_t_span_marks,
            "p_entity_pair_mask": p_entity_pair_mask,
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels,
        }
        if self.add_mc:
            inputs["labels"] = torch.zeros(len(examples), dtype=torch.long)
        return inputs

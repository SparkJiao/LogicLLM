import collections
import copy
import glob
import json
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, PreTrainedTokenizer
import transformers

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def rearrange_example_id(examples):
    id_set = {}
    for exp in examples:
        orig_id = str(exp["id"])
        if orig_id not in id_set:
            id_set[orig_id] = 0

        exp["id"] = orig_id + "_" + str(id_set[orig_id])
        id_set[orig_id] = id_set[orig_id] + 1
    return examples


class SeperatorInterface:
    h_s_sep = "<e1>"
    h_e_sep = "</e1>"
    t_s_sep = "<e2>"
    t_e_sep = "</e2>"


def process_erica_para_word(example, tokenizer: PreTrainedTokenizer, max_seq_length: int, mlm: bool):
    entities = example["vertexSet"]
    sentences = example["sents"]

    sent_len_offset = []
    words = []
    for sent in sentences:
        sent_len_offset.append(len(words))
        words.extend(sent)

    all_mentions = []
    for ent in entities:
        for mention in ent:
            offset = sent_len_offset[mention["sent_id"]]
            mention["pos"] = [mention["pos"][0] + offset, mention["pos"][1] + offset]
            all_mentions.append(mention)

    sorted_mentions = sorted(all_mentions, key=lambda x: x["pos"][0])
    for i in range(len(sorted_mentions)):
        if i == 0:
            continue
        if sorted_mentions[i]["pos"][0] < sorted_mentions[i - 1]["pos"][1]:
            logger.warning(f"Bad instance checked: {sorted_mentions[i - 1]}\t{sorted_mentions[i]}")

    tokens = [tokenizer.cls_token]
    last_e = 0
    entity_subword_spans = defaultdict(list)
    reach_limit_flag = False
    for mention in sorted_mentions:
        word_s, word_e = mention["pos"]
        if word_s > last_e:
            span = " ".join(words[last_e: word_s])
            if last_e > 0:
                span = " " + span
            tokens.extend(tokenizer.tokenize(span))

        subword_s = len(tokens)
        entity_span = " ".join(words[word_s: word_e])
        if word_s > 0:
            entity_span = " " + entity_span
        entity_tokens = tokenizer.tokenize(entity_span)
        if mlm:
            tokens.extend([tokenizer.mask_token] * len(entity_tokens))
        else:
            tokens.extend(entity_tokens)

        if len(tokens) >= max_seq_length - 1:  # `-1` for **sep token**.
            tokens = tokens[:(max_seq_length - 1)]
            reach_limit_flag = True

        subword_e = len(tokens)

        entity_subword_spans[mention["id"]].append((subword_s, subword_e))

        last_e = word_e

        if reach_limit_flag:
            break

    if last_e < len(words) and not reach_limit_flag:
        tokens.extend(tokenizer.tokenize(" " + " ".join(words[last_e:])))
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:(max_seq_length - 1)]

    tokens.append(tokenizer.sep_token)

    return tokenizer.convert_tokens_to_ids(tokens), entity_subword_spans, sorted_mentions


class ERICATextDataset(Dataset):
    def __init__(self, file_path: str, max_seq_length: int, tokenizer: PreTrainedTokenizer, mlm: bool = False):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files
        self.samples = []
        for _file in input_files:
            logger.info(f"Reading from {_file}")
            self.samples.extend(json.load(open(_file)))
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.special_token_num = self.tokenizer.num_special_tokens_to_add()
        self.mlm = mlm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.process_single_item(self.samples[index]) + (index,)

    def process_single_item(self, sample):
        entities = sample["vertexSet"]
        # relations = sample["labels"]
        sentences = sample["sents"]

        sent_len_offset = []
        words = []
        context = ""
        word_to_char_span = []
        for sent in sentences:
            sent_len_offset.append(len(words))
            words.extend(sent)

            for word in sent:
                if len(context) > 0:
                    context = context + " "
                s = len(context)
                context = context + word
                e = len(context)
                word_to_char_span.append((s, e))

            assert len(word_to_char_span) == len(words)

        all_mentions = []
        for ent in entities:
            for mention in ent:
                offset = sent_len_offset[mention["sent_id"]]
                mention["pos"] = [mention["pos"][0] + offset, mention["pos"][1] + offset]
                all_mentions.append(mention)

        sorted_mentions = sorted(all_mentions, key=lambda x: x["pos"][0])
        for i in range(len(sorted_mentions)):
            if i == 0:
                continue
            if sorted_mentions[i]["pos"][0] < sorted_mentions[i - 1]["pos"][1]:
                logger.warning(f"Bad instance checked: {sorted_mentions[i - 1]}\t{sorted_mentions[i]}")

        # Obtain the entity positions in subword sequence.
        tokens = []
        last_s = 0
        entity_subword_spans = defaultdict(list)
        for mention in sorted_mentions:
            char_s, _ = word_to_char_span[mention["pos"][0]]
            _, char_e = word_to_char_span[mention["pos"][1] - 1]
            if char_s > last_s:
                tokens.extend(self.tokenizer.tokenize(context[last_s: char_s]))
            ent_span_s = len(tokens)
            ent_tokens = self.tokenizer.tokenize(context[char_s: char_e])
            if self.mlm:
                tokens.extend([self.tokenizer.mask_token] * len(ent_tokens))
            else:
                tokens.extend(ent_tokens)
            ent_span_e = len(tokens)

            if ent_span_e < (self.max_seq_length - self.special_token_num):
                # entity_subword_spans.append((ent_span_s, ent_span_e))
                entity_subword_spans[mention["id"]].append((ent_span_s, ent_span_e))

            last_s = ent_span_e  # FIXME: This is also a bug!!!!!!! （2023/1/25. The entity hidden states extracted from ERICA maybe influenced.)

        return self.tokenizer.convert_tokens_to_ids(tokens), entity_subword_spans, sorted_mentions


class ERICATextDatasetV2(Dataset):
    def __init__(self, file_path: str, max_seq_length: int, tokenizer: PreTrainedTokenizer, mlm: bool = False):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files

        self.indices = []
        self.samples = []
        for _file in input_files:
            logger.info(f"Reading from {_file}")
            data = json.load(open(_file))
            file_name = _file.split("/")[-1]
            for item_id, item in enumerate(data):
                self.indices.append((file_name, item_id))
                self.samples.append(item)

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.special_token_num = self.tokenizer.num_special_tokens_to_add()
        self.mlm = mlm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return process_erica_para_word(self.samples[index], self.tokenizer, self.max_seq_length, self.mlm) + (self.indices[index],)


class ERICASentenceDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, cached_path: str = None, keep_original: bool = False):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files

        self.tokenizer = tokenizer

        self.h_s_sep = "<e1>"
        self.h_e_sep = "</e1>"
        self.t_s_sep = "<e2>"
        self.t_e_sep = "</e2>"
        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])
        self.keep_original = keep_original

        if cached_path is None or not os.path.exists(cached_path):
            samples = []
            for _file in input_files:
                logger.info(f"Reading from {_file}")
                samples.extend(json.load(open(_file)))

            self.examples = []
            for sample in tqdm(samples):
                self.examples.extend(self.process_single_para(sample))

            del samples

            if cached_path is not None:
                logger.info(f"Saving data to {cached_path}")
                torch.save(self.examples, cached_path)
        else:
            logger.info(f"Loading data from {cached_path}")
            self.examples = torch.load(cached_path)
            if isinstance(self.examples[0], tuple) and isinstance(self.examples[0][0], list):
                for exp_id, exp in enumerate(tqdm(self.examples)):
                    self.examples[exp_id] = (" ".join(exp[0]), exp[1])

    def __getitem__(self, index):
        # print(self.examples[index])
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def annotate_entity(self, sent, mention1, mention2):
        if mention1["pos"][0] > mention2["pos"][0]:
            mention2, mention1 = mention1, mention2
        assert mention1["pos"][1] <= mention2["pos"][0]

        text = ""
        prefix = sent[:mention1["pos"][0]]
        if len(prefix):
            text = text + " ".join(prefix)

        text = text + " " + " ".join([self.h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [self.h_e_sep])

        inter = sent[mention1["pos"][1]: mention2["pos"][0]]
        if len(inter):
            text = text + " " + " ".join(inter)

        text = text + " " + " ".join([self.t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [self.t_e_sep])

        text = text + " " + " ".join(sent[mention2["pos"][1]:])
        return text

    def process_single_para(self, sample):
        entities = sample["vertexSet"]
        sentences = sample["sents"]

        results = []
        for idx, ent1 in enumerate(entities):
            for ent2 in entities[(idx + 1):]:
                if ent1[0]["id"] == ent2[0]["id"]:
                    continue
                for mention1 in ent1:
                    for mention2 in ent2:
                        if mention1["sent_id"] == mention2["sent_id"]:
                            aug = self.annotate_entity(sentences[mention1["sent_id"]], mention1, mention2)
                            if self.keep_original:
                                results.append((sentences[mention1["sent_id"]], aug))
                            else:
                                results.append(aug)
        return results


class ERICASentenceFilterDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, cached_path: str = None):
        if os.path.exists(file_path):
            input_files = [file_path]
        else:
            input_files = list(glob.glob(file_path))
            input_files = sorted(input_files)
        assert input_files, input_files

        self.tokenizer = tokenizer

        self.h_s_sep = "<e1>"
        self.h_e_sep = "</e1>"
        self.t_s_sep = "<e2>"
        self.t_e_sep = "</e2>"
        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])

        if cached_path is None or not os.path.exists(cached_path):
            samples = []
            for _file in input_files:
                logger.info(f"Reading from {_file}")
                samples.extend(json.load(open(_file)))

            self.examples = []
            for sample in tqdm(samples):
                self.examples.extend(self.process_single_para(sample))

            del samples

            if cached_path is not None:
                logger.info(f"Saving data to {cached_path}")
                torch.save(self.examples, cached_path)
        else:
            logger.info(f"Loading data from {cached_path}")
            self.examples = torch.load(cached_path)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def annotate_entity(self, sent, mention1, mention2):
        if mention1["pos"][0] > mention2["pos"][0]:
            mention2, mention1 = mention1, mention2
        assert mention1["pos"][1] <= mention2["pos"][0]

        text = ""
        prefix = sent[:mention1["pos"][0]]
        if len(prefix):
            text = text + " ".join(prefix)

        text = text + " " + " ".join([self.h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [self.h_e_sep])

        inter = sent[mention1["pos"][1]: mention2["pos"][0]]
        if len(inter):
            text = text + " " + " ".join(inter)

        text = text + " " + " ".join([self.t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [self.t_e_sep])

        text = text + " " + " ".join(sent[mention2["pos"][1]:])
        return text

    def process_single_para(self, sample):
        entities = sample["vertexSet"]
        sentences = sample["sents"]

        sent_ent_num = Counter()
        for ent in entities:
            for mention in ent:
                sent_ent_num[mention["sent_id"]] += 1

        results = []
        for idx, ent1 in enumerate(entities):
            for ent2 in entities[(idx + 1):]:
                if ent1[0]["id"] == ent2[0]["id"]:
                    continue
                for mention1 in ent1:
                    for mention2 in ent2:
                        if mention1["sent_id"] == mention2["sent_id"]:
                            if sent_ent_num[mention1["sent_id"]] <= 2:
                                continue
                            aug = self.annotate_entity(sentences[mention1["sent_id"]], mention1, mention2)
                            results.append((" ".join(sentences[mention1["sent_id"]]), aug))

        return results


def annotate_entity(cls, sent, mention1, mention2):
    if mention1["pos"][0] > mention2["pos"][0]:
        mention2, mention1 = mention1, mention2
    assert mention1["pos"][1] <= mention2["pos"][0]

    text = ""
    prefix = sent[:mention1["pos"][0]]
    if len(prefix):
        text = text + " ".join(prefix)

    text = text + " " + " ".join([cls.h_s_sep] + sent[mention1["pos"][0]: mention1["pos"][1]] + [cls.h_e_sep])

    inter = sent[mention1["pos"][1]: mention2["pos"][0]]
    if len(inter):
        text = text + " " + " ".join(inter)

    text = text + " " + " ".join([cls.t_s_sep] + sent[mention2["pos"][0]: mention2["pos"][1]] + [cls.t_e_sep])

    text = text + " " + " ".join(sent[mention2["pos"][1]:])
    return text


class WikiPathSentenceDataset(Dataset, SeperatorInterface):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        data = pickle.load(open(file_path, "rb"))["examples"]
        self.sentences = []
        self.indices = []

        for item in tqdm(data, total=len(data)):
            selected_sentences = item["selected_sentences"]
            pos = item["pos"]

            for exp_id, exp in selected_sentences.items():
                self.sentences.append(annotate_entity(self, exp["sent"], exp["ent"][exp["h"]][0], exp["ent"][exp["t"]][0]))
                self.indices.append(f"{item['id']}-path-{exp_id}")

            for exp_id, exp in enumerate(pos):
                self.sentences.append(annotate_entity(self, exp["sent"], exp["ent"][exp["h"]][0], exp["ent"][exp["t"]][0]))
                self.indices.append(f"{item['id']}-pos-{exp_id}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.indices[index]


def extract_path_sent_follow_order(example):
    path = example["path"]
    rel_s_ids = []
    for i, item in enumerate(path):
        if i == 0:
            continue
        rel_s_ids.append(item[1])
    return rel_s_ids


class WikiPathSentenceConditionDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, cache_path: str = None):
        if cache_path is not None and os.path.exists(cache_path):
            self.sentences, self.indices = torch.load(cache_path)
        else:
            data = pickle.load(open(file_path, "rb"))["examples"]

            sentences: List[Dict[str, Any]] = []
            indices = []
            for item in tqdm(data, total=len(data)):
                for s_id, s in item["selected_sentences"].items():
                    sent = " ".join(s["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(s)
                        indices.append(f"{item['id']}-path-{s_id}")

                for pos_id, pos in enumerate(item["pos"]):
                    sent = " ".join(pos["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(pos)
                        indices.append(f"{item['id']}-pos-{pos_id}")

            if cache_path is not None:
                torch.save((sentences, indices), cache_path)

            self.sentences = sentences
            self.indices = indices

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        h = random.choice(sent["ent"][sent["h"]])
        t = random.choice(sent["ent"][sent["t"]])
        flag = True
        if h["pos"][0] > t["pos"][0]:
            h, t = t, h
            flag = False
        assert h["pos"][1] <= t["pos"][0]

        tokens = [self.tokenizer.cls_token]
        h_span = []
        t_span = []

        tokens.extend(self.tokenizer.tokenize(" ".join(sent["sent"][:h["pos"][0]])))
        h_span.append(len(tokens))

        h_mention = " ".join(sent["sent"][h["pos"][0]: h["pos"][1]])
        tokens.extend(self.tokenizer.tokenize(" " + h_mention))
        h_span.append(len(tokens) - 1)

        tokens.extend(self.tokenizer.tokenize(" " + " ".join(sent["sent"][h["pos"][1]: t["pos"][0]])))
        t_span.append(len(tokens))

        t_mention = " ".join(sent["sent"][t["pos"][0]: t["pos"][1]])
        tokens.extend(self.tokenizer.tokenize(" " + t_mention))
        t_span.append(len(tokens) - 1)

        tokens.extend(self.tokenizer.tokenize(" " + " ".join(sent["sent"][t["pos"][1]:])))
        tokens.append(self.tokenizer.sep_token)

        if not flag:
            h_span, t_span = t_span, h_span
            h_mention, t_mention = t_mention, h_mention

        text = " ".join(sent["sent"])

        return {
            "tokens": tokens,
            "text": text,
            "h_span": h_span,
            "t_span": t_span,
            "h": h_mention,
            "t": t_mention,
            "index": self.indices[index]
        }


# def annotate_para_entity_mask(, tokenizer: PreTrainedTokenizer):
#     # FIXME: 这里这个方法不太对，ERICA的训练方式决定了还是要从paragraph的角度去求entity pair的表示。考虑在paragraph的角度上计算隐表示的距离。
#     #   但或许也不太好操作，因为一个paragraph太短，因此根据距离取出来的对应的句子很可能是重复的。
#
#     entity_mentions = []
#     for ent_id in [ent1, ent2]:
#         for mention_id, e in enumerate(sent["ent"][ent_id]):
#             entity_mentions.append((e, ent_id, mention_id))
#     entity_mentions = sorted(entity_mentions, key=lambda x: x[0]["pos"][0])
#     for m_id, m in enumerate(entity_mentions):
#         if m_id == 0:
#             continue
#         assert m[0]["pos"][0] >= entity_mentions[m_id - 1][0]["pos"][1]
#
#     tokens = [tokenizer.cls_token]
#     words = sent["sent"]
#     _s = 0
#     ent_spans = collections.defaultdict(list)
#     for m in entity_mentions:
#         if m[0]["pos"][0] > _s:
#             span = " ".join(words[_s: m[0]["pos"][0]])
#             if _s > 0:
#                 span = " " + span
#             tokens.extend(tokenizer.tokenize(span))
#
#         entity = " ".join(words[m[0]["pos"][0]: m[0]["pos"][1]])
#         if m[0]["pos"][0] > 0:
#             entity = " " + entity
#         entity_tokens = tokenizer.tokenize(entity)
#
#         pos = [len(tokens)]
#         tokens.extend(entity_tokens)
#         pos.append(len(tokens))
#         pos = tuple(pos)
#
#         ent_spans[m[1]].append(pos)
#
#         _s = m[0]["pos"][1]
#
#     if _s < len(words):
#         tokens.extend(tokenizer.tokenize(" " + " ".join(words[_s:])))
#     tokens.append(tokenizer.sep_token)
#
#     return tokens, ent_spans[ent1], ent_spans[ent2]


class WikiSentenceMultipleConditionDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int,
                 cache_path: str = None, evaluating: bool = False, remove_path: bool = False,
                 entity_mask_ratio: float = 0.0, entity_shuffle: bool = False, counterfactual_ratio: float = 0.0):
        self.evaluating = evaluating
        self.entity_mask_ratio = entity_mask_ratio
        self.entity_shuffle = entity_shuffle
        self.remove_path = remove_path
        self.counterfactual_ratio = counterfactual_ratio
        logger.info(f"{self.evaluating}\t{self.remove_path}\t{self.entity_mask_ratio}\t{self.entity_shuffle}\t"
                    f"{self.counterfactual_ratio}")

        if cache_path is not None and os.path.exists(cache_path):
            self.sentences, self.indices = torch.load(cache_path)
        else:
            data = pickle.load(open(file_path, "rb"))["examples"]

            sentences: List[Dict[str, Any]] = []
            indices = []
            for item in tqdm(data, total=len(data)):
                for s_id, s in item["selected_sentences"].items():
                    sent = " ".join(s["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(s)
                        indices.append(f"{item['id']}-path-{s_id}")

                for pos_id, pos in enumerate(item["pos"]):
                    sent = " ".join(pos["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(pos)
                        indices.append(f"{item['id']}-pos-{pos_id}")

                for r_id, rest_s in item["rest_sentences"].items():
                    sent = " ".join(rest_s["sent"])
                    if len(tokenizer.tokenize(sent)) + 2 <= max_seq_length:
                        sentences.append(rest_s)
                        indices.append(f"{item['id']}-rest-{r_id}")

            if cache_path is not None:
                torch.save((sentences, indices), cache_path)

            self.sentences = sentences
            self.indices = indices

        sentences, indices = [], []
        for sent, idx in zip(self.sentences, self.indices):
            if len(sent["ent"]) <= 1:
                continue
            if self.remove_path and "rest" not in idx:
                if len(sent["ent"]) <= 2:
                    continue
            if self.evaluating and "rest" in idx:
                continue
            sentences.append(sent)
            indices.append(idx)
        self.sentences = sentences
        self.indices = indices

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]

        entity_mentions = []
        entities = {}
        if self.evaluating:
            entity_dict = {
                sent["h"]: copy.deepcopy(sent["ent"][sent["h"]]),
                sent["t"]: copy.deepcopy(sent["ent"][sent["t"]]),
            }
        else:
            entity_dict = copy.deepcopy(sent["ent"])

        for ent_id, ent in entity_dict.items():
            tmp = []
            for mention_id, e in enumerate(ent):
                entity_mentions.append((e, ent_id, mention_id))
                tmp.append(
                    {
                        "mention": " ".join(sent["sent"][e["pos"][0]: e["pos"][1]]),
                        "id": e["id"],
                    }
                )
            entities[ent_id] = tmp
        entity_mentions = sorted(entity_mentions, key=lambda x: x[0]["pos"][0])
        for m_id, m in enumerate(entity_mentions):
            if m_id == 0:
                continue
            assert m[0]["pos"][0] >= entity_mentions[m_id - 1][0]["pos"][1]

        tokens = [self.tokenizer.cls_token]
        words = sent["sent"]
        _s = 0
        masked_mention_num = 0
        for m in entity_mentions:
            # Before entity
            if m[0]["pos"][0] > _s:
                span = " ".join(words[_s: m[0]["pos"][0]])
                if _s > 0:
                    span = " " + span
                tokens.extend(self.tokenizer.tokenize(span))

            # Process entity
            entity = " ".join(words[m[0]["pos"][0]: m[0]["pos"][1]])
            if m[0]["pos"][0] > 0:
                entity = " " + entity
            entity_tokens = self.tokenizer.tokenize(entity)

            if self.entity_mask_ratio:
                _r = random.random()
                if _r < self.entity_mask_ratio:
                    _num_subwords = len(entity_tokens)
                    entity_tokens = [self.tokenizer.mask_token] * _num_subwords
                    masked_mention_num += 1

            pos = [len(tokens), -1]
            tokens.extend(entity_tokens)
            pos[1] = len(tokens)
            pos = tuple(pos)

            entities[m[1]][m[2]]["pos"] = pos

            # _s = pos[1]  # FIXED
            _s = m[0]["pos"][1]

        if _s < len(words):
            tokens.extend(self.tokenizer.tokenize(" " + " ".join(words[_s:])))
        tokens.append(self.tokenizer.sep_token)

        if self.entity_shuffle:
            ent_ids = list(entities.keys())
            random.shuffle(ent_ids)
            entities = {ent_id: entities[ent_id] for ent_id in ent_ids}

        if self.counterfactual_ratio:
            _r = random.random()
            if _r < self.counterfactual_ratio:
                mentions = [ent[0]["mention"] for ent in entities.values()]
                random.shuffle(mentions)
                # Only modify the first mention since we only use the first one as the inputs during reconstruction
                for i, ent_id in enumerate(entities.keys()):
                    entities[ent_id][0]["mention"] = mentions[i]

                # Reorganize the output text to be reconstructed.
                _s = 0
                new_words = []
                for m in entity_mentions:
                    # Before entity
                    new_words.extend(words[_s: m[0]["pos"][0]])

                    new_words.append(entities[m[1]][0]["mention"])
                    _s = m[0]["pos"][1]

                if _s < len(words):
                    new_words.extend(words[_s:])

                words = new_words

        # print(" ".join(words))
        # print(self.tokenizer.convert_tokens_to_string(tokens))
        # print("=====================")

        return {
            "text": " ".join(words),
            "tokens": tokens,
            "entities": entities,
            "index": self.indices[index],
            "h": sent["h"] if "h" in sent else -1,
            "t": sent["t"] if "t" in sent else -1,
            "entity_mask_num": masked_mention_num,
        }


class ERICATextCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        batch_ids, batch_entity_spans, batch_ent_mentions, indices = list(zip(*batch))

        batch_input_ids = []
        batch_attention_mask = []
        for ids in batch_ids:
            _inputs = self.tokenizer.prepare_for_model(ids, padding="longest", truncation=True, max_length=self.max_seq_length)
            batch_input_ids.append(_inputs["input_ids"])
            batch_attention_mask.append(_inputs["attention_mask"])

        batch_max_seq_len = max(map(len, batch_input_ids))
        input_ids = torch.zeros(len(batch_ids), batch_max_seq_len, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros(len(batch_ids), batch_max_seq_len, dtype=torch.long)
        for i, (ids, mask) in enumerate(zip(batch_input_ids, batch_attention_mask)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = torch.tensor(mask, dtype=torch.long)

        max_ent_num = max(map(len, batch_entity_spans))

        # ent_span_index = torch.zeros(len(batch_ids), max_ent_num, 2, dtype=torch.long)
        # ent_mask = torch.zeros(len(batch_ids), max_ent_num, dtype=torch.int)
        # for i, spans in enumerate(batch_entity_spans):
        #     ent_span_index[i, len(spans)] = torch.tensor(spans, dtype=torch.long)
        #     ent_mask[i, len(spans)] = 1

        ent_seq_mapping = torch.zeros(len(batch_ids), max_ent_num, batch_max_seq_len)
        for i, b_ent_spans in enumerate(batch_entity_spans):
            for j, ent_mentions in enumerate(b_ent_spans.values()):
                for ent_mention in ent_mentions:
                    # FIXME: A bug here!!!!
                    #   For previous version, i.e., ERICATextDataset, the span does not conut the special tokens.
                    #   So we need to use `ent_mention[0] + 1` and `ent_mention[1] + 1` instead.
                    ent_seq_mapping[i, j, ent_mention[0]: ent_mention[1]] = 1.0 / len(ent_mentions) / (ent_mention[1] - ent_mention[0])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ent_seq_mapping": ent_seq_mapping,
            "meta_data": {
                "index": indices,
                "entity_spans": batch_entity_spans,
                "entity_mentions": batch_ent_mentions,
            }
        }


class ERICATextCollatorV2(ERICATextCollator):
    def __call__(self, batch):
        batch_ids, batch_entity_spans, batch_ent_mentions, indices = list(zip(*batch))

        batch_size = len(batch)

        max_seq_length = max(map(len, batch_ids))
        batch_input_ids = torch.zeros(batch_size, max_seq_length, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        batch_attention_mask = torch.zeros(batch_size, max_seq_length, dtype=torch.long)
        for b_id, ids in enumerate(batch_ids):
            batch_input_ids[b_id, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            batch_attention_mask[b_id, :len(ids)] = 1

        max_ent_num = max(map(len, batch_entity_spans))

        ent_seq_mapping = torch.zeros(batch_size, max_ent_num, max_seq_length)
        for i, b_ent_spans in enumerate(batch_entity_spans):
            assert isinstance(b_ent_spans, dict), type(b_ent_spans)
            for j, ent_mentions in enumerate(b_ent_spans.values()):
                total_token_num = 0
                for span in ent_mentions:
                    ent_seq_mapping[i, j, span[0]: span[1]] = 1.0
                    total_token_num += span[1] - span[0]
                if total_token_num == 0:
                    logger.warning(f"Empty entity span list: {ent_mentions}")
                else:
                    ent_seq_mapping[i, j] /= total_token_num

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "ent_seq_mapping": ent_seq_mapping,
            "meta_data": {
                "index": indices,
                "entity_spans": batch_entity_spans,
                "entity_mentions": batch_ent_mentions,
            }
        }


class ERICASentenceCollator:
    def __init__(self, tokenizer, max_seq_length: int, partial_optim: bool = False, no_original: bool = False):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_length = max_seq_length

        self.h_s_sep = "<e1>"
        self.h_e_sep = "</e1>"
        self.t_s_sep = "<e2>"
        self.t_e_sep = "</e2>"
        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])
        self.partial_optim = partial_optim
        self.no_original = no_original

    def __call__(self, batch):
        if isinstance(batch[0], tuple) and isinstance(batch[0][1], str):
            if not self.no_original:
                orig_batch = [b[0] for b in batch]
            else:
                orig_batch = None
            batch = [b[1] for b in batch]
        else:
            orig_batch = None

        if transformers.__version__[:4] == "4.24":
            model_inputs = self.tokenizer(batch, text_target=batch,
                                          padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        else:
            model_inputs = self.tokenizer(batch, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(batch, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
                model_inputs["labels"] = labels["input_ids"]

        labels = model_inputs["labels"]  # [batch, seq_length]
        special_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids([
            self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep]), dtype=torch.long)[None, None, :]  # [1, 1, 4]
        tgt_mask = labels.unsqueeze(-1) == special_tokens
        assert tgt_mask.size() == (labels.size(0), labels.size(1), 4)
        ent_mark_mask = tgt_mask.sum(dim=-1).bool()
        model_inputs["ent_mark_mask"] = ent_mark_mask

        if self.partial_optim:
            tgt_mask = tgt_mask.sum(dim=-1).cumsum(dim=1)  # [0, 0, (1, 1, 2), 2, (3, 3, 4), 4, 4, 4]
            partial_mask = (tgt_mask == 1) | (tgt_mask == 3)
            for b_id, b_tgt_mask in enumerate(tgt_mask):
                b_tgt_mask = b_tgt_mask.tolist()
                if 2 not in b_tgt_mask or 1 not in b_tgt_mask:
                    partial_mask[b_id, :] = 1
                    continue
                else:
                    idx2 = b_tgt_mask.index(2)
                    partial_mask[b_id, idx2] = 1

                if 3 not in b_tgt_mask or 4 not in b_tgt_mask:
                    partial_mask[b_id, :] = 0
                else:
                    idx4 = b_tgt_mask.index(4)
                    partial_mask[b_id, idx4] = 1
            partial_mask = ~partial_mask
            model_inputs["partial_mask"] = partial_mask

        if orig_batch is not None:
            orig_outputs = self.tokenizer(orig_batch, padding="longest", truncation=True, max_length=self.max_seq_length,
                                          return_tensors="pt")
            model_inputs["extra_input_ids"] = orig_outputs["input_ids"]
            model_inputs["extra_attention_mask"] = orig_outputs["attention_mask"]

        return model_inputs


class WikiPathInferenceCollator(SeperatorInterface):
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_length = max_seq_length

        self.tokenizer.add_tokens([self.h_s_sep, self.h_e_sep, self.t_s_sep, self.t_e_sep])

    def __call__(self, batch):
        sentences, indices = list(zip(*batch))

        model_inputs = self.tokenizer(batch, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        model_inputs["meta_data"] = {"indices": indices}

        return model_inputs


class WikiPathSentenceConditionCollator:
    def __init__(self, enc_tokenizer: str, dec_tokenizer: str, max_seq_length: int, reverse: bool = False):
        self.enc_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(enc_tokenizer)
        self.dec_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(dec_tokenizer)
        self.max_seq_length = max_seq_length
        self.reverse = reverse

    def __call__(self, batch):
        seq_len = max(map(lambda x: len(x["tokens"]), batch))

        input_ids = torch.zeros(len(batch), seq_len, dtype=torch.int)
        attention_mask = torch.zeros(len(batch), seq_len, dtype=torch.int)
        h_span = []
        t_span = []
        indices = []
        decoder_inputs = []
        decoder_outputs = []
        for b_id, b in enumerate(batch):
            input_ids[b_id, :len(b["tokens"])] = torch.tensor(self.enc_tokenizer.convert_tokens_to_ids(b["tokens"]), dtype=torch.int)
            attention_mask[b_id, :len(b["tokens"])] = 1

            h_span.append(b["h_span"])
            t_span.append(b["t_span"])
            indices.append(b["index"])

            if not self.reverse:
                decoder_inputs.append(b["h"] + self.dec_tokenizer.sep_token + b["t"])
                decoder_outputs.append(b["text"])
            else:
                decoder_inputs.append(b["text"])
                decoder_outputs.append(b["h"] + self.dec_tokenizer.sep_token + b["t"])
                # print(decoder_inputs[-1])
                # print(decoder_outputs[-1])
                # print("------------------------------------")

        h_span = torch.tensor(h_span, dtype=torch.long)
        t_span = torch.tensor(t_span, dtype=torch.long)

        decoder_inputs = self.dec_tokenizer(decoder_inputs, text_target=decoder_outputs,
                                            padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_input_attention_mask": decoder_inputs["attention_mask"],
            "decoder_output_ids": decoder_inputs["labels"],
            "h_span": h_span,
            "t_span": t_span,
            "meta_data": {
                "indices": indices,
            }
        }


class WikiSentenceMultipleConditionCollator:
    def __init__(self, enc_tokenizer: str, dec_tokenizer: str, max_seq_length: int, reverse: bool = False, remove_path: bool = False,
                 entity_pair_dropout: float = 0.0):
        self.enc_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(enc_tokenizer)
        self.dec_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(dec_tokenizer)
        self.max_seq_length = max_seq_length
        self.reverse = reverse
        self.remove_path = remove_path
        self.entity_pair_dropout = entity_pair_dropout

    def __call__(self, batch):
        seq_len = max(map(lambda x: len(x["tokens"]), batch))
        entity_num = max(map(lambda x: len(x["entities"]), batch))

        input_ids = torch.zeros(len(batch), seq_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), seq_len, dtype=torch.int)
        entity_spans = torch.zeros(len(batch), entity_num, seq_len)
        relation_indices = []
        indices = []
        decoder_inputs = []
        decoder_outputs = []
        masked_mention_num = 0
        for b_id, b in enumerate(batch):
            input_ids[b_id, :len(b["tokens"])] = torch.tensor(self.enc_tokenizer.convert_tokens_to_ids(b["tokens"]), dtype=torch.int)
            attention_mask[b_id, :len(b["tokens"])] = 1

            for ent_id, ent in enumerate(b["entities"].values()):
                cnt = 0
                for e in ent:
                    entity_spans[b_id, ent_id, e["pos"][0]: e["pos"][1]] = 1
                    cnt += e["pos"][1] - e["pos"][0]
                entity_spans[b_id, ent_id] /= cnt

            indices.append(b["index"])

            triplets = []
            b_rel_indices = []
            for ent_id_1, ent1 in enumerate(b["entities"].values()):
                for ent_id_2, ent2 in enumerate(list(b["entities"].values())[(ent_id_1 + 1):]):
                    if self.remove_path and ((ent1[0]["id"] == b["h"] and ent2[0]["id"] == b["t"]) or (
                            ent1[0]["id"] == b["t"] and ent2[0]["id"] == b["h"])):
                        continue
                    triplets.append(f"{ent1[0]['mention']} {self.dec_tokenizer.mask_token} {ent2[0]['mention']}")
                    b_rel_indices.append((ent_id_1, ent_id_1 + 1 + ent_id_2))

            if self.entity_pair_dropout:
                _remove_num = int(round(len(triplets) * self.entity_pair_dropout))
                _kept_num = len(triplets) - _remove_num
                if _kept_num >= 1:
                    _kept_idx = random.sample(list(range(len(triplets))), _kept_num)
                    triplets = [triplets[i] for i in _kept_idx]
                    b_rel_indices = [b_rel_indices[i] for i in _kept_idx]
                    assert len(b_rel_indices)

            relation_indices.append(b_rel_indices)

            if not self.reverse:
                decoder_inputs.append(self.dec_tokenizer.sep_token.join(triplets))
                decoder_outputs.append(b["text"])
            else:
                # decoder_inputs.append(b["text"])
                # decoder_outputs.append(self.dec_tokenizer.sep_token.join([tri.replace(self.dec_tokenizer.mask_token, "")
                #                                                           for tri in triplets]))
                raise RuntimeError("`rel_emb_index` was not adjust for `self.reverse == True`.")

            masked_mention_num += b.pop("entity_mask_num")

        max_relation_num = max(map(len, relation_indices))
        rel_ent_index = torch.zeros(len(batch), max_relation_num, 2, dtype=torch.long)
        for b_id, b_rel_indices in enumerate(relation_indices):
            rel_ent_index[b_id, :len(b_rel_indices)] = torch.tensor(b_rel_indices, dtype=torch.long)

        decoder_inputs = self.dec_tokenizer(decoder_inputs, text_target=decoder_outputs,
                                            padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")

        rel_inputs = decoder_inputs["input_ids"] if not self.reverse else decoder_inputs["labels"]
        sep_mask = rel_inputs == self.dec_tokenizer.sep_token_id
        rel_emb_index = torch.cumsum(sep_mask.to(dtype=torch.long), dim=1)
        rel_emb_index[rel_emb_index >= max_relation_num] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_input_attention_mask": decoder_inputs["attention_mask"],
            "decoder_output_ids": decoder_inputs["labels"],
            "entity_spans": entity_spans,
            "rel_ent_index": rel_ent_index,
            "rel_emb_index": rel_emb_index,
            "entity_mask_num": masked_mention_num,
            "meta_data": {
                "indices": indices,
            }
        }

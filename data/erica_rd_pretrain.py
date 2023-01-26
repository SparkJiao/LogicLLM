import collections
import copy
import json
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List
import glob
from torch.utils.data import Dataset

import torch
from torch.distributions.geometric import Geometric
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.collators.wiki import WikiPathDatasetV6wPatternPair, WikiPathDatasetRelGenerateV1, WikiPathDatasetRelGenerateV3
from data.collators.wiki_structure_pair import WikiPathDatasetV6wPair
from data.data_utils import get_all_permutation, get_sep_tokens
from general_util.logger import get_child_logger

_tokenizer: PreTrainedTokenizer


def convert_single_example(example):
    example_id, example = example

    entities = example["vertextSet"]
    relations = example["labels"]
    sentences = example["sents"]

    relations = [{
        "h": entities[rel["h"]][0]["id"],
        "t": entities[rel["t"]][0]["id"],
        "r": rel["r"]
    } for rel in relations]

    sent_len_offset = []
    words = []
    for sent in sentences:
        sent_len_offset.append(len(words))
        words.extend(sent)

    entity_dict = collections.defaultdict(list)
    entity_mentions = []
    for ent in entities:
        for ent_mention in ent:
            sent_offset = sent_len_offset[ent_mention["sent_id"]]
            ent_mention["pos"] = [ent_mention["pos"][0] + sent_offset, ent_mention["pos"][1] + sent_offset]

            entity_mentions.append((copy.deepcopy(ent_mention), ent_mention["id"], len(entity_dict[ent_mention["id"]])))
            entity_dict[ent_mention["id"]].append(copy.deepcopy(ent_mention))

    entity_mentions = sorted(entity_mentions, key=lambda x: x[0]["pos"][0])
    for m_id, m in enumerate(entity_mentions):
        if m_id == 0:
            continue
        assert m[0]["pos"][0] >= entity_mentions[m_id - 1][0]["pos"][1]

    tokens = [_tokenizer.cls_token]
    _s = 0
    for m in entity_mentions:
        if m[0]["pos"][0] > _s:
            span = " ".join(words[_s: m[0]["pos"][0]])
            if _s > 0:
                span = " " + span
            tokens.extend(_tokenizer.tokenize(span))

        entity = " ".join(words[m[0]["pos"][0]: m[0]["pos"][1]])
        if m[0]["pos"][0] > 0:
            entity = " " + entity
        entity_tokens = _tokenizer.tokenize(entity)

        pos = [len(tokens)]
        tokens.extend(entity_tokens)
        pos.append(len(tokens))

        entity_dict[m[1]][m[2]]["pos"] = pos

        _s = m[0]["pos"][1]

    if _s < len(words):
        tokens.extend(_tokenizer.tokenize(" " + " ".join(words[_s:])))
    tokens.append(_tokenizer.sep_token)

    return {
        "id": example_id,
        "tokens": tokens,
        "relations": relations,
        "entity": entity_dict
    }


def _init_(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def convert_example_to_features(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int = 512,
                                num_workers: int = 48):
    if os.path.exists(file_path):
        input_files = [file_path]
    else:
        input_files = list(glob.glob(file_path))

    raw_examples = []
    for _file_id, _file in enumerate(input_files):
        raw_examples.extend(json.load(open(_file)))

    with Pool(num_workers, initializer=_init_, initargs=(tokenizer,)) as p:
        _annotate = partial(convert_single_example)
        _results = list(tqdm(
            p.imap(_annotate, list(enumerate(raw_examples)), chunksize=32),
            total=len(raw_examples),
            desc="converting raw examples into features"
        ))

    all_examples = []
    all_relation_labels = []
    for res in _results:
        if len(res["tokens"]) > max_seq_length:
            continue
        relations = res.pop("relations")
        for rel in relations:
            rel["example_id"] = len(all_examples)
            all_relation_labels.append(rel)

        all_examples.append(res)

    return ...


class RelationDiscriminationDataset(Dataset):
    def __init__(self, examples, relation_labels):
        self.examples = examples
        self.rel2pair = ...

    def __len__(self):
        return len(self.relation_labels)

    def __getitem__(self, index):
        ...




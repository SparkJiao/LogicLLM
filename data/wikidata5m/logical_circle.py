import argparse
import collections
import json
import os
import sys
import random
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.models.bert.tokenization_bert import whitespace_tokenize

# pwd = os.getcwd()
# f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".." + os.path.sep + "..")

# sys.path.append(f_pwd)
sys.path.append("../../")

from data.data_utils import span_chunk, tokenizer_get_name

_tokenizer: PreTrainedTokenizer
_id2ent: Dict[str, List[str]]
_id2rel: Dict[str, List[str]]
_triplet2sent: Dict[str, List[str]]
_edge2rel: Dict[str, List[str]]


def init(tokenizer: PreTrainedTokenizer,
         id2ent, id2rel, triplet2sent, edge2rel):
    global _tokenizer
    global _id2ent
    global _id2rel
    global _triplet2sent
    global _edge2rel
    _tokenizer = tokenizer
    _id2ent = id2ent
    _id2rel = id2rel
    _triplet2sent = triplet2sent
    _edge2rel = edge2rel


class LogicCircleDataset(Dataset):
    def __init__(self, logical_circle: str, id2ent: str, id2rel: str, triplet2sent: str):
        super(self).__init__()

        self.logical_circle = json.load(open(logical_circle, 'r'))
        self.id2ent = json.load(open(id2ent, 'r'))
        self.id2rel = json.load(open(id2rel, 'r'))
        self.triplet2sent = json.load(open(triplet2sent, 'r'))

    def __iter__(self):
        pass

    def __len__(self):
        pass


def load_triplet2sent(triplet2sent: str):
    triplet_sent_ls = json.load(open(triplet2sent))

    triplet2sent = collections.defaultdict(list)
    for item in triplet_sent_ls:
        s, rel, t, corpus = item["s"], item["rel"], item["t"], item["corpus"]
        key = "\t".join([s, rel, t])
        triplet2sent[key].extend(corpus)

    return triplet2sent


def triplet2texts(s, rel, t, triplet2sent, id2ent, id2rel) -> List[Dict[str, str]]:
    key = "\t".join([s, rel, t])
    texts = []
    if key in triplet2sent:
        for item in triplet2sent[key]:
            s_alias = item["s"]
            t_alias = item["t"]
            text = item["text"]
            assert s_alias in text, (s_alias, text)
            assert t_alias in text, (t_alias, text)
            texts.append({
                "s": s,
                "t": t,
                "rel": rel,
                "s_alias": s_alias,
                "t_alias": t_alias,
                "rel_alias": "",
                "text": text,
            })

    s_alias = random.choice(id2ent[s])
    t_alias = random.choice(id2ent[t])
    rel_alias = random.choice(id2rel[rel])
    text = ' '.join([s_alias, rel_alias, t_alias])
    texts.append({
        "s": s,
        "t": t,
        "rel": rel,
        "s_alias": s_alias,
        "t_alias": t_alias,
        "rel_alias": rel_alias,
        "text": text
    })

    return texts


def circle2text_mlm(path: List[Tuple[str, str, str]],
                    # id2ent: Dict[str, List[str]],
                    # id2rel: Dict[str, List[str]],
                    # triplet2sent: Dict[str, List[str]],
                    # edge2rel: Dict[str, List[str]],
                    shuffle_sentence: bool = True):
    assert len(path) >= 2

    # Obtain the relation with the given entity pair
    s = path[0][0]
    t = path[-1][-1]
    key = f"{s}\t{t}"

    assert len(_edge2rel[key])
    rel = random.choice(_edge2rel[key])

    # Symbols to text
    context = [triplet2texts(*_triplet, _triplet2sent, _id2ent, _id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, _triplet2sent, _id2ent, _id2rel)

    # Sample a text group
    context = [random.choice(sent_dict) for sent_dict in context]
    anchor = random.choice(anchor)

    sentences = context + [anchor]
    if shuffle_sentence:
        random.shuffle(sentences)

    # Annotate the span (of entities) and obtain de indicating mask (for which word belongs to an entity)
    text = ""
    spans = []
    indicate_mask = []
    for sent_dict in sentences:
        tgt_spans = [sent_dict["s_alias"], sent_dict["t_alias"], ]
        if sent_dict["rel_alias"]:
            # Usually multiple words. So, pass.
            pass
        sent_spans, sent_indicate_mask = span_chunk(sent_dict["text"],
                                                    tgt_spans,
                                                    space_tokenize=True)
        spans.extend(sent_spans)
        indicate_mask.extend(sent_indicate_mask)
        text += sent_dict["text"]

    """
                    ================== MLM =====================
    - Whole word mask is needed to avoid information leakage (right?)
    - For each triplet, different processing procedure for entity and relation (may be implicit) may be needed.
     
    - For MLM, we can only pre-process the token mask for each word and annotate the scope of entities,
      so that the we can perform online dynamic whole word masking during training.
    
    - The significance of masking entity and relation:
        - For entity, entity masking requires the model to predict correct subject and object within a triplet
          based on the context understanding.
          If there are more than one sentence involving entity masking, it will also propose the challenge about
          intermediate relations within a reasoning path.
        - For relation, relation masking makes the model learn the correct mapping (computation) in reasoning.
    """

    flag, recovered_text, tokens, _, token2word_index = generate_whole_word_index_mapping(text,
                                                                                          _tokenizer,
                                                                                          spans)

    return flag, text, tokens, spans, token2word_index, indicate_mask


def generate_whole_word_index_mapping(text: str, tokenizer: PreTrainedTokenizer, spans: List[str] = None):
    """

    Note: If `text` is indeed an input pair, the separator token, e.g., `[SEP]`, should be inserted into the text first.
    """
    if spans is None:
        words = whitespace_tokenize(text)
    else:
        # Using spans at the input to support span masking, e.g., treat an entity (phrase) as a single word.
        words = spans

    tokens = []
    # Used for `torch.scatter` method
    # i.e., `token_mask = torch.scatter(word_mask, dim=0, index=token2word_index)`,
    # where `word_mask` is a 0/1 tensor indicating whether a word should be masked.
    # word_mask: [word_num], token2word_index: [token_num]
    token2word_index = []
    for idx, word in enumerate(words):
        if idx > 0:
            sub_tokens = tokenizer.tokenize(' ' + word)
        else:
            sub_tokens = tokenizer.tokenize(word)

        tokens.extend(sub_tokens)
        token2word_index.extend([idx] * len(sub_tokens))

    # Consistency check
    target_tokens = tokenizer.tokenize(text)
    recovered_text = tokenizer.convert_tokens_to_string(tokens)
    flag = True
    if target_tokens != tokens or recovered_text != text:
        print("Warning: Inconsistent tokenization: ")
        print(f"Original:\t{target_tokens}")
        print(f"Pre-tokenized:\t{tokens}")
        print(f"Original text:\t{text}")
        print(f"Recovered text:\t{recovered_text}")
        flag = False

    return flag, recovered_text, tokens, words, token2word_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="mlm", help="Options: mlm, ctr, seq2seq")
    parser.add_argument("--path", type=str)
    parser.add_argument("--id2ent", type=str)
    parser.add_argument("--id2rel", type=str)
    parser.add_argument("--triplet2sent", type=str)
    parser.add_argument("--edge2rel", type=str)
    parser.add_argument("--tokenizer", type=str)
    # parser.add_argument("--shuffle_sentence", )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_path = json.load(open(args.path))
    id2ent = json.load(open(args.id2ent))
    id2rel = json.load(open(args.id2rel))
    triplet2sent = load_triplet2sent(args.triplet2sent)
    edge2rel = json.load(open(args.edge2rel))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.debug:
        all_path = all_path[:1000]

    print(len(all_path))

    if args.mode == "mlm":
        with Pool(args.num_workers, initializer=init, initargs=(tokenizer, id2ent, id2rel, triplet2sent, edge2rel)) as p:
            _annotate = partial(circle2text_mlm)
            _results = list(tqdm(
                p.imap(_annotate, all_path, chunksize=32),
                total=len(all_path),
                desc="constructing dataset",
                dynamic_ncols=True,
            ))

        tokenizer_name = tokenizer_get_name(tokenizer)
        output_file = os.path.join(args.output_dir, f"logic_circle_data_v1_{tokenizer_name}_s{args.seed}_{args.mode}.json")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=False)

        json.dump(_results, open(output_file, 'w'))
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()

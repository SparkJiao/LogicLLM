from torch.utils.data import Dataset, DataLoader
import json
import random
from typing import List, Dict, Tuple
import collections
import argparse
import numpy as np
import torch
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers import PreTrainedTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from data.data_utils import find_span, span_chunk

_tokenizer: PreTrainedTokenizer


def init(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


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
    else:
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
                    id2ent: Dict[str, List[str]],
                    id2rel: Dict[str, List[str]],
                    triplet2sent: Dict[str, List[str]],
                    edge2rel: Dict[str, List[str]],
                    shuffle_sentence: bool = True):
    assert len(path) >= 2

    # Obtain the relation with the given entity pair
    s = path[0][0]
    t = path[-1][-1]
    key = f"{s}\t{t}"

    assert len(edge2rel[key])
    rel = random.choice(edge2rel[key])

    # Symbols to text
    context = [triplet2texts(*_triplet, triplet2sent, id2ent, id2rel) for _triplet in path]
    anchor = triplet2texts(s, rel, t, triplet2sent, id2ent, id2rel)

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
    parser.add_argument("--mode", type=str, help="Options: mlm, ctr, seq2seq")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

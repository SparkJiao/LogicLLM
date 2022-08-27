from torch.utils.data import Dataset, DataLoader
import json
import random
from typing import List, Dict, Tuple
import collections
import argparse
import numpy as np
import torch


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


def circle2text(path: List[Tuple[str, str, str]], id2ent: Dict[str, List[str]], id2rel: Dict[str, List[str]],
                triplet2sent: Dict[str, List[str]]):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, help="Options: mlm, ctr, seq2seq")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

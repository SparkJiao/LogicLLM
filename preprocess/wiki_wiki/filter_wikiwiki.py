import json
import argparse
import os
import torch
from nltk import sent_tokenize
from multiprocessing import Pool
from typing import Dict
from tqdm import tqdm

_kg: Dict[str, str]


def load_kg(kg_file):
    kg = {}
    with open(kg_file, "r") as f:
        line = f.readline()
        while line:
            tmp = line.strip().split("\t")
            s, rel, t = tmp
            kg["\t".join([s, t])] = rel
            line = f.readline()
    return kg


def process_line(line):
    item = json.loads(line)

    mention2entity = item["mention2entity"]
    entities = set([v[0] for v in mention2entity.values()])

    sentences = sent_tokenize(item["context"])

    edge_num = 0
    for ent1 in entities:
        for ent2 in entities:
            if ent1 == ent2:
                continue
            if f"{ent1}\t{ent2}" in _kg:
                edge_num += 1

    return item, len(sentences), edge_num, len(entities)


def init(kg: Dict[str, str]):
    global _kg
    _kg = kg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_file", type=str)
    parser.add_argument("--wikiwiki", type=str)
    parser.add_argument("--sent_num", type=int)
    parser.add_argument("--edge_num", type=int)
    parser.add_argument("--ent_num", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    kg = load_kg(args.kg_file)
    with open(args.wikiwiki) as f:
        lines = f.readlines()

    with Pool(args.num_workers, initializer=init, initargs=(kg,)) as p:
        results = list(tqdm(
            p.imap(process_line, lines, chunksize=32),
            desc="processing lines",
            total=len(lines)
        ))

    results = [res[0] for res in results if res[1] >= args.sent_num and res[2] >= args.edge_num and res[3] >= args.ent_num]

    output_file = f"wiki_wiki_s{args.sent_num}_e{args.edge_num}_ent{args.ent_num}.bin"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(args.output_dir, output_file)
    torch.save(results, output_file)

    print(f"Saving data to {output_file} with {len(results)} samples.")


if __name__ == '__main__':
    main()

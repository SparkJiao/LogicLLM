import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

from nltk import sent_tokenize
from tqdm import tqdm

id2ent: Dict[str, List[str]]
id2rel: Dict[str, List[str]]
ent2text: Dict[str, List[str]]


def init(id2ent_alias, id2rel_alias, ent2corpus):
    global id2ent
    global id2rel
    global ent2text
    id2ent = id2ent_alias
    id2rel = id2rel_alias
    ent2text = ent2corpus


def entity_iterator(ent_a, ent_b):
    for a_alias in id2ent[ent_a]:
        for b_alias in id2ent[ent_b]:
            yield a_alias, b_alias


def extract_common_sent(ent_a, ent_b, document):
    res = []
    sentences = sent_tokenize(document)
    for sent in sentences:
        for a_alias, b_alias in entity_iterator(ent_a, ent_b):
            if a_alias in sent and b_alias in sent:
                res.append({
                    "text": sent,
                    "s": a_alias,
                    "s_id": ent_a,
                    "t": b_alias,
                    "t_id": ent_b
                })
                break
    return res


def align_sentences_with_triplet(triplet: Tuple[str, ...]):
    s, rel, t = triplet
    s_document = ent2text[s]
    t_document = ent2text[t]

    if s not in id2ent or t not in id2ent:
        return {
            "s": s if s in id2ent else -1,
            "rel": rel,
            "t": t if t in id2ent else -1,
            "corpus": []
        }

    # FIXME: there maybe repeat sentences for <s, rel, t> and <t, rel, s>
    results = []
    for doc in s_document + t_document:
        results.extend(extract_common_sent(s, t, doc))

    return {
        "s": s,
        "rel": rel,
        "t": t,
        "corpus": results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str, required=True)
    parser.add_argument("--entity_vocab", type=str, required=True)
    parser.add_argument("--relation_vocab", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    id2ent_alias = {}
    ent2id = {}
    with open(args.entity_vocab, 'r') as f:
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            ent_id = items[0]
            ent_names = items[1:]
            if len(ent_names) == 0:
                continue
            id2ent_alias[ent_id] = ent_names
            for ent_name in ent_names:
                if ent_name in ent2id:
                    print(f"Repeat entity with name: {ent_name}")
                ent2id[ent_name] = ent_id

            line = f.readline()

    print(f"Entity amount: {len(id2ent_alias)}")

    id2rel_alias = {}
    rel2id = {}
    with open(args.relation_vocab, 'r') as f:
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            rel_id = items[0]
            rel_names = items[1:]
            if len(rel_names) == 0:
                continue
            id2rel_alias[rel_id] = rel_names
            for rel in rel_names:
                rel2id[rel] = rel_id

            line = f.readline()

    print(f"Relation amount: {len(id2rel_alias)}")

    ent2corpus = {}
    with open(args.corpus, 'r') as f:
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            ent_id = items[0]
            docs = items[1:]

            ent2corpus[ent_id] = docs

            line = f.readline()

    kg = []
    with open(args.kg, 'r') as f:
        line = f.readline()
        while line:
            tmp = line.strip().split('\t')
            assert len(tmp) == 3
            s, rel, t = tmp
            kg.append((s, rel, t))

            line = f.readline()

    overlooked = 0
    results = []
    text_num = 0
    with Pool(args.num_workers, initializer=init, initargs=(id2ent_alias, id2rel_alias, ent2corpus)) as p:
        _annotate = partial(align_sentences_with_triplet)
        _results = list(tqdm(
            p.imap(_annotate, kg, chunksize=32),
            total=len(kg),
            desc="aligning sentences"
        ))
        # _results = [res for res in _results if len(res["corpus"])]
        for res in _results:
            if len(res["corpus"]) and res["s"] != -1 and res["t"] != -1:
                results.append(res)
                text_num += len(res["corpus"])
            else:
                overlooked += 1

    print(f"Generated {len(results)} text-triplet pairs with {overlooked} samples overlooked.")
    print(f"Total {text_num} aligned sentences with {text_num / len(results)} per triplet.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)

    json.dump(id2ent_alias, open(os.path.join(args.output_dir, "id2ent.json"), "w"), indent=2)
    json.dump(ent2id, open(os.path.join(args.output_dir, "ent2id.json"), "w"), indent=2)
    json.dump(id2rel_alias, open(os.path.join(args.output_dir, "id2rel.json"), "w"), indent=2)
    json.dump(rel2id, open(os.path.join(args.output_dir, "rel2id.json"), "w"), indent=2)
    json.dump(results, open(os.path.join(args.output_dir, "triplet2sent.json"), "w"), indent=2)


if __name__ == '__main__':
    main()

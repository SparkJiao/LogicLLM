import copy
import glob
import json
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Tuple, Dict, Set, List

from tqdm import tqdm

"""
Version 1.0:
    ...
        
"""


def extract_entities_of_sent(sent_id, sent2ent, entities) -> Dict[str, List]:
    ent_ls = defaultdict(list)
    for e_id in sent2ent[sent_id]:
        for ent in entities:
            for pos_id, e in enumerate(ent):
                if e["id"] == e_id and e["sent_id"] == sent_id:
                    ent_ls[e_id].append(e)
    return ent_ls


def workflow(sample):
    sample_id, sample = sample

    entities = sample["vertexSet"]
    relations = sample["labels"]
    sentences = sample["sents"]

    ent2sent = defaultdict(set)
    for ent in entities:
        for e in ent:
            ent2sent[e["id"]].add(e["sent_id"])

    sent2ent = defaultdict(set)
    for ent in entities:
        for e in ent:
            sent2ent[e["sent_id"]].add(e["id"])

    rel2sent = defaultdict(list)
    for rel in relations:
        h = entities[rel["h"]][0]["id"]
        t = entities[rel["t"]][0]["id"]
        rel_id = rel["r"]

        s_ids = set()
        for h_m in entities[rel["h"]]:
            for t_m in entities[rel["t"]]:
                if h_m["sent_id"] == t_m["sent_id"]:
                    if h_m["sent_id"] in s_ids:
                        continue

                    s_ids.add(h_m["sent_id"])
                    rel2sent[rel_id].append({
                        "h": h_m["id"],
                        "t": t_m["id"],
                        "s_id": h_m["sent_id"],
                        "sent": sentences[h_m["sent_id"]],
                        "ent": extract_entities_of_sent(h_m["sent_id"], sent2ent, entities),
                        "id": sample_id,
                    })

    return rel2sent


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample', default=False, action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))

    file_suffix = f'.rel_group_v1.0.pkl'
    # if args.output_dir and not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    all_examples_cnt = 0
    all_rel2sent = defaultdict(list)
    for _file_id, _file in enumerate(input_files):
        samples = json.load(open(_file))

        processed_samples = []
        with Pool(args.num_workers) as p:
            _annotate = partial(workflow)
            _results = list(tqdm(
                p.imap(_annotate, list(enumerate(samples)), chunksize=32),
                total=len(samples),
                desc='processing samples'
            ))

        for _res in _results:
            if len(_res):
                for _rel_id, _rel_sents in _res.items():
                    for _rel_sent in _rel_sents:
                        _rel_sent["id"] = f"{_rel_sent['id']}_{_file_id}_{all_examples_cnt}"
                        all_examples_cnt += 1
                        all_rel2sent[_rel_id].append(_rel_sent)

    pickle.dump(all_rel2sent, open(args.output_file, 'wb'))

    print("Done.")

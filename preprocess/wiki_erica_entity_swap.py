import argparse
import glob
import json
import os.path

import torch
import copy
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool


def pos2str(ent_s, ent_e, tokens):
    return " ".join(tokens[ent_s: ent_e])


def extract_entities_of_sent(sent_id, sent2ent, entities) -> Dict[str, List]:
    ent_ls = defaultdict(list)
    for e_id in sent2ent[sent_id]:
        for ent in entities:
            for pos_id, e in enumerate(ent):
                if e["id"] == e_id and e["sent_id"] == sent_id:
                    ent_ls[e_id].append(e)
    return ent_ls


def generate_template(candidate):
    candidate = copy.deepcopy(candidate)  # Avoid in-place modification of `r["tgt"]`
    ent_to_rep = []

    for ent_id in candidate["ent"]:
        for r in candidate["ent"][ent_id]:
            # r["tgt"] = rep_pairs[ent_id]
            ent_to_rep.append(r)

    re = sorted(ent_to_rep, key=lambda x: x["pos"][0])
    # Non-overlapping check.
    for _tmp_id, _tmp in enumerate(re):
        if _tmp_id == 0:
            continue
        assert _tmp["pos"][0] >= re[_tmp_id - 1]["pos"][1]

    new_spans = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        if s > _last_e:
            new_spans.append(" ".join(candidate["sent"][_last_e: s]))
        r["span_index"] = len(new_spans)
        # new_spans.append(r["tgt"])
        new_spans.append(pos2str(r["pos"][0], r["pos"][1], candidate["sent"]))
        _last_e = e

    if _last_e < len(candidate["sent"]):
        new_spans.append(" ".join(candidate["sent"][_last_e:]))

    return {
        "spans": new_spans,
        "entity_replacement": re
    }


def mention_swap(template: Dict, i: int, j: int):
    assert i != j
    m1 = template["entity_replacement"][i]
    m2 = template["entity_replacement"][j]

    span1 = template["spans"][m1["span_index"]]
    span2 = template["spans"][m2["span_index"]]

    spans = copy.deepcopy(template["spans"])
    spans[m1["span_index"]] = span2
    spans[m2["span_index"]] = span1
    return " ".join(spans)


def process_single_sample(sample):
    sample_id, sample = sample

    entities = sample["vertexSet"]
    # relations = sample["labels"]
    sentences = sample["sents"]

    ent2sent = defaultdict(set)
    for ent in entities:
        for e in ent:
            ent2sent[e["id"]].add(e["sent_id"])

    sent2ent = defaultdict(set)
    for ent in entities:
        for e in ent:
            sent2ent[e["sent_id"]].add(e["id"])

    ent_id2item = defaultdict(list)
    for ent in entities:
        for e in ent:
            ent_id2item[e["id"]].append(e)

    results = []
    for s_id, s in enumerate(sentences):
        sent = {
            "sent": s,
            "ent": extract_entities_of_sent(s_id, sent2ent, entities),
        }
        template = generate_template(sent)
        # generate negative candidates
        if len(sent["ent"]) <= 2:
            negs = []
        else:
            mention_num = len(template["entity_replacement"])
            negs = []
            for i in range(mention_num):
                for j in range(i + 1, mention_num):
                    if template["entity_replacement"][i]["id"] == template["entity_replacement"][j]["id"]:
                        continue
                    # neg = mention_swap(template, i, j)
                    negs.append({
                        # "text": neg,
                        "id1": i,
                        "id2": j,
                    })
        results.append({
            "s_id": s_id,
            "template": template,
            "neg": negs
        })
    return {
        "id": sample_id,
        "templates": results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = sorted(list(glob.glob(args.input_file)))

    all_results = []
    for file_id, file in enumerate(files):
        samples = json.load(open(file))

        with Pool(args.num_workers) as p:
            results = list(tqdm(
                p.imap(process_single_sample, list(enumerate(samples)), chunksize=32),
                total=len(samples),
            ))
        for res in results:
            res["id"] = f"{res['id']}_{file_id}"
        all_results.extend(results)

    torch.save(all_results, args.output_file)


if __name__ == '__main__':
    main()

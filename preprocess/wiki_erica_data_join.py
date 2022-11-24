import argparse
import collections
import copy
import json
import os.path
from glob import glob

from tqdm import tqdm


def join_sample(sample_a, sample_b):
    sample_a = copy.deepcopy(sample_a)
    sample_b = copy.deepcopy(sample_b)

    sample_a_sent_num = len(sample_a["sents"])
    sample_a_eid2idx = {}
    for ent_id, ent in enumerate(sample_a["vertexSet"]):
        sample_a_eid2idx[ent[0]["id"]] = ent_id

    vertex_b = sample_b["vertexSet"]
    for ent in vertex_b:
        for mention in ent:
            mention["sent_id"] += sample_a_sent_num

    for ent in vertex_b:
        eid = ent[0]["id"]
        if eid in sample_a_eid2idx:
            sample_a["vertexSet"][sample_a_eid2idx[eid]].extend(ent)
        else:
            sample_a["vertexSet"].append(ent)

    sample_a["sents"].extend(sample_b["sents"])
    return sample_a


def enumerate_sample_pair(samples, limit: int = 3):
    results = []
    joined_sample_ids = set()

    ent_id2sample_ids = collections.defaultdict(list)
    for sample_id, sample in tqdm(enumerate(samples), total=len(samples)):
        for ent in sample["vertexSet"]:
            ent_id2sample_ids[ent[0]["id"]].append(sample_id)

    for sample_id1, sample1 in tqdm(enumerate(samples), total=len(samples)):
        if sample_id1 in joined_sample_ids:
            continue

        sample1_ent_ids = set([ent[0]["id"] for ent in sample1["vertexSet"]])
        cand_sample_ids = set()
        for ent in sample1["vertexSet"]:
            cand_sample_ids.update(ent_id2sample_ids[ent[0]["id"]])

        # for sample_id2, sample2 in enumerate(samples[(sample_id1 + 1):]):
        #     if sample_id1 + 1 + sample_id2 in joined_sample_ids:
        #         continue
        for sample_id2 in cand_sample_ids:
            if sample_id2 in joined_sample_ids:
                continue
            sample2 = samples[sample_id2]

            sample2_ent_ids = set([ent[0]["id"] for ent in sample2["vertexSet"]])
            if len(sample1_ent_ids & sample2_ent_ids) >= limit:
                results.append(join_sample(sample1, sample2))
                joined_sample_ids.add(sample_id1)
                joined_sample_ids.add(sample_id2)
                break

    for sample_id, sample in enumerate(samples):
        if sample_id not in joined_sample_ids:
            results.append(sample)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob(args.input_file))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for _file in input_files:
        samples = json.load(open(_file))
        results = enumerate_sample_pair(samples, limit=args.limit)

        _file_name = _file.split("/")[-1]
        output_file = os.path.join(args.output_dir, _file_name)
        json.dump(results, open(output_file, "w"))
        print(len(results))

    print("Done")


if __name__ == '__main__':
    main()
